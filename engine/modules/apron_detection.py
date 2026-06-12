"""
Butcher Apron Compliance Alarm
------------------------------
Type B module. Receives person bounding boxes and tracker IDs from the main
pipeline, then runs its own classification model on each person inside a
"butcher" zone. Raises an alarm when a butcher without an apron is observed
for more than a configurable number of consecutive frames, with a cooldown
between successive alarms.

Config example:
    {
        "type":               "apron_compliance",
        "name":               "apron_cam1",
        "regions": [
            {"type": "butcher", "label": "Butcher Area",
             "points": [[700, 200], [1500, 200], [1500, 800], [700, 800]]}
        ],
        "classifier_path":    "runs/classify/yolo26n_classification/weights/best.pt",
        "apron_conf":         0.7,
        "alert_duration_sec": 5.0,
        "cooldown_seconds":   30,
        "reference":          "foot",
        "scale":              1.0,
        "show_panel":         true
    }

get_data() output:
    {
        "Butcher Count":      int,
        "With Apron":         int,
        "Without Apron":      int,
        "No Apron Seconds":   float,
        "Status":             str,    # "ALARM" | "Normal"
        "Total Alarms":       int,
        "apron_alert":        bool    # main.py watches this to fire an alert
    }
"""

import os
import json
import time as _time
import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .base import BaseModule
from engine.shared_memory import GPU_LOCK

# ==================== CONFIG ====================

STATE_DIR                  = "state"
SAVE_INTERVAL_SEC          = 30
DEFAULT_ALERT_DURATION_SEC = 5.0
DEFAULT_COOLDOWN_SEC       = 30.0
DEFAULT_APRON_CONF         = 0.7

# Visual constants (BGR)
COLOR_BUTCHER  = (0, 200, 200)
COLOR_APRON    = (0, 200, 0)
COLOR_NO_APRON = (0, 0, 255)
COLOR_UNCERT   = (170, 170, 170)
COLOR_NEUTRAL  = (255, 150, 50)
COLOR_ALARM    = (0, 0, 255)
COLOR_PANEL    = (0, 140, 255)
COLOR_TEXT     = (255, 255, 255)

# Panel layout
PANEL_PADDING  = 12
PANEL_LINE_H   = 28
PANEL_ORIGIN   = (10, 10)

ZONE_LABELS    = {"butcher": "Butcher Area"}
ZONE_COLORS    = {"butcher": COLOR_BUTCHER}

# Classifier model outputs Turkish class names — map to English internally
CLASS_LABEL_MAP = {"onluklu": "apron", "onluksuz": "no_apron"}


# ==================== MODULE ====================

class ApronComplianceModule(BaseModule):

    def __init__(self, name: str,
                 regions: list = None,
                 classifier_path: str = None,
                 apron_conf: float = DEFAULT_APRON_CONF,
                 alert_duration_sec: float = DEFAULT_ALERT_DURATION_SEC,
                 cooldown_seconds: float = DEFAULT_COOLDOWN_SEC,
                 reference: str = "foot",
                 scale: float = 1.0,
                 show_panel: bool = True,
                 **_kwargs):
        # 1) config parameters
        self.name             = name
        self.classifier_path  = classifier_path
        self.apron_conf         = float(apron_conf)
        self.alert_duration_sec = float(alert_duration_sec)
        self.cooldown_seconds   = float(cooldown_seconds)
        self.reference        = reference
        self.scale            = float(scale)
        self.show_panel       = show_panel

        # pre-scale region polygons once
        self._zones = []
        for region in (regions or []):
            pts = (np.array(region["points"], dtype=np.float32) * self.scale)
            self._zones.append({
                "type":   region.get("type", "butcher"),
                "label":  region.get("label"),
                "points": pts.astype(np.int32),
            })

        # 2) own classification model (Type B)
        if not classifier_path:
            raise ValueError(f"[{name}] classifier_path is required")
        self._classifier  = YOLO(classifier_path)
        self._class_names = self._classifier.names
        print(f"[{name}] classifier loaded: {classifier_path}")
        print(f"[{name}] classes: {self._class_names}")

        # 3) internal state
        self._butcher_count       = 0
        self._with_apron_count    = 0
        self._without_apron_count = 0
        self._no_apron_since      = None
        self._no_apron_seconds    = 0.0
        self._last_alarm_time     = 0.0
        self._alert_active        = False
        self._alert_active_until  = 0.0
        self._should_alert        = False
        # daily-accumulated, persisted
        self._total_alarms        = 0
        self._last_reset_date     = None
        self._last_save_time      = 0.0

        # 4) draw() safety — must be None before first update()
        self._last_status     = None
        self._last_detections = []

        # 5) load persisted state (always last)
        self._load_state()

        # 6) ready
        print(f"✅ ApronComplianceModule ready [{name}]")

    # ==================== PERSISTENCE ====================

    def _state_path(self) -> str:
        return os.path.join(STATE_DIR, f"apron_compliance_{self.name}.json")

    def _save_state(self):
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            state = {
                "date":         (self._last_reset_date.isoformat()
                                 if self._last_reset_date else None),
                "total_alarms": self._total_alarms,
            }
            tmp = self._state_path() + f".{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            for attempt in range(5):
                try:
                    os.replace(tmp, self._state_path())
                    break
                except OSError:
                    if attempt < 4:
                        _time.sleep(0.05)
                    else:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                        raise
        except Exception as e:
            print(f"[{self.name}] State save error: {e}")

    def _load_state(self):
        path = self._state_path()
        if not os.path.exists(path):
            print(f"[{self.name}] No state file, starting fresh.")
            return
        try:
            with open(path, encoding="utf-8") as f:
                state = json.load(f)
            saved_date = state.get("date")
            today      = datetime.datetime.now().date().isoformat()
            if saved_date != today:
                print(f"[{self.name}] State outdated ({saved_date}), starting fresh.")
                return
            self._total_alarms    = int(state.get("total_alarms", 0))
            self._last_reset_date = datetime.date.fromisoformat(saved_date)
            print(f"[{self.name}] State loaded ({self._total_alarms} alarms today)")
        except Exception as e:
            print(f"[{self.name}] State load error: {e} — starting fresh.")

    # ==================== HELPERS ====================

    def _check_daily_reset(self):
        today = datetime.datetime.now().date()
        if self._last_reset_date is None:
            self._last_reset_date = today
            return
        if today != self._last_reset_date:
            self._total_alarms     = 0
            self._alert_active     = False
            self._no_apron_since   = None
            self._no_apron_seconds = 0.0
            self._last_alarm_time  = 0.0
            self._last_reset_date  = today
            self._save_state()
            print(f"[{self.name}] Daily reset → {today}")

    def _reference_point(self, bbox):
        x1, y1, x2, y2 = bbox
        if self.reference == "center":
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return ((x1 + x2) / 2.0, y2)

    def _zone_of(self, point):
        """Return the zone type the point falls in, or None."""
        for zone in self._zones:
            inside = cv2.pointPolygonTest(
                zone["points"], (float(point[0]), float(point[1])), False)
            if inside >= 0:
                return zone["type"]
        return None

    def _classify_crop(self, frame, bbox):
        """Run the apron classifier on the person crop. Returns (label, conf)."""
        x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]),
                          int(bbox[2]), int(bbox[3]))
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, 0.0

        # GPU work inside the lock
        with GPU_LOCK, torch.no_grad():
            result = self._classifier(crop, verbose=False)[0]

        # CPU work outside the lock
        top1 = int(result.probs.top1)
        conf = float(result.probs.top1conf)
        raw_label = self._class_names[top1]
        return CLASS_LABEL_MAP.get(raw_label, raw_label), conf

    # ==================== UPDATE ====================

    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        self._check_daily_reset()           # always first line
        self._should_alert = False          # reset every frame
        now = _time.time()

        butcher_count       = 0
        with_apron_count    = 0
        without_apron_count = 0
        detections          = []
        any_no_apron        = False

        n = len(bboxes) if bboxes is not None else 0
        if n > 0:
            for i in range(n):
                # skip non-person detections when class info is available
                cls_id   = int(class_ids[i]) if i < len(class_ids) else 0
                cls_name = class_names.get(cls_id) if class_names else None
                if cls_name is not None and cls_name != "person":
                    continue

                bbox = bboxes[i]
                zone = self._zone_of(self._reference_point(bbox))

                # object_ids may be None — never index it directly otherwise
                if object_ids is not None and i < len(object_ids):
                    track_id = int(object_ids[i])
                else:
                    track_id = -1

                if zone != "butcher":
                    detections.append({
                        "bbox":     bbox,
                        "track_id": track_id,
                        "zone":     "neutral",
                        "label":    None,
                        "conf":     0.0,
                    })
                    continue

                # inside butcher zone — classify apron status
                butcher_count += 1
                label, conf = self._classify_crop(frame, bbox)

                if label is None or conf < self.apron_conf:
                    detections.append({
                        "bbox":     bbox,
                        "track_id": track_id,
                        "zone":     "butcher",
                        "label":    "uncertain",
                        "conf":     conf,
                    })
                elif label == "apron":
                    with_apron_count += 1
                    detections.append({
                        "bbox":     bbox,
                        "track_id": track_id,
                        "zone":     "butcher",
                        "label":    "apron",
                        "conf":     conf,
                    })
                else:                                  # "no_apron"
                    without_apron_count += 1
                    any_no_apron = True
                    detections.append({
                        "bbox":     bbox,
                        "track_id": track_id,
                        "zone":     "butcher",
                        "label":    "no_apron",
                        "conf":     conf,
                    })

        self._butcher_count       = butcher_count
        self._with_apron_count    = with_apron_count
        self._without_apron_count = without_apron_count

        # ----- alarm logic (duration-debounced + cooldown) -----
        in_cooldown = (now - self._last_alarm_time) < self.cooldown_seconds

        if any_no_apron:
            if self._no_apron_since is None:
                self._no_apron_since = now
            self._no_apron_seconds = now - self._no_apron_since
            if not in_cooldown and self._no_apron_seconds >= self.alert_duration_sec:
                self._total_alarms      += 1
                self._last_alarm_time    = now
                self._alert_active       = True
                self._alert_active_until = now + 3.0
                self._should_alert       = True
                self._no_apron_since     = now
                self._no_apron_seconds   = 0.0
                print(f"[{self.name}] !!! ALARM #{self._total_alarms}: "
                      f"butcher without apron")
        else:
            self._no_apron_since   = None
            self._no_apron_seconds = 0.0

        if self._alert_active and now >= self._alert_active_until:
            self._alert_active = False

        # draw() state snapshot
        self._last_detections = detections
        self._last_status = {
            "butcher_count":    butcher_count,
            "with_apron":       with_apron_count,
            "without_apron":    without_apron_count,
            "no_apron_seconds": self._no_apron_seconds,
            "alert_active":     self._alert_active,
        }

        # periodic persistence
        if now - self._last_save_time >= SAVE_INTERVAL_SEC:
            self._save_state()
            self._last_save_time = now

    # ==================== DATA ====================

    def get_data(self) -> dict:
        return {
            "Butcher Count":      self._butcher_count,
            "With Apron":         self._with_apron_count,
            "Without Apron":      self._without_apron_count,
            "No Apron Seconds":   round(self._no_apron_seconds, 1),
            "Status":             "ALARM" if self._alert_active else "Normal",
            "Total Alarms":       self._total_alarms,
            "apron_alert":        self._should_alert,
        }

    # ==================== DRAW ====================

    def draw(self, frame):
        if self._last_status is None:
            return frame

        frame = self._draw_zones(frame)
        frame = self._draw_detections(frame)
        if self._last_status["alert_active"]:
            frame = self._draw_alarm(frame)
        if self.show_panel:
            frame = self._draw_panel(frame)
        return frame

    def _draw_zones(self, frame):
        if self._zones:
            overlay = frame.copy()
            for zone in self._zones:
                color = ZONE_COLORS.get(zone["type"], COLOR_NEUTRAL)
                cv2.fillPoly(overlay, [zone["points"]], color)
            cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
            for zone in self._zones:
                color = ZONE_COLORS.get(zone["type"], COLOR_NEUTRAL)
                label = zone["label"] or ZONE_LABELS.get(zone["type"], zone["type"])
                cx, cy = zone["points"].mean(axis=0).astype(int)
                cv2.polylines(frame, [zone["points"]], True, color, 2)
                cv2.putText(frame, label, (int(cx) - 50, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return frame

    def _draw_detections(self, frame):
        for det in self._last_detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]),
                              int(bbox[2]), int(bbox[3]))
            label = det["label"]
            conf  = det["conf"]

            if det["zone"] != "butcher":
                color = COLOR_NEUTRAL
                text  = "outside"
                thick = 2
            elif label == "apron":
                color = COLOR_APRON
                text  = f"WITH APRON ({conf*100:.0f}%)"
                thick = 3
            elif label == "no_apron":
                color = COLOR_NO_APRON
                text  = f"NO APRON ({conf*100:.0f}%)"
                thick = 3
            else:
                color = COLOR_UNCERT
                text  = f"uncertain ({conf*100:.0f}%)"
                thick = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            cv2.rectangle(frame, (x1, y1 - 24), (x1 + 230, y1), color, -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2, cv2.LINE_AA)

            if det["track_id"] >= 0:
                cv2.putText(frame, f"ID {det['track_id']}", (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        return frame

    def _draw_alarm(self, frame):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h), COLOR_ALARM, 10)
        cv2.putText(frame, "ALARM: BUTCHER WITHOUT APRON!", (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_ALARM, 3, cv2.LINE_AA)
        return frame

    def _draw_panel(self, frame):
        """Orange box with white English text, fed from get_data()."""
        lines = [f"{k}: {v}" for k, v in self.get_data().items()]
        font, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2

        width = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, font, sc, th)
            width = max(width, tw)
        x0, y0 = PANEL_ORIGIN
        x1 = x0 + width + PANEL_PADDING * 2
        y1 = y0 + PANEL_LINE_H * len(lines) + PANEL_PADDING

        cv2.rectangle(frame, (x0, y0), (x1, y1), COLOR_PANEL, -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), COLOR_TEXT, 1)
        for i, line in enumerate(lines):
            y = y0 + PANEL_PADDING + PANEL_LINE_H * i + 14
            cv2.putText(frame, line, (x0 + PANEL_PADDING, y),
                        font, sc, COLOR_TEXT, th, cv2.LINE_AA)
        return frame

    # ==================== SHUTDOWN / RESET ====================

    def shutdown(self):
        self._save_state()

    def reset(self):
        self._butcher_count       = 0
        self._with_apron_count    = 0
        self._without_apron_count = 0
        self._no_apron_since      = None
        self._no_apron_seconds    = 0.0
        self._last_alarm_time     = 0.0
        self._alert_active        = False
        self._total_alarms        = 0