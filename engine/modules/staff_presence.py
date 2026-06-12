"""
Staff Presence / Unattended Customer Alarm
------------------------------------------
Type A module. Uses the main pipeline detections (person bounding boxes and
tracker IDs). Raises an alarm when a customer stands in the "customer" zone
while no staff member is present in the "staff" zone for longer than a
configurable threshold. Also accumulates the daily unattended time and the
number of alarm activations.

Config example:
    {
        "type":           "staff_presence",
        "name":           "staff_presence_cam1",
        "regions":        [
            {"type": "customer", "label": "Customer Area",
             "points": [[40, 200], [620, 200], [620, 700], [40, 700]]},
            {"type": "staff",    "label": "Staff Area",
             "points": [[660, 200], [1240, 200], [1240, 700], [660, 700]]}
        ],
        "wait_threshold": 10.0,
        "reference":      "foot",
        "scale":          1.0,
        "show_panel":     true
    }

get_data() output:
    {
        "Customer Count":      int,
        "Staff Count":         int,
        "Waiting Seconds":     float,
        "Status":              str,     # "ALARM" | "Normal"
        "Total Alert Minutes": float,
        "Alert Triggers":      int,
        "staff_absent_alert":  bool     # main.py watches this to fire an alert
    }
"""

import os
import json
import time as _time
import datetime

import cv2
import numpy as np

from .base import BaseModule

# ==================== CONFIG ====================

STATE_DIR              = "state"
SAVE_INTERVAL_SEC      = 30
DEFAULT_WAIT_THRESHOLD = 10.0      # seconds before an unattended customer alarms
BEEP_INTERVAL_SEC      = 1.0       # min seconds between terminal beeps while alarming

# Visual constants (BGR)
COLOR_CUSTOMER = (255, 160, 0)     # customer zone
COLOR_STAFF    = (0, 200, 0)       # staff zone
COLOR_ALARM    = (0, 0, 255)       # alarm (red)
COLOR_NEUTRAL  = (170, 170, 170)   # detection outside any zone
COLOR_PANEL    = (0, 140, 255)     # orange info box
COLOR_TEXT     = (255, 255, 255)   # white font

# Panel layout
PANEL_PADDING  = 12
PANEL_LINE_H   = 28
PANEL_ORIGIN   = (10, 10)

ZONE_LABELS    = {"customer": "Customer Area", "staff": "Staff Area"}
ZONE_COLORS    = {"customer": COLOR_CUSTOMER, "staff": COLOR_STAFF}


# ==================== MODULE ====================

class StaffPresenceModule(BaseModule):

    def __init__(self, name: str,
                 regions: list = None,
                 wait_threshold: float = DEFAULT_WAIT_THRESHOLD,
                 reference: str = "foot",
                 scale: float = 1.0,
                 show_panel: bool = True,
                 show_zones: bool = True,
                 **_kwargs):
        # 1) config parameters
        self.name           = name
        self.show_zones     = show_zones
        self.wait_threshold = float(wait_threshold)
        self.reference      = reference          # "foot" | "center"
        self.scale          = float(scale)
        self.show_panel     = show_panel

        # pre-scale region polygons once
        self._zones = []
        for region in (regions or []):
            pts = (np.array(region["points"], dtype=np.float32) * self.scale)
            self._zones.append({
                "type":   region.get("type", "customer"),
                "label":  region.get("label"),
                "points": pts.astype(np.int32),
            })

        # 2) (Type A: no own model)

        # 3) internal state
        self._customer_count      = 0
        self._staff_count         = 0
        self._elapsed             = 0.0     # current unattended duration (transient)
        self._condition_start     = None    # when "customer & no staff" began (transient)
        self._alert_active        = False
        self._alert_session_start = None    # when current alarm session began
        self._should_alert        = False
        # daily-accumulated, persisted
        self._total_alert_seconds = 0.0
        self._alert_count         = 0
        self._last_reset_date     = None
        self._last_save_time      = 0.0
        self._last_beep           = 0.0

        # 4) draw() safety — must be None before first update()
        self._last_status     = None
        self._last_detections = []

        # 5) load persisted state (always last)
        self._load_state()

        # 6) ready
        print(f"✅ StaffPresenceModule ready [{name}]")

    # ==================== PERSISTENCE ====================

    def _state_path(self) -> str:
        return os.path.join(STATE_DIR, f"staff_presence_{self.name}.json")

    def _save_state(self):
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            state = {
                "date":                (self._last_reset_date.isoformat()
                                        if self._last_reset_date else None),
                "total_alert_seconds": self._total_alert_seconds,
                "alert_count":         self._alert_count,
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
            self._total_alert_seconds = float(state.get("total_alert_seconds", 0.0))
            self._alert_count         = int(state.get("alert_count", 0))
            self._last_reset_date     = datetime.date.fromisoformat(saved_date)
            print(f"[{self.name}] State loaded "
                  f"({self._total_alert_seconds/60:.1f} min, "
                  f"{self._alert_count} alerts)")
        except Exception as e:
            print(f"[{self.name}] State load error: {e} — starting fresh.")

    # ==================== HELPERS ====================

    def _check_daily_reset(self):
        today = datetime.datetime.now().date()
        if self._last_reset_date is None:
            self._last_reset_date = today
            return
        if today != self._last_reset_date:
            self._total_alert_seconds = 0.0
            self._alert_count         = 0
            self._alert_active        = False
            self._alert_session_start = None
            self._condition_start     = None
            self._last_reset_date     = today
            self._save_state()
            print(f"[{self.name}] Daily reset → {today}")

    def _reference_point(self, bbox):
        x1, y1, x2, y2 = bbox
        if self.reference == "center":
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return ((x1 + x2) / 2.0, y2)          # "foot" = bottom-center (ground)

    def _zone_of(self, point):
        """Return the zone type the point falls in. Staff zone has priority."""
        found = None
        for zone in self._zones:
            inside = cv2.pointPolygonTest(
                zone["points"], (float(point[0]), float(point[1])), False)
            if inside >= 0:
                if zone["type"] == "staff":
                    return "staff"
                found = zone["type"]
        return found

    # ==================== UPDATE ====================

    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        self._check_daily_reset()           # always first line
        self._should_alert = False          # reset every frame
        now = _time.time()

        customer_count = 0
        staff_count    = 0
        detections     = []

        n = len(bboxes)
        if n > 0:
            for i in range(n):
                # skip non-person detections when class info is available
                cls_id = int(class_ids[i]) if i < len(class_ids) else 0
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

                detections.append((bbox, track_id, zone if zone else "neutral"))
                if zone == "staff":
                    staff_count += 1
                elif zone == "customer":
                    customer_count += 1

        self._customer_count = customer_count
        self._staff_count    = staff_count

        # ----- alarm logic (debounced) -----
        condition = (customer_count > 0 and staff_count == 0)
        if condition:
            if self._condition_start is None:
                self._condition_start = now
            self._elapsed = now - self._condition_start

            if self._elapsed >= self.wait_threshold:
                if not self._alert_active:
                    self._alert_active        = True
                    self._alert_session_start = now
                    self._alert_count        += 1
                    print(f"[{self.name}] !!! ALARM: no staff, "
                          f"customer waiting {self._elapsed:.1f}s")
                self._should_alert = True
                if now - self._last_beep > BEEP_INTERVAL_SEC:
                    print("\a", end="", flush=True)
                    self._last_beep = now
        else:
            if self._alert_active and self._alert_session_start is not None:
                # close the open alarm session into the daily total
                self._total_alert_seconds += now - self._alert_session_start
                print(f"[{self.name}] alarm cleared")
            self._alert_active        = False
            self._alert_session_start = None
            self._condition_start     = None
            self._elapsed             = 0.0

        # draw() state snapshot (draw() runs right after update())
        self._last_detections = detections
        self._last_status = {
            "customer_count": customer_count,
            "staff_count":    staff_count,
            "elapsed":        self._elapsed,
            "alert_active":   self._alert_active,
        }

        # periodic persistence
        if now - self._last_save_time >= SAVE_INTERVAL_SEC:
            self._save_state()
            self._last_save_time = now

    # ==================== DATA ====================

    def get_data(self) -> dict:
        now   = _time.time()
        total = self._total_alert_seconds
        if self._alert_active and self._alert_session_start is not None:
            total += now - self._alert_session_start   # include open session

        return {
            "Customer Count":      self._customer_count,
            "Staff Count":         self._staff_count,
            "Waiting Seconds":     round(self._elapsed, 1),
            "Status":              "ALARM" if self._alert_active else "Normal",
            "Total Alert Minutes": round(total / 60, 1),
            "Alert Triggers":      self._alert_count,
            "staff_absent_alert":  self._should_alert,
        }

    # ==================== DRAW ====================

    def draw(self, frame):
        if self._last_status is None:        # update() not called yet
            return frame

        if self.show_zones:
            frame = self._draw_zones(frame)
        if self.show_panel:
            frame = self._draw_detections(frame)
            if self._last_status["alert_active"]:
                frame = self._draw_alarm(frame)
            frame = self._draw_panel(frame)
        return frame

    def _draw_zones(self, frame):
        if self._zones:
            overlay = frame.copy()
            for zone in self._zones:
                color = ZONE_COLORS.get(zone["type"], COLOR_NEUTRAL)
                cv2.fillPoly(overlay, [zone["points"]], color)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            for zone in self._zones:
                color = ZONE_COLORS.get(zone["type"], COLOR_NEUTRAL)
                label = zone["label"] or ZONE_LABELS.get(zone["type"], zone["type"])
                cx, cy = zone["points"].mean(axis=0).astype(int)
                cv2.polylines(frame, [zone["points"]], True, color, 2)
                cv2.putText(frame, label, (int(cx) - 50, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return frame

    def _draw_detections(self, frame):
        for bbox, track_id, zone in self._last_detections:
            x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]),
                              int(bbox[2]), int(bbox[3]))
            color = ZONE_COLORS.get(zone, COLOR_NEUTRAL)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if track_id >= 0:
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        return frame

    def _draw_alarm(self, frame):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h), COLOR_ALARM, 10)
        cv2.putText(frame, "ALARM: NO STAFF!", (20, h - 25),
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
        # close any open alarm session into the daily total, then persist
        if self._alert_active and self._alert_session_start is not None:
            self._total_alert_seconds += _time.time() - self._alert_session_start
            self._alert_session_start = None
            self._alert_active        = False
        self._save_state()

    def reset(self):
        self._total_alert_seconds = 0.0
        self._alert_count         = 0
        self._alert_active        = False
        self._alert_session_start = None
        self._condition_start     = None
        self._elapsed             = 0.0
