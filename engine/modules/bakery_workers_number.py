"""
Bakery Workers Number
---------------------
Counts workers inside the kitchen zone each frame and triggers an alarm
when the count exceeds the configured maximum.

Uses a two-pass strategy: pipeline tracker results for well-lit workers,
plus a dedicated crop-inference pass to catch dark or occluded workers
the tracker missed.

Config example:
    {
        "type":             "bakery_workers_number",
        "name":             "bakery_workers_number_cam1",
        "model_path":       "models/yolo26l-pose.pt",
        "kitchen_zone_rel": [[0.720, 0.000], [0.928, 0.000], [0.889, 0.545],
                             [0.815, 0.558], [0.720, 0.492]],
        "max_people":       3,
        "show_panel":       true
    }

get_data() output:
    {
        "People In Kitchen":  int,
        "Max People Today":   int,
        "overcrowding_alert": bool
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

STATE_DIR         = "state"
SAVE_INTERVAL_SEC = 30

DEFAULT_KITCHEN_ZONE_REL = [
    [0.720, 0.000],   # top-left:     72% from left, top edge
    [0.928, 0.000],   # top-right:    93% from left, top edge
    [0.889, 0.545],   # bottom-right: 89% from left, 55% down
    [0.815, 0.558],   # bottom-mid:   82% from left, 56% down
    [0.720, 0.492],   # bottom-left:  72% from left, 49% down
]
DEFAULT_MAX_PEOPLE     = 3
MIN_CROP_BOX_AREA      = 3000   # px² — skip body-fragment detections below this
CROP_OVERLAP_THRESHOLD = 0.30   # overlap ratio to skip already-tracked boxes

COLOR_OK      = (0, 220, 0)
COLOR_ALARM   = (0, 0, 255)
COLOR_TRACKED = (0, 0, 255)
COLOR_CROP    = (0, 165, 255)
COLOR_ZONE    = (0, 220, 0)
ALARM_ALPHA   = 0.15


# ==================== MODULE ====================

class BakeryWorkersNumberModule(BaseModule):

    def __init__(self, name: str,
                 model_path: str        = "models/yolo26l-pose.pt",
                 kitchen_zone_rel: list = None,
                 max_people: int        = DEFAULT_MAX_PEOPLE,
                 show_panel: bool       = True,
                 **_kwargs):
        self.name             = name
        self.model_path       = model_path
        self.kitchen_zone_rel = np.array(
            kitchen_zone_rel if kitchen_zone_rel is not None else DEFAULT_KITCHEN_ZONE_REL,
            dtype=np.float32,
        )
        self.max_people = max_people
        self.show_panel = show_panel

        self._model = YOLO(model_path)

        # Internal state
        self._people_in_kitchen = 0
        self._max_people_today  = 0
        self._should_alert      = False
        self._last_reset_date   = None
        self._last_save_time    = 0.0

        # draw() state — None until first update()
        self._last_status      = None
        self._draw_boxes       = []    # list of (x1, y1, x2, y2, color, label)
        self._kitchen_zone_abs = None  # absolute pixel coords for current frame

        self._load_state()
        print(f"✅ BakeryWorkersNumberModule ready [{name}]")

    # ==================== PERSISTENCE ====================

    def _state_path(self) -> str:
        return os.path.join(STATE_DIR, f"bakery_workers_number_{self.name}.json")

    def _save_state(self):
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            state = {
                "date":             (self._last_reset_date.isoformat()
                                     if self._last_reset_date else None),
                "max_people_today": self._max_people_today,
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
            self._max_people_today = int(state.get("max_people_today", 0))
            self._last_reset_date  = datetime.date.fromisoformat(saved_date)
            print(f"[{self.name}] State loaded (max today: {self._max_people_today})")
        except Exception as e:
            print(f"[{self.name}] State load error: {e} — starting fresh.")

    # ==================== HELPERS ====================

    def _check_daily_reset(self):
        today = datetime.datetime.now().date()
        if self._last_reset_date is None:
            self._last_reset_date = today
            return
        if today != self._last_reset_date:
            self._max_people_today  = 0
            self._people_in_kitchen = 0
            self._should_alert      = False
            self._last_reset_date   = today
            self._save_state()
            print(f"[{self.name}] Daily reset → {today}")

    @staticmethod
    def _is_in_zone(cx, cy, zone_poly) -> bool:
        return cv2.pointPolygonTest(zone_poly, (float(cx), float(cy)), False) >= 0

    @staticmethod
    def _bbox_overlaps_zone(x1, y1, x2, y2, zone_poly) -> bool:
        # Counts person even if body is partially outside the frame — any of
        # five key points inside the zone is enough to include them.
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        points = [
            (mx, y2),   # feet
            (mx, my),   # center
            (mx, y1),   # head
            (x1, my),   # mid-left
            (x2, my),   # mid-right
        ]
        return any(
            cv2.pointPolygonTest(zone_poly, (float(px), float(py)), False) >= 0
            for px, py in points
        )

    # ==================== UPDATE ====================

    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        self._check_daily_reset()
        self._should_alert = False
        now = _time.time()

        h, w = frame.shape[:2]
        kitchen_zone           = (self.kitchen_zone_rel * [w, h]).astype(np.int32)
        self._kitchen_zone_abs = kitchen_zone

        people_in_kitchen = 0
        tracked_boxes     = []   # (x1, y1, x2, y2) already counted
        draw_boxes        = []

        # — Tracker pass: pipeline-supplied bboxes —
        if object_ids is not None and len(bboxes) > 0:
            for i, (bbox, cls_id) in enumerate(zip(bboxes, class_ids)):
                if int(cls_id) != 0:
                    continue
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                if not self._bbox_overlaps_zone(x1, y1, x2, y2, kitchen_zone):
                    continue
                obj_id = int(object_ids[i]) if object_ids[i] is not None else -1
                conf   = float(scores[i]) if i < len(scores) else 0.0
                people_in_kitchen += 1
                tracked_boxes.append((x1, y1, x2, y2))
                draw_boxes.append((x1, y1, x2, y2, COLOR_TRACKED,
                                   f"ID:{obj_id} {conf:.2f}"))

        # — Crop pass: catches dark/occluded workers the tracker missed —
        zx1 = int(kitchen_zone[:, 0].min())
        zy1 = int(kitchen_zone[:, 1].min())
        zx2 = int(kitchen_zone[:, 0].max())
        zy2 = int(kitchen_zone[:, 1].max())
        kitchen_crop = frame[zy1:zy2, zx1:zx2]

        with GPU_LOCK, torch.no_grad():
            crop_results = self._model(kitchen_crop, classes=[0],
                                       conf=0.12, iou=0.60, verbose=False)

        for box in crop_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2 - x1) * (y2 - y1) < MIN_CROP_BOX_AREA:
                continue
            x1 += zx1; y1 += zy1; x2 += zx1; y2 += zy1
            if not self._bbox_overlaps_zone(x1, y1, x2, y2, kitchen_zone):
                continue
            already_counted = any(
                max(0, min(x2, tx2) - max(x1, tx1)) *
                max(0, min(y2, ty2) - max(y1, ty1))
                > CROP_OVERLAP_THRESHOLD * (x2 - x1) * (y2 - y1)
                for tx1, ty1, tx2, ty2 in tracked_boxes
            )
            if already_counted:
                continue
            conf = float(box.conf[0])
            people_in_kitchen += 1
            tracked_boxes.append((x1, y1, x2, y2))
            draw_boxes.append((x1, y1, x2, y2, COLOR_CROP, f"? {conf:.2f}"))

        self._people_in_kitchen = people_in_kitchen
        if people_in_kitchen > self._max_people_today:
            self._max_people_today = people_in_kitchen
        if people_in_kitchen > self.max_people:
            self._should_alert = True

        self._draw_boxes  = draw_boxes
        self._last_status = {
            "people_in_kitchen": people_in_kitchen,
            "should_alert":      self._should_alert,
        }

        if now - self._last_save_time >= SAVE_INTERVAL_SEC:
            self._save_state()
            self._last_save_time = now

    # ==================== DATA ====================

    def get_data(self) -> dict:
        return {
            "People In Kitchen":  self._people_in_kitchen,
            "Max People Today":   self._max_people_today,
            "overcrowding_alert": self._should_alert,
        }

    # ==================== DRAW ====================

    def draw(self, frame):
        if self._last_status is None:
            return frame
        frame = self._draw_zones(frame)
        frame = self._draw_detections(frame)
        if self.show_panel:
            frame = self._draw_panel(frame)
        if self._last_status["should_alert"]:
            frame = self._draw_alarm(frame)
        return frame

    def _draw_zones(self, frame):
        if self._kitchen_zone_abs is None:
            return frame
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self._kitchen_zone_abs], COLOR_ZONE)
        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)
        cv2.polylines(frame, [self._kitchen_zone_abs],
                      isClosed=True, color=COLOR_ZONE, thickness=2)
        h, w = frame.shape[:2]
        cv2.putText(frame, "Kitchen",
                    (int(0.725 * w), int(0.090 * h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ZONE, 2, cv2.LINE_AA)
        return frame

    def _draw_detections(self, frame):
        for x1, y1, x2, y2, color, label in self._draw_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        return frame

    def _draw_panel(self, frame):
        count = self._last_status["people_in_kitchen"]
        color = COLOR_OK if count <= self.max_people else COLOR_ALARM
        cv2.putText(frame, f"People in kitchen: {count}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        return frame

    def _draw_alarm(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, ALARM_ALPHA, frame, 1 - ALARM_ALPHA, 0, frame)
        cv2.rectangle(frame, (10, 55), (620, 115), (0, 0, 180), -1)
        cv2.rectangle(frame, (10, 55), (620, 115), COLOR_ALARM, 3)
        cv2.putText(frame, "!! ALARM: TOO MANY PEOPLE IN KITCHEN !!",
                    (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    # ==================== SHUTDOWN ====================

    def shutdown(self):
        self._save_state()
