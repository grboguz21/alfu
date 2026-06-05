"""
Zone Dwell Time Analysis
------------------------
Tracks dwell time and visitor count for persons inside defined polygon zones.
Uses pipeline bboxes + tracker object_ids (Type A).

Config example:
    {
        "type":            "bolge_vakit_analizi",
        "name":            "bolge_vakit_cam1",
        "zones": [
            {
                "name":   "Produce Section",
                "points": [[100, 100], [400, 100], [400, 400], [100, 400]],
                "color":  [0, 255, 0]
            }
        ],
        "dwell_threshold": 10.0,
        "reference":       "foot",
        "scale":           1.0,
        "show_panel":      true
    }

get_data() output:
    {
        "Produce Section Dwell Time Minutes": 2.1
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

STATE_DIR         = "state"
SAVE_INTERVAL_SEC = 30


# ==================== MODULE ====================

class BolgeVakitAnaliziModule(BaseModule):

    def __init__(self, name: str,
                 zones: list          = None,
                 dwell_threshold: float = 10.0,
                 reference: str       = "foot",
                 scale: float         = 1.0,
                 show_panel: bool     = True,
                 **_kwargs):
        self.name            = name
        self.dwell_threshold = dwell_threshold
        self.reference       = reference
        self.show_panel      = show_panel

        zones = zones or []

        self._regions = []
        for z in zones:
            pts = (np.array(z.get("points", []), np.float32) * scale).astype(np.int32)
            pts = pts.reshape((-1, 1, 2))
            name = z.get("name", "Zone")
            self._regions.append({
                "name":  name,
                "key":   z.get("key", f"{name} Dwell Time Minutes"),
                "color": tuple(z.get("color", [0, 0, 255])),
                "pts":   pts,
            })

        # Persistent counters
        self._zone_dwell   = {r["name"]: 0.0 for r in self._regions}
        self._zone_visitors = {r["name"]: 0   for r in self._regions}
        self._counted       = set()  # "track_id_zone_idx" format

        # Transient tracking state
        self._person_states = {}  # {"track_id_zone_idx": entry_time}

        self._last_reset_date = None
        self._last_save_time  = 0.0

        # draw() guard — None until update() is called
        self._last_status = None

        self._load_state()
        print(f"✅ BolgeVakitAnaliziModule ready [{name}] — {len(self._regions)} zones")

    # ==================== PERSISTENCE ====================

    def _state_path(self) -> str:
        return os.path.join(STATE_DIR, f"bolge_vakit_{self.name}.json")

    def _save_state(self):
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            state = {
                "date":          (self._last_reset_date.isoformat()
                                  if self._last_reset_date else None),
                "zone_dwell":    self._zone_dwell,
                "zone_visitors": self._zone_visitors,
                "counted":       list(self._counted),
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

            loaded_dwell = state.get("zone_dwell", {})
            for k in self._zone_dwell:
                self._zone_dwell[k] = float(loaded_dwell.get(k, 0.0))

            loaded_visitors = state.get("zone_visitors", {})
            for k in self._zone_visitors:
                self._zone_visitors[k] = int(loaded_visitors.get(k, 0))

            self._counted         = set(state.get("counted", []))
            self._last_reset_date = datetime.date.fromisoformat(saved_date)
            print(f"[{self.name}] State loaded")
        except Exception as e:
            print(f"[{self.name}] State load error: {e} — starting fresh.")

    # ==================== HELPERS ====================

    def _check_daily_reset(self):
        today = datetime.datetime.now().date()
        if self._last_reset_date is None:
            self._last_reset_date = today
            return
        if today != self._last_reset_date:
            for k in self._zone_dwell:
                self._zone_dwell[k]    = 0.0
                self._zone_visitors[k] = 0
            self._person_states.clear()
            self._counted.clear()
            self._last_reset_date = today
            self._save_state()
            print(f"[{self.name}] Daily reset → {today}")

    def _get_reference_point(self, x1, y1, x2, y2):
        cx = int((x1 + x2) / 2)
        if self.reference == "head":
            cy = int(y1)
        elif self.reference == "foot":
            cy = int(y2)
        else:  # "center"
            cy = int(y1 + (y2 - y1) * 0.15)
        return cx, cy

    # ==================== UPDATE ====================

    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        self._check_daily_reset()
        now        = _time.time()
        active_boxes = []
        current_ids  = set()

        if object_ids is not None:
            current_ids = set(int(oid) for oid in object_ids)

        if object_ids is not None and len(bboxes) > 0:
            for bbox, cls_id, obj_id in zip(bboxes, class_ids, object_ids):
                if int(cls_id) != 0:
                    continue

                tid            = int(obj_id)
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cx, cy         = self._get_reference_point(x1, y1, x2, y2)
                box_info       = {"bbox": (x1, y1, x2, y2), "cx": cx, "cy": cy,
                                  "tid": tid, "status": []}

                for zi, r in enumerate(self._regions):
                    inside = cv2.pointPolygonTest(r["pts"], (cx, cy), False) >= 0
                    key    = f"{tid}_{zi}"

                    if inside:
                        if key not in self._person_states:
                            self._person_states[key] = now
                        elapsed = now - self._person_states[key]
                        if elapsed >= self.dwell_threshold:
                            box_info["status"].append((r["name"], elapsed, True))
                            if key not in self._counted:
                                self._counted.add(key)
                                self._zone_visitors[r["name"]] += 1
                        else:
                            box_info["status"].append((r["name"], elapsed, False))
                    else:
                        if key in self._person_states:
                            duration = now - self._person_states[key]
                            if duration >= self.dwell_threshold:
                                self._zone_dwell[r["name"]] += duration
                            del self._person_states[key]

                active_boxes.append(box_info)

        # Clean up IDs no longer visible
        keys_to_remove = [k for k in self._person_states
                          if int(k.split("_")[0]) not in current_ids]
        for k in keys_to_remove:
            duration = now - self._person_states[k]
            zi       = int(k.split("_")[1])
            zone_name = self._regions[zi]["name"]
            if duration >= self.dwell_threshold:
                self._zone_dwell[zone_name] += duration
            del self._person_states[k]

        self._last_status = {"active_boxes": active_boxes}

        if now - self._last_save_time >= SAVE_INTERVAL_SEC:
            self._save_state()
            self._last_save_time = now

    # ==================== DATA ====================

    def get_data(self) -> dict:
        return {
            r["key"]: round(self._zone_dwell.get(r["name"], 0.0) / 60.0, 1)
            for r in self._regions
        }

    # ==================== DRAW ====================

    def draw(self, frame):
        if self._last_status is None:
            return frame

        for r in self._regions:
            cv2.polylines(frame, [r["pts"]], True, r["color"], 2)
            corner = r["pts"].reshape(-1, 2)[0]
            cv2.putText(frame, r["name"], (int(corner[0]), int(corner[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, r["color"], 2)

        if self.show_panel:
            for box_info in self._last_status.get("active_boxes", []):
                x1, y1, x2, y2 = box_info["bbox"]
                cx, cy         = box_info["cx"], box_info["cy"]
                tid            = box_info["tid"]

                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

                color = (0, 0, 255)
                label = f"ID:{tid}"

                if box_info["status"]:
                    active = [s for s in box_info["status"] if s[2]]
                    if active:
                        color = (0, 255, 0)
                        label = f"ID:{tid} | {active[0][1]:.1f}s"
                    else:
                        color = (0, 255, 255)
                        label = f"ID:{tid} | Passing..."

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    # ==================== SHUTDOWN ====================

    def shutdown(self):
        self._save_state()
        print(f"[{self.name}] Shutdown complete, state saved.")

    def reset(self):
        for k in self._zone_dwell:
            self._zone_dwell[k]    = 0.0
            self._zone_visitors[k] = 0
        self._person_states.clear()
        self._counted.clear()
        self._save_state()
        print(f"[{self.name}] Manual reset done.")
