"""
Cashier Working Hours Module
----------------------------
Tracks active working time of cashiers by detecting persons
inside defined polygon ROIs using pipeline bboxes (Type A).

Config example:
    {
        "type":        "kasiyer_suresi",
        "name":        "kasiyer_suresi_cam1",
        "kasalar": [
            {
                "id":     "Payment Points 1",
                "coords": [[1520,456],[2110,394],[2026,8],[1448,0],[1520,458]]
            },
            {
                "id":     "Payment Points 2",
                "coords": [[6,144],[420,272],[226,608],[4,594],[10,144]]
            }
        ],
        "original_w":  2560,
        "original_h":  1440,
        "display_w":   1280,
        "show_panel":  true
    }

get_data() output:
    {
        "Payment Points 1 Active Minutes": float,
        "Payment Points 1 Is Active":      bool,
        "Payment Points 2 Active Minutes": float,
        "Payment Points 2 Is Active":      bool
    }
"""

import os
import json
import time as _time
import datetime

import cv2
import numpy as np
from shapely.geometry import Polygon, Point

from .base import BaseModule

# ==================== CONFIG ====================

STATE_DIR         = "state"
SAVE_INTERVAL_SEC = 30

# Default image dimensions
DEFAULT_ORIGINAL_W = 2560
DEFAULT_ORIGINAL_H = 1440
DEFAULT_DISPLAY_W  = 1280

# Colors (BGR)
COLOR_PALETTE = [
    (235, 206, 135),   # Yellowish  - Point 1
    (255, 105, 180),   # Pink       - Point 2
    (0,   255, 165),   # Green      - Point 3
    (255, 165,   0),   # Orange     - Point 4
]
COLOR_PERSON_OUT  = (150, 240, 0)    # Lime green - person outside zone
COLOR_WHITE       = (255, 255, 255)
COLOR_DARK        = (15,  15,  15)
COLOR_DARK_BORDER = (40,  40,  40)
COLOR_PANEL_BG    = (15, 108, 242)   # Panel background


# ==================== MODULE ====================

class KasiyerSuresiModule(BaseModule):
    """
    Type A — uses pipeline bboxes.
    Detects persons inside each cashier polygon using the main YOLO model output.
    """

    def __init__(
        self,
        name: str,
        kasalar: list             = None,
        original_w: int           = DEFAULT_ORIGINAL_W,
        original_h: int           = DEFAULT_ORIGINAL_H,
        display_w: int            = DEFAULT_DISPLAY_W,
        show_panel: bool          = True,
        **_kwargs,
    ):
        self.name       = name
        self.show_panel = show_panel

        # Scale ratio
        self._orig_w   = original_w
        self._orig_h   = original_h
        self._disp_w   = display_w
        self._ratio    = display_w / float(original_w)
        self._disp_h   = int(original_h * self._ratio)

        # Build cashier zone list
        if kasalar is None:
            kasalar = []
        self._kasalar = self._build_kasalar(kasalar)

        # Per-zone counters
        for k in self._kasalar:
            k["total_seconds"] = 0.0
            k["is_active"]     = False
            k["session_start"] = None

        # draw() guard — None until update() is called
        self._last_status    = None
        self._last_points    = []
        self._last_save_time = 0.0
        self._last_reset_date = None

        self._load_state()
        print(f"✅ KasiyerSuresiModule ready [{name}] — {len(self._kasalar)} zones")

    # ==================== SETUP ====================

    def _build_kasalar(self, kasalar_cfg: list) -> list:
        """Builds Shapely polygons and scaled display coords from config."""
        result = []
        for idx, cfg in enumerate(kasalar_cfg):
            kasa_id = cfg.get("id", f"Zone-{idx+1:02d}")
            coords  = [tuple(p) for p in cfg.get("coords", [])]
            if len(coords) < 3:
                print(f"[{self.name}] Warning: {kasa_id} has insufficient coordinates, skipping.")
                continue

            color         = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
            scaled_coords = [(int(x * self._ratio), int(y * self._ratio)) for x, y in coords]

            result.append({
                "id":             kasa_id,
                "poly":           Polygon(scaled_coords),
                "coords_scaled":  scaled_coords,
                "border_color":   color,
            })
        return result

    # ==================== PERSISTENCE ====================

    def _state_path(self) -> str:
        return os.path.join(STATE_DIR, f"kasiyer_suresi_{self.name}.json")

    def _save_state(self):
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            kasalar_state = [
                {
                    "id":            k["id"],
                    "total_seconds": k["total_seconds"],
                }
                for k in self._kasalar
            ]
            state = {
                "date":    (self._last_reset_date.isoformat()
                            if self._last_reset_date else None),
                "kasalar": kasalar_state,
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

            saved_kasalar = {s["id"]: s for s in state.get("kasalar", [])}
            for k in self._kasalar:
                if k["id"] in saved_kasalar:
                    k["total_seconds"] = float(
                        saved_kasalar[k["id"]].get("total_seconds", 0.0)
                    )

            self._last_reset_date = datetime.date.fromisoformat(saved_date)
            totals = [f"{k['id']}={k['total_seconds']/60:.1f}min" for k in self._kasalar]
            print(f"[{self.name}] State loaded: {', '.join(totals)}")
        except Exception as e:
            print(f"[{self.name}] State load error: {e} — starting fresh.")

    # ==================== HELPERS ====================

    def _check_daily_reset(self):
        today = datetime.datetime.now().date()
        if self._last_reset_date is None:
            self._last_reset_date = today
            return
        if today != self._last_reset_date:
            for k in self._kasalar:
                k["total_seconds"] = 0.0
                k["is_active"]     = False
                k["session_start"] = None
            self._last_reset_date = today
            self._save_state()
            print(f"[{self.name}] Daily reset → {today}")

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    # ==================== UPDATE ====================

    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        self._check_daily_reset()
        now = _time.time()

        # Which zones are occupied this frame?
        occupied = {k["id"]: False for k in self._kasalar}

        if len(bboxes) > 0:
            for bbox, cls_id in zip(bboxes, class_ids):
                if int(cls_id) != 0:
                    continue
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                center = Point((x1 + x2) / 2, (y1 + y2) / 2)
                for k in self._kasalar:
                    if k["poly"].contains(center):
                        occupied[k["id"]] = True
                        break

        # Accumulate time
        for k in self._kasalar:
            is_occupied = occupied[k["id"]]
            if is_occupied:
                if not k["is_active"]:
                    k["is_active"]     = True
                    k["session_start"] = now
                else:
                    k["total_seconds"] += now - k["session_start"]
                    k["session_start"]  = now
            else:
                if k["is_active"]:
                    k["is_active"]     = False
                    k["session_start"] = None

        # Store state for draw()
        self._last_status = {
            k["id"]: {
                "is_active":     k["is_active"],
                "total_seconds": k["total_seconds"],
                "session_start": k["session_start"],
                "coords_scaled": k["coords_scaled"],
                "border_color":  k["border_color"],
            }
            for k in self._kasalar
        }

        # Store person center points for draw()
        self._last_points = []
        if len(bboxes) > 0:
            for bbox, cls_id in zip(bboxes, class_ids):
                if int(cls_id) != 0:
                    continue
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cx     = int((x1 + x2) / 2)
                cy     = int((y1 + y2) / 2)
                center = Point(cx, cy)

                point_color = COLOR_PERSON_OUT
                for k in self._kasalar:
                    if k["poly"].contains(center):
                        point_color = k["border_color"]
                        break

                self._last_points.append((cx, cy, point_color))

        # Periodic save
        if now - self._last_save_time >= SAVE_INTERVAL_SEC:
            self._save_state()
            self._last_save_time = now

    # ==================== DATA ====================

    def get_data(self) -> dict:
        now = _time.time()
        return {
            key: val
            for k in self._kasalar
            for key, val in {
                f"{k['id']} Active Minutes": round(
                    (k["total_seconds"] + (now - k["session_start"]
                     if k["is_active"] and k["session_start"] else 0)) / 60, 1
                ),
                f"{k['id']} Is Active": k["is_active"],
            }.items()
        }

    # ==================== DRAW ====================

    def draw(self, frame):
        if self._last_status is None:
            return frame

        now = _time.time()

        # 1. Draw zone polygons
        for k in self._kasalar:
            status = self._last_status[k["id"]]
            pts    = np.array(status["coords_scaled"], np.int32)
            cv2.polylines(frame, [pts], True, status["border_color"], 2, cv2.LINE_AA)

            # Label capsule (top-left corner of zone)
            total = status["total_seconds"]
            if status["is_active"] and status["session_start"]:
                total += now - status["session_start"]

            label       = f" {k['id']} | {self._fmt_time(total)} "
            label_x, label_y = status["coords_scaled"][0]
            label_y_adj = max(label_y - 12, 25)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)

            # Background box
            cv2.rectangle(
                frame,
                (label_x - 2, label_y_adj - th - 5),
                (label_x + tw + 2, label_y_adj + 5),
                COLOR_DARK, -1,
            )
            # Colored left border
            cv2.rectangle(
                frame,
                (label_x - 2, label_y_adj - th - 5),
                (label_x + 1, label_y_adj + 5),
                status["border_color"], -1,
            )
            # Thin dark outline
            cv2.rectangle(
                frame,
                (label_x - 2, label_y_adj - th - 5),
                (label_x + tw + 2, label_y_adj + 5),
                COLOR_DARK_BORDER, 1, cv2.LINE_AA,
            )
            cv2.putText(
                frame, label,
                (label_x + 2, label_y_adj),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1, cv2.LINE_AA,
            )

        # 2. Person center dots
        for (cx, cy, color) in self._last_points:
            cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)

        # 3. Status panel (top-right)
        if self.show_panel:
            frame = self._draw_panel(frame, now)

        return frame

    def _draw_panel(self, frame, now: float):
        h, w   = frame.shape[:2]
        n      = len(self._kasalar)
        MARGIN = 12
        STEP   = 24
        box_h  = 52 + n * STEP
        box_w  = 200
        px     = w - box_w - MARGIN
        py     = h - box_h - MARGIN

        cv2.rectangle(frame, (px, py), (px + box_w, py + box_h), COLOR_PANEL_BG, -1)
        cv2.putText(
            frame, "CASHIER HOURS",
            (px + 7, py + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_WHITE, 1, cv2.LINE_AA,
        )
        for idx, k in enumerate(self._kasalar):
            total = k["total_seconds"]
            if k["is_active"] and k["session_start"]:
                total += now - k["session_start"]
            label = f"{k['id']}: {self._fmt_time(total)}"
            cv2.putText(
                frame, label,
                (px + 7, py + 46 + idx * STEP),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA,
            )
        return frame

    # ==================== SHUTDOWN ====================

    def shutdown(self):
        now = _time.time()
        for k in self._kasalar:
            if k["is_active"] and k["session_start"]:
                k["total_seconds"] += now - k["session_start"]
                k["session_start"]  = None
                k["is_active"]      = False
        self._save_state()
        print(f"[{self.name}] Shutdown complete, state saved.")

    # ==================== RESET ====================

    def reset(self):
        for k in self._kasalar:
            k["total_seconds"] = 0.0
            k["is_active"]     = False
            k["session_start"] = None
        self._save_state()
        print(f"[{self.name}] Manual reset done.")
