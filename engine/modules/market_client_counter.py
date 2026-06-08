"""
Market Client Counter
---------------------
Count IN/OUT when a tracked person crosses between two ROIs and the track is lost.

Type A — uses pipeline bboxes + tracker object_ids (BoT-SORT).

Rules:
  • ROI IN  (polygon_in)  = inside shop / market (green)
  • ROI OUT (polygon_out) = outside street (blue)
  • OUT → IN, then track lost → +1 IN
  • IN  → OUT, then track lost → +1 OUT
  • If foot is in both polygons, OUT wins.

Config example:
    {
        "type":            "market_client_counter",
        "name":            "market_client_counter_cam1",
        "polygon_in":      [[120, 300], [500, 300], [500, 700], [120, 700]],
        "polygon_out":     [[20, 200], [600, 200], [600, 900], [20, 900]],
        "target_class_id": 0,
        "show_panel":      true
    }

get_data() output:
    {
        "Total In Count":  12,
        "Total Out Count": 10,
        "Active Tracks":   3
    }
"""

from __future__ import annotations

import datetime
import json
import os
import time as _time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .base import BaseModule

# ==================== CONFIG ====================

STATE_DIR = "state"
SAVE_INTERVAL_SEC = 30

COLOR_IN = (0, 255, 0)
COLOR_OUT = (0, 0, 255)
COLOR_NEUTRAL = (140, 140, 140)
COLOR_BOX_IN = (0, 255, 0)
COLOR_BOX_OUT = (0, 120, 255)


@dataclass
class _TrackRoiState:
    last_roi: Optional[str] = None
    pending_cross: Optional[str] = None


# ==================== MODULE ====================


class MarketClientCounterModule(BaseModule):
    """Type A — dual ROI crossing counted on track loss."""

    def __init__(
        self,
        name: str,
        polygon_in: list,
        polygon_out: list,
        target_class_id: int = 0,
        show_panel: bool = True,
        **_kwargs,
    ):
        self.name = name
        self.target_class_id = int(target_class_id)
        self.show_panel = bool(show_panel)

        pin = polygon_in or []
        pout = polygon_out or []
        if len(pin) < 3 or len(pout) < 3:
            raise ValueError("polygon_in and polygon_out each need at least 3 points")
        self._poly_in = np.array(pin, dtype=np.int32)
        self._poly_out = np.array(pout, dtype=np.int32)

        self._count_in = 0
        self._count_out = 0
        self._track_state: dict[int, _TrackRoiState] = {}
        self._prev_active: set[int] = set()
        self._last_reset_date: Optional[datetime.date] = None
        self._last_save_time = 0.0

        self._last_status = None

        self._load_state()
        print(f"✅ MarketClientCounterModule ready [{name}]")

    # ==================== PERSISTENCE ====================

    def _state_path(self) -> str:
        return os.path.join(STATE_DIR, f"market_client_counter_{self.name}.json")

    def _save_state(self):
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            state = {
                "date": (
                    self._last_reset_date.isoformat() if self._last_reset_date else None
                ),
                "count_in": self._count_in,
                "count_out": self._count_out,
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
            today = datetime.datetime.now().date().isoformat()
            if saved_date != today:
                print(f"[{self.name}] State outdated ({saved_date}), starting fresh.")
                return
            self._last_reset_date = datetime.date.fromisoformat(saved_date)
            self._count_in = int(state.get("count_in", 0))
            self._count_out = int(state.get("count_out", 0))
            print(
                f"[{self.name}] State loaded IN={self._count_in} OUT={self._count_out}"
            )
        except Exception as e:
            print(f"[{self.name}] State load error: {e} — starting fresh.")

    # ==================== HELPERS ====================

    def _check_daily_reset(self):
        today = datetime.datetime.now().date()
        if self._last_reset_date is None:
            self._last_reset_date = today
            return
        if today != self._last_reset_date:
            self._count_in = 0
            self._count_out = 0
            self._track_state.clear()
            self._prev_active.clear()
            self._last_reset_date = today
            self._save_state()
            print(f"[{self.name}] Daily reset → {today}")

    @staticmethod
    def _foot_of(bbox) -> tuple[int, int]:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        return ((x1 + x2) // 2, y2)

    def _which_roi(self, foot: tuple[int, int]) -> Optional[str]:
        in_out = cv2.pointPolygonTest(self._poly_out, foot, False) >= 0
        in_in = cv2.pointPolygonTest(self._poly_in, foot, False) >= 0
        if in_out:
            return "out"
        if in_in:
            return "in"
        return None

    def _on_roi_update(self, track_id: int, roi: Optional[str]) -> None:
        st = self._track_state.setdefault(track_id, _TrackRoiState())
        prev = st.last_roi
        if roi is not None and prev is not None and roi != prev:
            if prev == "out" and roi == "in":
                st.pending_cross = "in"
            elif prev == "in" and roi == "out":
                st.pending_cross = "out"
        if roi is not None:
            st.last_roi = roi

    def _apply_pending_cross(self, track_id: int, st: _TrackRoiState) -> None:
        if st.pending_cross == "in":
            self._count_in += 1
            print(
                f"[{self.name}] IN  +1  track={track_id}  (out→in, track lost)  "
                f"totals IN={self._count_in} OUT={self._count_out}"
            )
        elif st.pending_cross == "out":
            self._count_out += 1
            print(
                f"[{self.name}] OUT +1  track={track_id}  (in→out, track lost)  "
                f"totals IN={self._count_in} OUT={self._count_out}"
            )

    def _forget_track(self, track_id: int) -> None:
        st = self._track_state.get(track_id)
        if st is not None:
            self._apply_pending_cross(track_id, st)
        self._track_state.pop(track_id, None)

    # ==================== UPDATE ====================

    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        self._check_daily_reset()
        now = _time.time()

        curr_active: set[int] = set()
        box_draw: list[tuple[tuple[int, int, int, int], str | None, int]] = []

        if object_ids is not None and len(bboxes) > 0:
            n = len(bboxes)
            for i in range(n):
                if int(class_ids[i]) != self.target_class_id:
                    continue
                track_id = int(object_ids[i])
                curr_active.add(track_id)
                bbox = bboxes[i]
                foot = self._foot_of(bbox)
                roi = self._which_roi(foot)
                self._on_roi_update(track_id, roi)
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                box_draw.append(((x1, y1, x2, y2), roi, track_id))

        for lost_id in self._prev_active - curr_active:
            self._forget_track(lost_id)

        self._prev_active = curr_active

        self._last_status = {
            "count_in": self._count_in,
            "count_out": self._count_out,
            "active_tracks": len(curr_active),
            "boxes": box_draw,
        }

        if now - self._last_save_time >= SAVE_INTERVAL_SEC:
            self._save_state()
            self._last_save_time = now

    # ==================== DATA ====================

    def get_data(self) -> dict:
        return {
            "Total In Count": int(self._count_in),
            "Total Out Count": int(self._count_out),
            "Active Tracks": int(
                self._last_status["active_tracks"] if self._last_status else 0
            ),
        }

    # ==================== DRAW ====================

    def draw(self, frame):
        if self._last_status is None:
            return frame
        frame = self._draw_zones(frame)
        for (x1, y1, x2, y2), roi, track_id in self._last_status.get("boxes", []):
            if roi == "in":
                col = COLOR_BOX_IN
            elif roi == "out":
                col = COLOR_BOX_OUT
            else:
                col = COLOR_NEUTRAL
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(
                frame,
                f"id:{track_id}",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                col,
                2,
                cv2.LINE_AA,
            )
            foot = ((x1 + x2) // 2, y2)
            cv2.circle(frame, foot, 4, (255, 255, 255), -1)
        if self.show_panel:
            frame = self._draw_panel(frame)
        return frame

    def _draw_zones(self, frame):
        cv2.polylines(frame, [self._poly_in], True, COLOR_IN, 2)
        cv2.polylines(frame, [self._poly_out], True, COLOR_OUT, 2)
        cx, cy = int(self._poly_in[:, 0].mean()), int(self._poly_in[:, 1].mean())
        cv2.putText(
            frame,
            "IN",
            (cx - 15, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_IN,
            2,
            cv2.LINE_AA,
        )
        cx, cy = int(self._poly_out[:, 0].mean()), int(self._poly_out[:, 1].mean())
        cv2.putText(
            frame,
            "OUT",
            (cx - 20, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_OUT,
            2,
            cv2.LINE_AA,
        )
        return frame

    def _draw_panel(self, frame):
        st  = self._last_status
        pad = 10
        cv2.rectangle(frame, (pad, pad), (pad + 200, pad + 52), (0, 140, 255), -1)
        cv2.putText(
            frame,
            f"IN  {st['count_in']:>6}",
            (pad + 8, pad + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    # ==================== SHUTDOWN ====================

    def shutdown(self):
        self._save_state()

    def reset(self):
        self._count_in = 0
        self._count_out = 0
        self._track_state.clear()
        self._prev_active.clear()
        self._save_state()
