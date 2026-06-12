"""
Warehouse Freezer Monitor
-------------------------
YOLO classify on a fixed ROI crop — open/closed state with daily stats.

Type B — ignores pipeline bboxes; runs its own YOLO classification model.

Rules:
  • Median-smoothed labels + sustain_sec before stable state changes
  • open_count — transitions to open today (closed/unknown → open)
  • total open time — cumulative seconds open today (includes current session)
  • open_for_long_alert — True after alert_open_sec open, holds alert_duration_sec

Config example:
    {
        "type":                "warehouse_freezer",
        "name":                "warehouse_freezer_cam1",
        "polygon":             [[9, 355], [131, 287], [343, 971], [277, 1071]],
        "model_path":          "freezer_cls_best.pt",
        "sustain_sec":         2.0,
        "smooth_window":       15,
        "alert_open_sec":      300,
        "alert_duration_sec":  1800,
        "conf":                0.45,
        "imgsz":               640,
        "half":                true,
        "infer_every":         1,
        "timezone":            "Europe/London",
        "time_offset_hours":   0.0,
        "show_panel":          true
    }

get_data() output:
    {
        "Open Count Today":      3,
        "Total Open Minutes":    12.5,
        "Total Open Seconds":    750.0,
        "Stable State":          "open",
        "Frame Prediction":      "open",
        "Is Open For Long":      false,
        "open_for_long_alert":   false
    }
"""

from __future__ import annotations

import datetime
import json
import os
import time as _time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime as dt_datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from engine.shared_memory import GPU_LOCK

from .base import BaseModule

# ==================== CONFIG ====================

STATE_DIR = "state"
SAVE_INTERVAL_SEC = 30
ROI_PAD = 4
CROP_MIN_MEAN = 20.0
CROP_MIN_STD = 3.0

YOLO_TO_STATE = {
    "freezer_open": "open",
    "freezer_closed": "closed",
    "open": "open",
    "closed": "closed",
}

COLOR_POLYGON = (0, 200, 255)
COLOR_OVERLAY_BG = (2, 109, 253)
COLOR_OVERLAY_TEXT = (255, 255, 255)
COLOR_ALERT = (255, 255, 255)


@dataclass
class _FreezerStateMachine:
    smooth_window: int = 15
    sustain_sec: float = 2.0
    frame_preds: deque = field(default_factory=deque)
    stable_state: str = "unknown"
    candidate: str | None = None
    candidate_since: float = 0.0

    def __post_init__(self) -> None:
        self.frame_preds = deque(maxlen=self.smooth_window)

    def _smooth_label(self) -> str:
        if not self.frame_preds:
            return "unknown"
        counts = Counter(self.frame_preds)
        label, n = counts.most_common(1)[0]
        if n < max(3, len(self.frame_preds) // 2):
            return "unknown"
        return label

    def update(self, pred: str, wall_now: float) -> str:
        if pred in ("open", "closed"):
            self.frame_preds.append(pred)
        smooth = self._smooth_label()
        if smooth == "unknown":
            return self.stable_state
        if self.candidate != smooth:
            self.candidate = smooth
            self.candidate_since = wall_now
            return self.stable_state
        if wall_now - self.candidate_since < self.sustain_sec:
            return self.stable_state
        if smooth == self.stable_state:
            return self.stable_state
        prev = self.stable_state
        self.stable_state = smooth
        if prev != "unknown":
            print(f"  transition: {prev} → {smooth}")
        return self.stable_state


# ==================== MODULE ====================


class WarehouseFreezerModule(BaseModule):
    """Type B — YOLO classify on ROI crop; GPU_LOCK required."""

    def __init__(
        self,
        name: str,
        polygon: list,
        model_path: str = "freezer_cls_best.pt",
        sustain_sec: float = 2.0,
        smooth_window: int = 15,
        alert_open_sec: float = 300.0,
        alert_duration_sec: float = 1800.0,
        conf: float = 0.45,
        imgsz: int = 640,
        half: bool = True,
        infer_every: int = 1,
        timezone: str = "",
        time_offset_hours: float = 0.0,
        show_panel: bool = True,
        **_kwargs,
    ):
        self.name = name
        self.sustain_sec = float(sustain_sec)
        self.smooth_window = int(smooth_window)
        self.alert_open_sec = float(alert_open_sec)
        self.alert_duration_sec = float(alert_duration_sec)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.half = bool(half)
        self.infer_every = max(1, int(infer_every))
        self.timezone = timezone.strip() or None
        self.time_offset_hours = float(time_offset_hours)
        self.show_panel = bool(show_panel)

        if self.timezone:
            try:
                ZoneInfo(self.timezone)
            except ZoneInfoNotFoundError as exc:
                raise ValueError(f"Unknown timezone: {self.timezone!r}") from exc

        pts = polygon or []
        if len(pts) < 3:
            raise ValueError("polygon must have at least 3 points")
        self._poly = np.array(pts, dtype=np.int32)

        self._model = YOLO(model_path)
        self._sm = _FreezerStateMachine(
            smooth_window=self.smooth_window,
            sustain_sec=self.sustain_sec,
        )

        self._open_count = 0
        self._total_open_seconds = 0.0
        self._open_for_long = False
        self._current_open_started: Optional[dt_datetime] = None
        self._stable_freezer = "unknown"
        self._alert_logged = False
        self._alert_started_at: Optional[dt_datetime] = None
        self._alert_expired_this_session = False

        self._frame_pred = "unknown"
        self._frame_probs: dict[str, float] = {}
        self._frame_idx = 0
        self._should_alert = False

        self._last_reset_date: Optional[datetime.date] = None
        self._last_save_time = 0.0
        self._last_status = None

        self._load_state()
        print(f"✅ WarehouseFreezerModule ready [{name}]  model={model_path}")

    # ==================== PERSISTENCE ====================

    def _state_path(self) -> str:
        return os.path.join(STATE_DIR, f"warehouse_freezer_{self.name}.json")

    def _save_state(self):
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            state = {
                "date": (
                    self._last_reset_date.isoformat() if self._last_reset_date else None
                ),
                "open_count": self._open_count,
                "total_open_seconds": self._total_open_seconds,
                "open_for_long": self._open_for_long,
                "stable_freezer": self._stable_freezer,
                "current_open_started": (
                    self._current_open_started.isoformat(timespec="seconds")
                    if self._current_open_started
                    else None
                ),
                "alert_threshold_seconds": self.alert_open_sec,
                "alert_duration_seconds": self.alert_duration_sec,
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
            self._open_count = int(state.get("open_count", 0))
            self._total_open_seconds = float(state.get("total_open_seconds", 0))
            self._open_for_long = bool(state.get("open_for_long", False))
            self._stable_freezer = str(state.get("stable_freezer", "unknown"))
            cos = state.get("current_open_started")
            if cos:
                self._current_open_started = dt_datetime.fromisoformat(cos)
            if "alert_threshold_seconds" in state:
                self.alert_open_sec = float(state["alert_threshold_seconds"])
            if "alert_duration_seconds" in state:
                self.alert_duration_sec = float(state["alert_duration_seconds"])
            self._sm.stable_state = self._stable_freezer
            print(
                f"[{self.name}] State loaded opens={self._open_count}  "
                f"total_open_min={self._total_open_seconds / 60:.1f}"
            )
        except Exception as e:
            print(f"[{self.name}] State load error: {e} — starting fresh.")

    # ==================== HELPERS ====================

    def _now(self) -> dt_datetime:
        if self.timezone:
            base = dt_datetime.now(ZoneInfo(self.timezone)).replace(tzinfo=None)
        else:
            base = dt_datetime.now()
        if self.time_offset_hours:
            return base + timedelta(hours=self.time_offset_hours)
        return base

    def _check_daily_reset(self):
        today = datetime.datetime.now().date()
        if self._last_reset_date is None:
            self._last_reset_date = today
            return
        if today != self._last_reset_date:
            self._open_count = 0
            self._total_open_seconds = 0.0
            self._open_for_long = False
            self._current_open_started = None
            self._stable_freezer = "unknown"
            self._alert_logged = False
            self._alert_started_at = None
            self._alert_expired_this_session = False
            self._sm = _FreezerStateMachine(
                smooth_window=self.smooth_window,
                sustain_sec=self.sustain_sec,
            )
            self._last_reset_date = today
            self._save_state()
            print(f"[{self.name}] Daily reset → {today}")

    def _clear_alert_state(self) -> None:
        self._open_for_long = False
        self._should_alert = False
        self._alert_logged = False
        self._alert_started_at = None
        self._alert_expired_this_session = False

    @staticmethod
    def _crop_roi_rect(frame: np.ndarray, poly: np.ndarray, pad: int = ROI_PAD) -> np.ndarray:
        x, y, w, h = cv2.boundingRect(poly)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(frame.shape[1], x + w + pad)
        y1 = min(frame.shape[0], y + h + pad)
        return frame[y0:y1, x0:x1]

    @staticmethod
    def _is_bad_crop(crop: np.ndarray | None) -> bool:
        if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return True
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if float(gray.mean()) < CROP_MIN_MEAN:
            return True
        if float(gray.std()) < CROP_MIN_STD:
            return True
        return False

    @staticmethod
    def _remap_probs(raw: dict[str, float]) -> dict[str, float]:
        out = {"open": 0.0, "closed": 0.0}
        for name, prob in raw.items():
            key = YOLO_TO_STATE.get(name)
            if key in out:
                out[key] = max(out[key], float(prob))
        return out

    def _predict_crop(self, crop: np.ndarray) -> tuple[str, dict[str, float]]:
        with GPU_LOCK, torch.no_grad():
            results = self._model.predict(
                crop,
                imgsz=self.imgsz,
                half=self.half,
                verbose=False,
            )
        if not results:
            return "unknown", {}
        r = results[0]
        if r.probs is None:
            return "unknown", {}
        names = r.names
        raw = {names[i]: float(r.probs.data[i]) for i in range(len(names))}
        probs = self._remap_probs(raw)
        top_yolo = names[int(r.probs.top1)]
        label = YOLO_TO_STATE.get(top_yolo, "unknown")
        if float(r.probs.top1conf) < self.conf:
            label = "unknown"
        return label, probs

    def _on_stable_change(self, prev: str, new: str, now: dt_datetime) -> None:
        self._stable_freezer = new
        if new == "open" and prev in ("closed", "unknown"):
            if self._current_open_started is None:
                self._open_count += 1
                self._current_open_started = now
                self._clear_alert_state()
                print(
                    f"[{self.name}] FREEZER OPEN #{self._open_count}  "
                    f"{now.isoformat(timespec='seconds')}"
                )
        elif new == "closed" and prev == "open":
            if self._current_open_started is not None:
                self._total_open_seconds += (
                    now - self._current_open_started
                ).total_seconds()
            self._current_open_started = None
            self._clear_alert_state()
            print(f"[{self.name}] FREEZER CLOSED  {now.isoformat(timespec='seconds')}")

    def _tick_alert(self, now: dt_datetime, stable: str) -> None:
        self._stable_freezer = stable
        self._should_alert = False
        if stable != "open" or self._current_open_started is None:
            self._clear_alert_state()
            return
        if self._alert_expired_this_session:
            self._open_for_long = False
            return

        elapsed = (now - self._current_open_started).total_seconds()

        if self._alert_started_at is not None:
            alert_age = (now - self._alert_started_at).total_seconds()
            if alert_age >= self.alert_duration_sec:
                self._open_for_long = False
                self._should_alert = False
                self._alert_expired_this_session = True
                print(
                    f"[{self.name}] open_for_long cleared after "
                    f"{self.alert_duration_sec / 60:.0f} min (freezer still open)"
                )
            else:
                self._open_for_long = True
                self._should_alert = True
            return

        if elapsed >= self.alert_open_sec:
            self._open_for_long = True
            self._should_alert = True
            self._alert_started_at = now
            if not self._alert_logged:
                self._alert_logged = True
                print(
                    f"[{self.name}] ALERT open_for_long  open {elapsed / 60:.1f} min "
                    f"(threshold {self.alert_open_sec / 60:.1f} min)"
                )
        else:
            self._open_for_long = False

    def _total_open_seconds_live(self, now: dt_datetime) -> float:
        total = self._total_open_seconds
        if self._stable_freezer == "open" and self._current_open_started is not None:
            total += (now - self._current_open_started).total_seconds()
        return total

    # ==================== UPDATE ====================

    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        self._check_daily_reset()
        wall_now = _time.time()
        clock = self._now()
        self._frame_idx += 1

        pred = self._frame_pred
        probs = self._frame_probs
        should_infer = self.infer_every <= 1 or self._frame_idx % self.infer_every == 0

        if should_infer:
            crop = self._crop_roi_rect(frame, self._poly)
            if self._is_bad_crop(crop):
                pred, probs = "bad", {}
            else:
                pred, probs = self._predict_crop(crop)
                prev_stable = self._sm.stable_state
                stable = self._sm.update(pred if pred != "bad" else "unknown", wall_now)
                if stable != prev_stable:
                    self._on_stable_change(prev_stable, stable, clock)

        stable = self._sm.stable_state
        self._tick_alert(clock, stable)

        self._frame_pred = pred
        self._frame_probs = probs

        self._last_status = {
            "pred": pred,
            "probs": probs,
            "stable": stable,
            "open_count": self._open_count,
            "open_for_long": self._open_for_long,
            "clock": clock,
        }

        if wall_now - self._last_save_time >= SAVE_INTERVAL_SEC:
            self._save_state()
            self._last_save_time = wall_now

    # ==================== DATA ====================

    def get_data(self) -> dict:
        now = self._now()
        live_total = self._total_open_seconds_live(now)
        open_duration = 0.0
        if self._current_open_started is not None:
            open_duration = (now - self._current_open_started).total_seconds()
        return {
            "Open Count Today": int(self._open_count),
            "Total Open Minutes": round(live_total / 60.0, 2),
            "Total Open Seconds": round(live_total, 1),
            "Open Duration Minutes": round(open_duration / 60.0, 2),
            "Stable State": self._stable_freezer,
            "Frame Prediction": self._frame_pred,
            "Is Open For Long": bool(self._open_for_long),
            "open_for_long_alert": bool(self._should_alert),
        }

    # ==================== DRAW ====================

    def draw(self, frame):
        if self._last_status is None:
            return frame
        frame = self._draw_zones(frame)
        if self.show_panel:
            frame = self._draw_panel(frame)
        return frame

    def _draw_zones(self, frame):
        cv2.polylines(frame, [self._poly], True, COLOR_POLYGON, 2)
        x, y, w, h = cv2.boundingRect(self._poly)
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_POLYGON, 1)
        return frame

    def _draw_panel(self, frame):
        st = self._last_status
        now = st["clock"]
        pred = st["pred"]
        stable = st["stable"]
        probs = st["probs"]
        pad = 6
        box_w = 260
        fw = frame.shape[1]
        prob = probs.get(stable if stable in probs else pred, 0.0) if probs else 0.0
        live_min = self._total_open_seconds_live(now) / 60.0
        alert_min = self.alert_open_sec / 60.0
        hold_min = self.alert_duration_sec / 60.0

        lines = [
            f"FREEZER  {now.strftime('%H:%M:%S')}",
            f"pred: {pred} ({prob:.0%})  stable: {stable}",
            f"opens today: {st['open_count']}",
            f"total open: {live_min:.1f} min",
            (
                f"open_for_long: {st['open_for_long']}  "
                f"(after {alert_min:.0f}m, holds {hold_min:.0f}m)"
            ),
        ]
        if st["open_for_long"]:
            lines.append("*** ALERT: FREEZER OPEN TOO LONG ***")

        line_h = 15
        box_h = 12 + len(lines) * line_h
        x1 = fw - box_w - pad
        y1 = pad
        cv2.rectangle(frame, (x1, y1), (fw - pad, y1 + box_h), COLOR_OVERLAY_BG, -1)
        for i, line in enumerate(lines):
            color = COLOR_ALERT if "ALERT" in line else COLOR_OVERLAY_TEXT
            cv2.putText(
                frame,
                line[:48],
                (x1 + 6, y1 + 13 + i * line_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                color,
                1,
                cv2.LINE_AA,
            )
        return frame

    # ==================== SHUTDOWN ====================

    def shutdown(self):
        now = self._now()
        if self._stable_freezer == "open" and self._current_open_started is not None:
            self._total_open_seconds += (
                now - self._current_open_started
            ).total_seconds()
            self._current_open_started = None
        self._save_state()

    def reset(self):
        self._open_count = 0
        self._total_open_seconds = 0.0
        self._current_open_started = None
        self._clear_alert_state()
        self._stable_freezer = "unknown"
        self._sm = _FreezerStateMachine(
            smooth_window=self.smooth_window,
            sustain_sec=self.sustain_sec,
        )
        self._save_state()
