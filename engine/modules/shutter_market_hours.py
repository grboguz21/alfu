"""
Shutter Market Hours
--------------------
Store open/close times from YOLO classification on a shutter ROI crop (Type B).

Ignores pipeline bboxes. Runs a dedicated classify model on the bounding-rect
crop of a configured polygon. Stable closed→open records opening time;
stable open→closed records closing time.

Production schedule (default):
- 08:00–09:30  YOLO on until opening time saved
- 20:30–22:00  YOLO on until closing time saved
- development=true skips time windows (playback / testing)

Config example:
    {
        "type":                "shutter_market_hours",
        "name":                "shutter_market_hours_cam1",
        "polygon":             [[120, 80], [600, 80], [600, 400], [120, 400]],
        "model_path":          "models/shutter_cls_best.pt",
        "open_window_start":   "08:00",
        "open_window_end":     "09:30",
        "close_window_start":  "20:30",
        "close_window_end":    "22:00",
        "development":         false,
        "timezone":            "Europe/London",
        "time_offset_hours":   0.0,
        "conf":                0.45,
        "imgsz":               224,
        "sustain_sec":         5.0,
        "smooth_window":       15,
        "show_panel":          true
    }

get_data() output:
    {
        "Opening Time":      "2026-06-08T08:12:00" or "",
        "Closing Time":      "2026-06-08T20:45:00" or "",
        "Stable Shutter":    "open",
        "YOLO Active":       true,
        "Schedule Status":   "open window"
    }
"""

from __future__ import annotations

import datetime
import json
import os
import time as _time
from collections import Counter, deque
from datetime import datetime as dt_datetime
from datetime import time as dt_time
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
MIN_CROP_SIZE = 8
MIN_CROP_MEAN = 20.0
MIN_CROP_STD = 3.0

COLOR_ROI = (0, 200, 255)
OVERLAY_BG = (2, 109, 253)   # #fd6d02 BGR
OVERLAY_TEXT = (255, 255, 255)
PANEL_WIDTH = 400

YOLO_TO_STATE = {
    "closed": "closed",
    "market_open": "open",
    "person": "person",
    "open": "open",
}


# ==================== HELPERS ====================


def _parse_hhmm(value: str) -> dt_time:
    s = str(value).strip()
    if ":" in s:
        h, m = s.split(":", 1)
        return dt_time(int(h) % 24, int(m) % 60)
    return dt_time(int(s) % 24, 0)


def _time_in_range(t: dt_time, start: dt_time, end: dt_time) -> bool:
    return start <= t < end


def _crop_roi_rect(frame: np.ndarray, poly: np.ndarray, pad: int) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(poly)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(frame.shape[1], x + w + pad)
    y1 = min(frame.shape[0], y + h + pad)
    return frame[y0:y1, x0:x1].copy()


def _is_bad_crop(crop: np.ndarray | None) -> bool:
    if crop is None or crop.size == 0:
        return True
    if crop.shape[0] < MIN_CROP_SIZE or crop.shape[1] < MIN_CROP_SIZE:
        return True
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) < MIN_CROP_MEAN:
        return True
    if float(gray.std()) < MIN_CROP_STD:
        return True
    return False


def _remap_probs(raw: dict[str, float]) -> dict[str, float]:
    out = {"closed": 0.0, "open": 0.0, "person": 0.0}
    for name, prob in raw.items():
        key = YOLO_TO_STATE.get(name)
        if key:
            out[key] = max(out[key], float(prob))
    return out


class _ShutterStateMachine:
    def __init__(self, smooth_window: int, sustain_sec: float):
        self.smooth_window = max(3, int(smooth_window))
        self.sustain_sec = float(sustain_sec)
        self.frame_preds: deque[str] = deque(maxlen=self.smooth_window)
        self.stable_state = "unknown"
        self.candidate: str | None = None
        self.candidate_since = 0.0
        self.person_since: float | None = None
        self.opened_at: Optional[dt_datetime] = None
        self.closed_at: Optional[dt_datetime] = None

    def _smooth_label(self) -> str:
        if not self.frame_preds:
            return "unknown"
        counts = Counter(self.frame_preds)
        label, n = counts.most_common(1)[0]
        if n < max(3, len(self.frame_preds) // 2):
            return "unknown"
        return label

    def update(self, pred: str, wall_now: float, ts: dt_datetime) -> str:
        if pred == "person":
            if self.person_since is None:
                self.person_since = wall_now
            if wall_now - self.person_since < 2.0:
                return self.stable_state
        else:
            self.person_since = None
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
        if smooth == "open":
            self.opened_at = ts
        elif smooth == "closed":
            self.closed_at = ts
        if prev != "unknown":
            print(f"[shutter] {prev} → {smooth} @ {ts.strftime('%H:%M:%S')}")
        return self.stable_state


# ==================== MODULE ====================


class ShutterMarketHoursModule(BaseModule):
    """Type B — YOLO classify on shutter ROI crop."""

    def __init__(
        self,
        name: str,
        polygon: list,
        model_path: str = "models/shutter_cls_best.pt",
        open_window_start: str = "08:00",
        open_window_end: str = "09:30",
        close_window_start: str = "20:30",
        close_window_end: str = "22:00",
        development: bool = False,
        timezone: str = "Europe/London",
        time_offset_hours: float = 0.0,
        conf: float = 0.45,
        imgsz: int = 224,
        sustain_sec: float = 5.0,
        smooth_window: int = 15,
        roi_pad: int = ROI_PAD,
        show_panel: bool = True,
        **_kwargs,
    ):
        self.name = name
        self.model_path = model_path
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.roi_pad = int(roi_pad)
        self.show_panel = bool(show_panel)
        self.development = bool(development)
        self.time_offset_hours = float(time_offset_hours)
        self.timezone = timezone.strip() or None

        if self.timezone:
            try:
                ZoneInfo(self.timezone)
            except ZoneInfoNotFoundError as exc:
                raise ValueError(f"Unknown timezone: {self.timezone!r}") from exc

        pts = polygon or []
        if len(pts) < 3:
            raise ValueError("polygon must have at least 3 points")
        self._poly = np.array(pts, dtype=np.int32)

        self._open_start = _parse_hhmm(open_window_start)
        self._open_end = _parse_hhmm(open_window_end)
        self._close_start = _parse_hhmm(close_window_start)
        self._close_end = _parse_hhmm(close_window_end)
        if self._open_start >= self._open_end:
            raise ValueError("open_window_start must be before open_window_end")
        if self._close_start >= self._close_end:
            raise ValueError("close_window_start must be before close_window_end")

        self._opening_time: Optional[dt_datetime] = None
        self._closing_time: Optional[dt_datetime] = None
        self._last_reset_date: Optional[datetime.date] = None
        self._last_save_time = 0.0

        self._opened_just_now = False
        self._closed_just_now = False

        self._state_machine = _ShutterStateMachine(smooth_window, sustain_sec)
        self._model = YOLO(self.model_path)
        self._last_status = None

        self._load_state()
        print(f"✅ ShutterMarketHoursModule ready [{name}]  model={self.model_path}")

    # ==================== PERSISTENCE ====================

    def _state_path(self) -> str:
        return os.path.join(STATE_DIR, f"shutter_market_hours_{self.name}.json")

    def _save_state(self):
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            state = {
                "date": (
                    self._last_reset_date.isoformat() if self._last_reset_date else None
                ),
                "opening_time": (
                    self._opening_time.isoformat(timespec="seconds")
                    if self._opening_time
                    else None
                ),
                "closing_time": (
                    self._closing_time.isoformat(timespec="seconds")
                    if self._closing_time
                    else None
                ),
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
            ot = state.get("opening_time")
            ct = state.get("closing_time")
            if ot:
                self._opening_time = dt_datetime.fromisoformat(ot)
            if ct:
                self._closing_time = dt_datetime.fromisoformat(ct)
            print(f"[{self.name}] State loaded")
        except Exception as e:
            print(f"[{self.name}] State load error: {e} — starting fresh.")

    # ==================== HELPERS ====================

    def _now(self) -> dt_datetime:
        if self.timezone:
            base = dt_datetime.now(ZoneInfo(self.timezone)).replace(tzinfo=None)
        else:
            base = dt_datetime.now()
        if self.time_offset_hours:
            base += datetime.timedelta(hours=self.time_offset_hours)
        return base

    def _check_daily_reset(self):
        today = datetime.datetime.now().date()
        if self._last_reset_date is None:
            self._last_reset_date = today
            return
        if today != self._last_reset_date:
            self._opening_time = None
            self._closing_time = None
            self._state_machine = _ShutterStateMachine(
                self._state_machine.smooth_window,
                self._state_machine.sustain_sec,
            )
            self._last_reset_date = today
            self._save_state()
            print(f"[{self.name}] Daily reset → {today}")

    def _schedule_label(self) -> str:
        if self.development:
            return "development (no time gates)"
        return (
            f"open {self._open_start.strftime('%H:%M')}–{self._open_end.strftime('%H:%M')}  |  "
            f"close {self._close_start.strftime('%H:%M')}–{self._close_end.strftime('%H:%M')}"
        )

    def _should_infer(self, now: dt_datetime) -> tuple[bool, str]:
        if self.development:
            return True, "development"

        t = now.time()
        if self._opening_time and self._closing_time:
            return False, "done for today"

        if self._opening_time is None:
            if _time_in_range(t, self._open_start, self._open_end):
                return True, "open window"
            if _time_in_range(t, self._close_start, self._close_end):
                return True, "close window"
            if t < self._open_start:
                return False, "wait open window"
            if t < self._close_start:
                return False, "wait close window"
            return False, "after close window"

        if self._closing_time is None:
            if _time_in_range(t, self._close_start, self._close_end):
                return True, "close window"
            if t < self._close_start:
                return False, "wait close window"
            return False, "after close window"

        return False, "idle"

    def _on_stable_change(self, prev: str, new: str, now: dt_datetime) -> None:
        if new == "open" and prev == "closed":
            self._opening_time = now
            self._opened_just_now = True
            print(f"  OPENING  {now.isoformat(timespec='seconds')}  closed→open")
        if new == "closed" and prev == "open":
            self._closing_time = now
            self._closed_just_now = True
            print(f"  CLOSING  {now.isoformat(timespec='seconds')}  open→closed")

    def _predict_crop(self, crop: np.ndarray) -> tuple[str, dict[str, float]]:
        with GPU_LOCK, torch.no_grad():
            results = self._model.predict(
                crop, imgsz=self.imgsz, verbose=False, conf=self.conf
            )
        if not results:
            return "unknown", {}
        r = results[0]
        if r.probs is None:
            return "unknown", {}
        names = r.names
        raw = {names[i]: float(r.probs.data[i]) for i in range(len(names))}
        probs = _remap_probs(raw)
        top_yolo = names[int(r.probs.top1)]
        label = YOLO_TO_STATE.get(top_yolo, "unknown")
        if float(r.probs.top1conf) < self.conf:
            label = "unknown"
        return label, probs

    # ==================== UPDATE ====================

    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        self._check_daily_reset()
        now = _time.time()
        ts = self._now()
        infer_on, idle_reason = self._should_infer(ts)

        pred = "—"
        probs: dict[str, float] = {}
        stable = self._state_machine.stable_state

        if infer_on:
            crop = _crop_roi_rect(frame, self._poly, self.roi_pad)
            if not _is_bad_crop(crop):
                pred, probs = self._predict_crop(crop)
                prev_stable = self._state_machine.stable_state
                stable = self._state_machine.update(pred, now, ts)
                if stable != prev_stable:
                    self._on_stable_change(prev_stable, stable, ts)

        self._last_status = {
            "infer_on": infer_on,
            "idle_reason": idle_reason,
            "pred": pred,
            "probs": probs,
            "stable": stable,
            "opening": self._opening_time,
            "closing": self._closing_time,
            "schedule": self._schedule_label(),
            "clock": ts.strftime("%H:%M:%S"),
        }

        if now - self._last_save_time >= SAVE_INTERVAL_SEC:
            self._save_state()
            self._last_save_time = now

    # ==================== DATA ====================

    def get_data(self) -> dict:
        st = self._last_status or {}
        opening = self._opening_time
        closing = self._closing_time
        opened_alert = self._opened_just_now
        closed_alert = self._closed_just_now
        self._opened_just_now = False
        self._closed_just_now = False
        return {
            "Opening Time": opening.isoformat(timespec="seconds") if opening else "",
            "Closing Time": closing.isoformat(timespec="seconds") if closing else "",
            "Stable Shutter": st.get("stable", self._state_machine.stable_state),
            "Frame Prediction": st.get("pred", ""),
            "YOLO Active": bool(st.get("infer_on", False)),
            "Schedule Status": st.get("idle_reason", ""),
            "Open Window": (
                f"{self._open_start.strftime('%H:%M')}-{self._open_end.strftime('%H:%M')}"
            ),
            "Close Window": (
                f"{self._close_start.strftime('%H:%M')}-{self._close_end.strftime('%H:%M')}"
            ),
            "Development Mode": self.development,
            "shutter_opened_alert": opened_alert,
            "shutter_closed_alert": closed_alert,
        }

    # ==================== DRAW ====================

    def draw(self, frame):
        if self._last_status is None:
            return frame
        cv2.polylines(frame, [self._poly], True, COLOR_ROI, 2)
        x, y, w, h = cv2.boundingRect(self._poly)
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_ROI, 1)
        return frame

    def _draw_panel(self, frame):
        st = self._last_status
        if st is None:
            return frame

        fw = frame.shape[1]
        pad = 10
        infer_on = st["infer_on"]
        idle_reason = st["idle_reason"]
        pred = st["pred"]
        probs = st.get("probs") or {}
        stable = st["stable"]
        opening = st["opening"]
        closing = st["closing"]

        prob = probs.get(pred if pred in probs else stable, 0.0)
        if pred in ("—", "unknown", "bad"):
            prob = probs.get(stable, 0.0)

        mode = "YOLO" if infer_on else f"IDLE ({idle_reason})"
        lines = [
            f"{mode}  {st['clock']}",
            st["schedule"],
            f"pred: {pred} ({prob:.0%})  stable: {stable}",
            f"opening: {opening.strftime('%H:%M:%S') if opening else '—'}",
            f"closing: {closing.strftime('%H:%M:%S') if closing else '—'}",
        ]

        box_h = 16 + len(lines) * 20
        x1 = fw - PANEL_WIDTH - pad
        y1 = pad
        cv2.rectangle(frame, (x1, y1), (fw - pad, y1 + box_h), OVERLAY_BG, -1)
        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line[:64],
                (x1 + 8, y1 + 18 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                OVERLAY_TEXT,
                1,
                cv2.LINE_AA,
            )
        return frame

    # ==================== SHUTDOWN ====================

    def shutdown(self):
        self._save_state()

    def reset(self):
        self._opening_time = None
        self._closing_time = None
        self._state_machine = _ShutterStateMachine(
            self._state_machine.smooth_window,
            self._state_machine.sustain_sec,
        )
        self._save_state()
