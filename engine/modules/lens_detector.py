import time as _time
import datetime
import json
import os
from collections import deque

import cv2
import numpy as np

from .base import BaseModule

# ── Persistence ──────────────────────────────────────────────────
STATE_DIR         = "state"
STATE_FILE        = "lens_detector_state.json"
SAVE_INTERVAL_SEC = 30   # Write state to disk every 30 seconds

# ── Visual constants ─────────────────────────────────────────────
COLOR_OK       = (0, 200, 0)       # Green — lens open
COLOR_ALERT    = (0, 0, 220)       # Red — lens closed
COLOR_WARNING  = (0, 165, 255)     # Orange — warning
LOG_FONT_SCALE = 0.52
LOG_THICKNESS  = 1
LOG_COLOR      = (220, 220, 220)
LOG_BG_COLOR   = (30, 30, 30)
LOG_X          = 10
LOG_Y_BOTTOM   = 40
MAX_LOG_LINES  = 8


class LensDetectorModule(BaseModule):
    """
    Detects whether the camera lens has been covered.

    This is a special module with no model of its own that also does not use
    bboxes from the system tracker. Inside update() it runs only classical
    image-processing techniques on `frame`.
    The parameters `bboxes, class_ids, scores, object_ids, class_names` are
    ignored (signature is preserved).

    Detection Methods:
        Daytime (3 of 5 criteria):
            - Brightness drop (adaptive baseline)
            - Edge loss (Canny edge)
            - Low variance
            - Histogram entropy
            - Blur (Laplacian)
        Night (2 of 3 criteria):
            - Histogram entropy
            - IR reflection
            - Low variance
        Special case:
            - Bright object detection → Direct alarm

    Config example:
        {
            "type":            "lens_detector",
            "name":            "camera1_lens",
            "cooldown":        3600,
            "blur_threshold":  100.0,
            "resize_factor":   10,
            "night_threshold": 30.0,
            "baseline_window": 50,
            "show_panel":      true
        }
    """

    def __init__(self, name: str,
                 cooldown: int = 3600,
                 blur_threshold: float = 100.0,
                 resize_factor: int = 10,
                 night_threshold: float = 30.0,
                 baseline_window: int = 50,
                 alert_delay_sec: float = 10.0,
                 show_panel: bool = True):
        self.name            = name
        self.cooldown        = cooldown
        self.blur_threshold  = blur_threshold
        self.resize_factor   = resize_factor
        self.night_threshold = night_threshold
        self.alert_delay_sec = alert_delay_sec
        self.show_panel      = show_panel

        # ── Detection state ───────────────────────────────────────
        self.brightness_history  = deque(maxlen=baseline_window)
        self.baseline_brightness = None
        self.last_alert_time     = None      # datetime
        self.last_was_covered    = False
        self.covered_start_time  = None      # datetime
        self._alerted_this_cover = False
        self._last_reset_date    = None
        self._daily_alert_count  = 0

        # Last status computed in update() (for draw)
        self._last_info         = None
        self._last_is_covered   = False
        self._last_should_alert = False

        # Screen log
        self._screen_logs = deque(maxlen=MAX_LOG_LINES * 3)

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # ── Persistence ──────────────────────────────────────────
        self._last_save_time = 0.0
        self._load_state()

        print(f"✅ Lens Detector ready [{self.name}]")
        print(f"   ├── Cooldown: {cooldown}s ({cooldown//60} min)")
        print(f"   ├── Resize: 1/{resize_factor}")
        print(f"   └── Blur threshold: {blur_threshold}")

    # ── Persistence ──────────────────────────────────────────────

    def _state_path(self) -> str:
        return os.path.join(STATE_DIR, STATE_FILE)

    def _save_state(self):
        """Write cooldown and daily counter state to the shared JSON file."""
        try:
            os.makedirs(STATE_DIR, exist_ok=True)

            my_state = {
                "date":              (self._last_reset_date.isoformat()
                                       if self._last_reset_date else None),
                "last_alert_time":   (self.last_alert_time.isoformat()
                                       if self.last_alert_time else None),
                "last_was_covered":  self.last_was_covered,
                "covered_start_time": (self.covered_start_time.isoformat()
                                       if self.covered_start_time else None),
                "alerted_this_cover": self._alerted_this_cover,
                "daily_alert_count": self._daily_alert_count,
            }

            path = self._state_path()
            all_state = {}
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        all_state = json.load(f)
                    if not isinstance(all_state, dict):
                        all_state = {}
                except Exception:
                    all_state = {}

            all_state[self.name] = my_state

            tmp_path = path + f".{os.getpid()}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(all_state, f, ensure_ascii=False, indent=2)
            for _attempt in range(5):
                try:
                    os.replace(tmp_path, path)
                    break
                except OSError:
                    if _attempt < 4:
                        _time.sleep(0.05)
                    else:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                        raise
        except Exception as e:
            print(f"[{self.name}] State save error: {e}")

    def _load_state(self):
        """Only state from TODAY is loaded — midnight reset principle."""
        path = self._state_path()
        if not os.path.exists(path):
            print(f"[{self.name}] State file not found, starting fresh.")
            return

        try:
            with open(path, encoding="utf-8") as f:
                all_state = json.load(f)

            if not isinstance(all_state, dict) or self.name not in all_state:
                print(f"[{self.name}] No record for this camera in the shared state file, "
                      f"starting fresh.")
                return

            state = all_state[self.name]
            saved_date_str = state.get("date")
            today_str      = datetime.datetime.now().date().isoformat()

            if saved_date_str != today_str:
                print(f"[{self.name}] State is outdated ({saved_date_str}), "
                      f"starting fresh for today ({today_str}).")
                return

            last_alert = state.get("last_alert_time")
            if last_alert:
                self.last_alert_time = datetime.datetime.fromisoformat(last_alert)

            covered_start = state.get("covered_start_time")
            if covered_start:
                self.covered_start_time = datetime.datetime.fromisoformat(covered_start)

            self.last_was_covered   = bool(state.get("last_was_covered", False))
            self._alerted_this_cover = bool(state.get("alerted_this_cover", False))
            self._daily_alert_count = int(state.get("daily_alert_count", 0))
            self._last_reset_date   = datetime.date.fromisoformat(saved_date_str)

            print(f"[{self.name}] State loaded (date: {saved_date_str}, "
                  f"daily alerts: {self._daily_alert_count}, "
                  f"currently covered: {self.last_was_covered})")

        except Exception as e:
            print(f"[{self.name}] State load error: {e} — starting fresh.")

    # ── Helpers ───────────────────────────────────────────────────

    def _add_log(self, msg, color=None):
        ts    = datetime.datetime.now().strftime("%H:%M:%S")
        color = color or LOG_COLOR
        self._screen_logs.append({"time": ts, "msg": msg, "color": color})
        print(f"[{ts}] [{self.name}] {msg}")

    def _check_daily_reset(self):
        today = datetime.datetime.now().date()
        if self._last_reset_date is None:
            self._last_reset_date = today
            return
        if today != self._last_reset_date:
            self._reset_daily()

    def _reset_daily(self):
        self._daily_alert_count = 0
        self._last_reset_date   = datetime.datetime.now().date()
        self._add_log(
            f"Daily reset -> {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')}",
            color=(255, 200, 0),
        )
        self._save_state()

    # ── Detection methods ─────────────────────────────────────────

    def _check_brightness_adaptive(self, frame):
        """Adaptive brightness check"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_brightness = float(np.mean(gray))

        self.brightness_history.append(current_brightness)
        if len(self.brightness_history) >= 10:
            self.baseline_brightness = float(np.median(list(self.brightness_history)))

        is_night = False
        if self.baseline_brightness is not None:
            is_night = self.baseline_brightness < self.night_threshold
        else:
            is_night = current_brightness < self.night_threshold

        is_covered         = False
        brightness_drop    = 0.0
        adaptive_threshold = 10.0

        if self.baseline_brightness is not None:
            drop_threshold_percent = 0.15
            adaptive_threshold = self.baseline_brightness * drop_threshold_percent

            if is_night:
                adaptive_threshold = max(adaptive_threshold, 2)
            else:
                adaptive_threshold = max(adaptive_threshold, 8)

            is_covered = current_brightness < adaptive_threshold
            brightness_drop = ((self.baseline_brightness - current_brightness)
                               / self.baseline_brightness * 100) \
                              if self.baseline_brightness > 0 else 0
        else:
            is_covered = current_brightness < 10

        return is_covered, {
            'brightness':   current_brightness,
            'baseline':     self.baseline_brightness,
            'threshold':    adaptive_threshold,
            'is_night':     is_night,
            'drop_percent': brightness_drop,
        }

    def _check_variance(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = float(np.var(gray))
        return variance < 50, variance

    def _check_edges_adaptive(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        if mean_brightness < 30:
            edges = cv2.Canny(gray, 20, 60)
            min_edge_ratio = 0.005
        else:
            edges = cv2.Canny(gray, 50, 150)
            min_edge_ratio = 0.01

        edge_count   = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_ratio   = edge_count / total_pixels

        return edge_ratio < min_edge_ratio, {
            'edge_ratio':       edge_ratio,
            'threshold':        min_edge_ratio,
            'brightness_level': 'dark' if mean_brightness < 30 else 'bright',
        }

    def _check_histogram(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist.flatten() / (hist.sum() + 1e-10)

        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))

        pixel_values   = np.arange(256)
        mean_intensity = np.sum(pixel_values * hist_normalized)
        std_intensity  = np.sqrt(np.sum(((pixel_values - mean_intensity) ** 2) * hist_normalized))

        low_entropy = entropy < 4.0
        low_std     = std_intensity < 15

        return low_entropy and low_std, {
            'entropy': float(entropy),
            'std':     float(std_intensity),
            'mean':    float(mean_intensity),
        }

    def _check_ir_reflection(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bright_pixels = np.sum(gray > 200)
        total_pixels  = gray.shape[0] * gray.shape[1]
        bright_ratio  = bright_pixels / total_pixels

        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        center_mean   = float(np.mean(center_region))

        ir_reflection = bright_ratio > 0.1 and center_mean > 150

        return ir_reflection, {
            'bright_ratio': float(bright_ratio),
            'center_mean':  center_mean,
        }

    def _check_bright_occlusion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        variance        = float(np.var(gray))

        is_bright  = mean_brightness > 100
        is_flat    = variance < 50
        is_covered = is_bright and is_flat

        return is_covered, {
            'mean':      mean_brightness,
            'variance':  variance,
            'is_bright': is_bright,
            'is_flat':   is_flat,
        }

    def _check_blur(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < self.blur_threshold, float(laplacian_var)

    # ── Main update ───────────────────────────────────────────────

    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        # bboxes, class_ids, scores, object_ids, class_names → not used
        self._check_daily_reset()

        # Downscale frame (for performance)
        h, w = frame.shape[:2]
        small_h = max(1, h // self.resize_factor)
        small_w = max(1, w // self.resize_factor)
        small_frame = cv2.resize(frame, (small_w, small_h))

        # Run all checks
        brightness_covered, brightness_info = self._check_brightness_adaptive(small_frame)
        variance_covered,   variance_val    = self._check_variance(small_frame)
        edge_covered,       edge_info       = self._check_edges_adaptive(small_frame)
        histogram_covered,  histogram_info  = self._check_histogram(small_frame)
        ir_covered,         ir_info         = self._check_ir_reflection(small_frame)
        bright_occlusion,   bright_info     = self._check_bright_occlusion(small_frame)
        blur_covered,       blur_val        = self._check_blur(small_frame)

        is_night = brightness_info.get('is_night', False)

        # ── Decision logic ────────────────────────────────────────
        if bright_occlusion:
            lens_covered      = True
            mode              = 'bright_blockage'
            reason            = 'Bright object detected'
            criteria_met      = 1
            criteria_required = 1

        elif is_night:
            criteria          = [histogram_covered, ir_covered, variance_covered]
            criteria_met      = sum(criteria)
            criteria_required = 2
            lens_covered      = criteria_met >= criteria_required
            mode              = 'night'
            reason            = 'Night mode criteria'

        else:
            criteria          = [brightness_covered, edge_covered, variance_covered,
                                 histogram_covered, blur_covered]
            criteria_met      = sum(criteria)
            criteria_required = 3

            # Special case: very blurry + low variance
            if blur_covered and variance_val < 1000:
                criteria_met = criteria_required

            lens_covered = criteria_met >= criteria_required
            mode         = 'day'
            reason       = 'Day mode criteria'

        # ── Alert / cooldown ──────────────────────────────────────
        should_alert = False
        current_time = datetime.datetime.now()

        # Duration lens has been covered
        covered_duration = 0.0

        if lens_covered:
            if not self.last_was_covered:
                # First closure detected — start the timer, don't alert yet
                self.covered_start_time  = current_time
                self.last_was_covered    = True
                self._alerted_this_cover = False
                self._add_log(
                    f"LENS COVER DETECTED ({mode}) | {reason} | "
                    f"Criteria {criteria_met}/{criteria_required} | "
                    f"waiting {self.alert_delay_sec:.0f}s",
                    color=(0, 165, 255),
                )

            if self.covered_start_time:
                covered_duration = (current_time - self.covered_start_time).total_seconds()

            if not self._alerted_this_cover:
                if covered_duration >= self.alert_delay_sec:
                    should_alert             = True
                    self.last_alert_time     = current_time
                    self._alerted_this_cover = True
                    self._daily_alert_count += 1
                    self._add_log(
                        f"LENS COVERED ({mode}) for {self.alert_delay_sec:.0f}s+ | "
                        f"{reason} | Criteria {criteria_met}/{criteria_required}",
                        color=(0, 0, 255),
                    )
            elif self.last_alert_time is not None:
                # Still covered — cooldown check for repeat alerts
                elapsed = (current_time - self.last_alert_time).total_seconds()
                if elapsed >= self.cooldown:
                    should_alert             = True
                    self.last_alert_time     = current_time
                    self._daily_alert_count += 1
                    self._add_log(
                        f"LENS STILL COVERED (repeat alert after cooldown) | "
                        f"Total alerts: {self._daily_alert_count}",
                        color=(0, 0, 255),
                    )
        else:
            if self.last_was_covered:
                self._add_log("Lens is open again", color=(0, 255, 120))
            self.last_was_covered    = False
            self.covered_start_time  = None
            self._alerted_this_cover = False

        # Store last status for draw()
        self._last_is_covered   = lens_covered
        self._last_should_alert = should_alert
        self._last_info = {
            'lens_covered':      lens_covered,
            'should_alert':      should_alert,
            'mode':              mode,
            'reason':            reason,
            'is_night':          is_night,
            'criteria_met':      criteria_met,
            'criteria_required': criteria_required,
            'brightness':        brightness_info,
            'variance':          variance_val,
            'edges':             edge_info,
            'histogram':         histogram_info,
            'ir':                ir_info,
            'blur':              blur_val,
            'bright_occlusion':  bright_info,
            'covered_duration':  covered_duration,
        }

        # Periodic state save
        now_ts = _time.time()
        if now_ts - self._last_save_time >= SAVE_INTERVAL_SEC:
            self._save_state()
            self._last_save_time = now_ts

    # ── Data ──────────────────────────────────────────────────────

    def get_data(self) -> dict:
        """
        Returns a flat dict. main.py can use this to send alerts:
            data = modules.get_data()
            if data.get("should_alert"):
                report.send_alarm("lens_covered_alarm", data=data, media_path=...)
        """
        info = self._last_info or {}
        br   = info.get("brightness", {}) if isinstance(info.get("brightness"), dict) else {}
        return {
            "lens_covered":      bool(self._last_is_covered),
            "should_alert":      bool(self._last_should_alert),
            "mode":              info.get("mode", "unknown"),
            "criteria_met":      int(info.get("criteria_met", 0)),
            "criteria_required": int(info.get("criteria_required", 0)),
            "brightness":        float(br.get("brightness", 0.0)),
            "baseline":          float(br.get("baseline") or 0.0),
            "is_night":          bool(info.get("is_night", False)),
            "covered_duration":  float(info.get("covered_duration", 0.0)),
            "daily_alert_count": int(self._daily_alert_count),
        }

    # ── Drawing ───────────────────────────────────────────────────

    def draw(self, frame):
        if self._last_info is None:
            return frame
        if self.show_panel:
            frame = self._draw_logs(frame)
        return frame

    def _draw_status(self, frame, is_covered, info):
        if is_covered:
            if info.get('mode') == 'bright_blockage':
                status = "LENS COVERED (BRIGHT OBJECT)!"
            else:
                status = "LENS COVERED!"
            color = COLOR_ALERT
        else:
            status = "Lens Open"
            color = COLOR_OK

        # Background box
        cv2.rectangle(frame, (10, 10), (360, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (360, 150), color, 2)

        # Status
        cv2.putText(frame, status, (20, 40), self.font, 0.8, color, 2)

        # Mode
        mode_text = f"Mode: {info['mode'].upper()}"
        cv2.putText(frame, mode_text, (20, 65), self.font, 0.5, (255, 255, 255), 1)

        # Criteria
        criteria_text = f"Criteria: {info['criteria_met']}/{info['criteria_required']}"
        cv2.putText(frame, criteria_text, (20, 85), self.font, 0.5, (255, 255, 255), 1)

        # Brightness
        br_dict = info.get('brightness', {})
        if isinstance(br_dict, dict):
            br   = br_dict.get('brightness', 0.0)
            base = br_dict.get('baseline')
            if base:
                br_text = f"Brightness: {br:.1f} (Base: {base:.1f})"
            else:
                br_text = f"Brightness: {br:.1f}"
            cv2.putText(frame, br_text, (20, 105), self.font, 0.4, (255, 255, 255), 1)

        # Duration covered
        if is_covered and info.get('covered_duration', 0) > 0:
            duration = int(info['covered_duration'])
            dur_text = f"Covered: {duration}s"
            cv2.putText(frame, dur_text, (20, 125), self.font, 0.4, (0, 0, 255), 1)

        # Daily alert count
        alert_text = f"Today's alerts: {self._daily_alert_count}"
        cv2.putText(frame, alert_text, (20, 145), self.font, 0.4, (200, 200, 200), 1)

        return frame

    def _draw_logs(self, frame):
        visible = list(self._screen_logs)[-MAX_LOG_LINES:]
        if not visible:
            return frame
        h, _w  = frame.shape[:2]
        line_h = int(cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX,
                                     LOG_FONT_SCALE, LOG_THICKNESS)[0][1] * 2.2)
        panel_h = line_h * len(visible) + 8
        panel_y = h - LOG_Y_BOTTOM - panel_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (LOG_X-4, panel_y),
                      (LOG_X+520, h-LOG_Y_BOTTOM+4), LOG_BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i, entry in enumerate(visible):
            y   = panel_y + 6 + (i+1)*line_h
            txt = f"[{entry['time']}] {entry['msg']}"
            cv2.putText(frame, txt, (LOG_X, y),
                        cv2.FONT_HERSHEY_SIMPLEX, LOG_FONT_SCALE,
                        entry["color"], LOG_THICKNESS, cv2.LINE_AA)
        return frame