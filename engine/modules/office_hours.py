"""
Work Hours
----------
Tracks the daily total working time of people in defined desk / area zones.
Uses a YOLO pose model + optional background subtractor fallback
(useful for steep top-down cameras where YOLO may miss people).

Only seated workers are counted — standing or walking people are filtered out
by checking whether lower-body keypoints (knees, ankles) are visible from the
camera. From a top-down office camera those joints are hidden under the desk
when seated.

Two area detection modes (per area):
  is_special: false  →  At least one critical keypoint must be inside the area.
  is_special: true   →  Bbox centre OR foot OR any keypoint inside the area +
                         lock mechanism (best for top-down / partial-occlusion
                         cameras).

Config example:
    {
        "type":                "work_hours",
        "name":                "office_cam1",
        "model_path":          "models/yolo26l-pose.pt",
        "occlusion_tolerance": 5.0,
        "use_bg_subtractor":   true,
        "min_person_area":     3000,
        "areas": [
            {
                "name":       "Desk 1",
                "points":     [[x,y], [x,y], ...],
                "threshold":  1.5,
                "is_special": true
            }
        ],
        "show_panel": true
    }

get_data() output:
    {
        "Desk 1":              120.5,
        "Desk 2":               45.2,
        "daily_total_minutes": 165.7
    }
"""

import os
import json
import time as _time
import datetime
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from .base import BaseModule
except ImportError:
    class BaseModule:   # minimal standalone stub — replaced by the real class in prod
        pass

try:
    from engine.shared_memory import GPU_LOCK
except ImportError:
    import threading
    GPU_LOCK = threading.Lock()


# ==================== CONFIG ====================

STATE_DIR         = "state"
SAVE_INTERVAL_SEC = 30

# ── Detection ─────────────────────────────────────────────────────────────────
MIN_CONF        = 0.15
CRITICAL_KP_IDX = [0, 1, 2, 3, 4, 5, 6, 9, 10]   # head, shoulders, wrists

# ── Special-zone stabilisation ────────────────────────────────────────────────
SPECIAL_LOCK_TTL = 8.0   # how long (s) a confirmed worker is carried through a YOLO
#                           miss once the detection streak has been established
SPECIAL_LOSS_CAP = 8.0   # upper bound (s) on the loss-tolerance window for is_special

# ── Ghost / flash detection filter ───────────────────────────────────────────
# Screen flashes or bright reflections create ghost "person" bboxes that have
# almost no valid keypoints. A real seated worker always shows at least
# MIN_VISIBLE_KP upper-body keypoints from a top-down camera.
UPPER_BODY_KP_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # nose → wrists (11 kpts)
MIN_VISIBLE_KP    = 3     # minimum upper-body keypoints that must be visible;
#                           raised to 3 — the 7 s suppress+settle window ensures
#                           the camera is fully recovered before normal detection
#                           resumes, so real workers always show 3+ keypoints
#                           (nose + both shoulders at minimum). Requiring 3 also
#                           blocks ghosts produced in the 1-2 frames before the
#                           rolling-average flash threshold triggers.
MIN_KP_CONF       = 0.25  # keypoint confidence threshold to count as "visible"

# ── Frame-edge filter ─────────────────────────────────────────────────────────
FRAME_EDGE_MARGIN = 0   # pixels — disabled (0): Desk 2 worker's head clips the top
#                          edge; enabling this filter was dropping them before zone
#                          assignment. The is_special streak threshold already
#                          prevents edge-walkers from triggering false greens.

# ── Seated-posture filter ─────────────────────────────────────────────────────
# From a top-down view a seated person's knees and ankles are hidden under the
# desk (low keypoint confidence). A standing visitor has them clearly visible.
LOWER_BODY_KP_IDX = [13, 14, 15, 16]   # left/right knee, left/right ankle
SEATED_CONF       = 0.82               # confidence above which a lower-body kpt is
#                                        "visible". Seated workers register 0.55-0.78
#                                        (pass ✓). Standing visitors register 0.80-0.90
#                                        (filtered ✓ at 0.82).

# ── Flash / brightness-spike filter ──────────────────────────────────────────
# A monitor flash or camera exposure spike causes a sudden scene-wide brightness
# jump that makes YOLO detect ghost persons everywhere. We measure mean frame
# brightness per tick against a rolling average. Exceeding FLASH_DELTA_THRESH
# triggers two consecutive protection windows:
#
#   Suppression (FLASH_SUPPRESS_SEC):  YOLO inference skipped entirely.
#   Settle      (FLASH_SETTLE_SEC):    YOLO runs but BG fallback is disabled and
#                                       only already-locked zones may stay active.
#
# Combined window: 4 + 3 = 7 s of total flash protection.
FLASH_ROLLING_N    = 30    # rolling-average window length (frames)
FLASH_DELTA_THRESH = 18    # deviation from rolling avg that signals a flash;
#                            grey flashes register 10-20, white 25+;
#                            18 catches both without false-triggering on normal
#                            lighting drift (typically < 8 units)
FLASH_SUPPRESS_SEC = 4.0   # YOLO suppression duration; white flashes take 3-4 s for
#                            the camera AGC to stabilise
FLASH_SETTLE_SEC   = 3.0   # post-suppression settle duration; BG subtractor model
#                            was trained on bright flash frames and labels the entire
#                            scene as foreground for several seconds after the flash

# ── Background subtractor ─────────────────────────────────────────────────────
BG_HISTORY       = 500
BG_VAR_THRESHOLD = 80

# ── Visual ────────────────────────────────────────────────────────────────────
PANEL_FG_COLOR = (220, 220, 220)


# ==================== MODULE ====================

class WorkHoursModule(BaseModule):
    """
    Area-based working-time tracker — Type B module.
    Ignores the main pipeline's bboxes; runs its own YOLO pose model every frame.
    """

    def __init__(self, name: str,
                 model_path: str,
                 areas: list,
                 occlusion_tolerance: float = 5.0,
                 use_bg_subtractor: bool = True,
                 min_person_area: int = 3000,
                 show_panel: bool = True,
                 **_kwargs):
        self.name                = name
        self.occlusion_tolerance = occlusion_tolerance
        self.show_panel          = show_panel
        self.min_person_area     = min_person_area

        # YOLO pose model
        print(f"[{name}] Loading YOLO pose model: {model_path}")
        self._model = YOLO(model_path)
        print(f"[{name}] Model ready")

        # Background subtractor (MOG2 fallback for top-down views)
        if use_bg_subtractor:
            self._bg_sub         = cv2.createBackgroundSubtractorMOG2(
                history=BG_HISTORY,
                varThreshold=BG_VAR_THRESHOLD,
                detectShadows=False,
            )
            self._morph_kernel   = np.ones((5, 5), np.uint8)
            self._bg_warmup_left = BG_HISTORY   # frames until model is reliable
        else:
            self._bg_sub         = None
            self._morph_kernel   = None
            self._bg_warmup_left = 0

        # Area states
        self._area_states = self._build_area_states(areas)

        # Daily tracking
        self._last_reset_date = None
        self._last_frame_time = None
        self._screen_logs     = deque(maxlen=30)

        # Flash suppression
        self._brightness_history  = deque(maxlen=FLASH_ROLLING_N)
        self._flash_suppress_until = 0.0   # suppress YOLO until this time
        self._flash_settle_until   = 0.0   # block new zone activations until this time

        # draw() guard — None until first update() call
        self._last_status = None

        # Persistence
        self._last_save_time = 0.0
        self._load_state()

        print(f"✅ WorkHoursModule ready [{name}]")
        print(f"   ├── Area count     : {len(areas)}")
        print(f"   ├── Occlusion tol  : {occlusion_tolerance}s")
        print(f"   ├── BG subtractor  : {'yes' if use_bg_subtractor else 'no'}")
        for s in self._area_states.values():
            tag = "SPECIAL" if s["is_special"] else "normal"
            print(f"   ├── {s['name']:12s} thr={s['threshold']}s  [{tag}]")

    # ==================== PERSISTENCE ====================

    def _state_path(self) -> str:
        return os.path.join(STATE_DIR, f"work_hours_{self.name}.json")

    def _save_state(self):
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            state = {
                "date": (self._last_reset_date.isoformat()
                         if self._last_reset_date else None),
                "areas": {
                    s["name"]: s["total_seconds"]
                    for s in self._area_states.values()
                },
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
            area_data = state.get("areas", {})
            for s in self._area_states.values():
                if s["name"] in area_data:
                    s["total_seconds"] = float(area_data[s["name"]])
            self._last_reset_date = datetime.date.fromisoformat(saved_date)
            total = sum(s["total_seconds"] for s in self._area_states.values())
            print(f"[{self.name}] State loaded — total {total / 60:.1f} min")
        except Exception as e:
            print(f"[{self.name}] State load error: {e} — starting fresh.")

    # ==================== HELPERS ====================

    def _check_daily_reset(self):
        today = datetime.datetime.now().date()
        if self._last_reset_date is None:
            self._last_reset_date = today
            return
        if today != self._last_reset_date:
            for s in self._area_states.values():
                s["total_seconds"]        = 0.0
                s["first_seen_in_streak"] = None
                s["first_seen_time"]      = None
                s["is_locked"]            = False
            self._last_reset_date = today
            self._add_log(
                f"Daily reset → {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')}",
                color=(255, 200, 0),
            )
            self._save_state()

    def _add_log(self, msg: str, color=None):
        ts    = datetime.datetime.now().strftime("%H:%M:%S")
        color = color or PANEL_FG_COLOR
        self._screen_logs.append({"time": ts, "msg": msg, "color": color})
        print(f"[{ts}][{self.name}] {msg}")

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        h, rem = divmod(int(seconds), 3600)
        m, s   = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _build_area_states(self, areas: list) -> dict:
        # Push last_seen far into the past so zones start RED at launch.
        # Using current time would put every zone inside the loss-tolerance window,
        # making them appear yellow (or green) before any worker is detected.
        distant_past = _time.time() - (SPECIAL_LOSS_CAP + 10.0)
        states = {}
        for i, area in enumerate(areas):
            states[i] = {
                "name":                   area["name"],
                "polygon":                np.array(area["points"], np.int32).reshape((-1, 1, 2)),
                "threshold":              float(area.get("threshold", 3.0)),
                "is_special":             bool(area.get("is_special", False)),
                "total_seconds":          0.0,
                "last_seen":              distant_past,
                # is_special streak tracking
                "first_seen_in_streak":   None,
                "last_tentative_seen":    distant_past,
                # normal-area first-seen tracking
                "first_seen_time":        None,
                # is_special lock
                "is_locked":              False,
                "last_valid_pose_time":   0.0,
                # drawing
                "_color":                 (0, 0, 255),
            }
        return states

    # ==================== BACKGROUND SUBTRACTOR ====================

    def _compute_fg_mask(self, frame):
        if self._bg_sub is None:
            return None
        raw  = self._bg_sub.apply(frame)
        mask = cv2.morphologyEx(raw, cv2.MORPH_OPEN, self._morph_kernel)
        return cv2.dilate(mask, self._morph_kernel, iterations=1)

    def _fg_person_in_zone(self, fg, polygon) -> bool:
        zone_mask  = np.zeros(fg.shape[:2], dtype=np.uint8)
        cv2.fillPoly(zone_mask, [polygon.reshape(-1, 2)], 255)
        fg_in_zone = cv2.bitwise_and(fg, zone_mask)
        contours, _ = cv2.findContours(fg_in_zone, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        return any(cv2.contourArea(c) >= self.min_person_area for c in contours)

    # ==================== DETECTION ====================

    def _is_real_person(self, p_idx: int, keypoints_conf) -> bool:
        if keypoints_conf is None:
            return True
        visible = sum(
            1 for kp in UPPER_BODY_KP_IDX
            if keypoints_conf[p_idx][kp] > MIN_KP_CONF
        )
        return visible >= MIN_VISIBLE_KP

    def _is_seated(self, p_idx: int, keypoints_conf) -> bool:
        if keypoints_conf is None:
            return True
        return not any(
            keypoints_conf[p_idx][kp] > SEATED_CONF
            for kp in LOWER_BODY_KP_IDX
        )

    def _detect_in_areas(self, results, fg) -> dict:
        """Determine person presence per area this frame → {area_idx: bool}."""
        tentative         = {i: False for i in range(len(self._area_states))}
        yolo_found_anyone = len(results[0].boxes) > 0

        if yolo_found_anyone:
            fh, fw         = results[0].orig_shape
            boxes          = results[0].boxes.xyxy.cpu().numpy()
            has_keypoints  = results[0].keypoints is not None
            keypoints_xy   = results[0].keypoints.xy.cpu().numpy()   if has_keypoints else None
            keypoints_conf = results[0].keypoints.conf.cpu().numpy() if has_keypoints else None

            for p_idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                fy = int(y2)

                # Skip bbox clipped by the frame edge.
                if FRAME_EDGE_MARGIN > 0 and (
                        x1 <= FRAME_EDGE_MARGIN or y1 <= FRAME_EDGE_MARGIN or
                        x2 >= fw - FRAME_EDGE_MARGIN or y2 >= fh - FRAME_EDGE_MARGIN):
                    continue

                # Reject ghost detections.
                if not self._is_real_person(p_idx, keypoints_conf if has_keypoints else None):
                    continue

                # Skip standing / walking people.
                if not self._is_seated(p_idx, keypoints_conf if has_keypoints else None):
                    continue

                # Primary assignment: bbox centre owns the zone.
                # Prevents reaching arms / bbox spill triggering adjacent zones.
                primary = next(
                    (i for i, s in self._area_states.items()
                     if cv2.pointPolygonTest(s["polygon"], (float(cx), float(cy)), False) >= 0),
                    None,
                )
                if primary is not None:
                    tentative[primary] = True
                    continue

                # Centre not in any zone: fall back to foot / keypoint logic.
                for i, state in self._area_states.items():
                    if tentative[i]:
                        continue
                    poly = state["polygon"]

                    if state["is_special"]:
                        in_poly = cv2.pointPolygonTest(poly, (cx, fy), False) >= 0
                        if not in_poly and has_keypoints:
                            for kp_idx in range(17):
                                if keypoints_conf[p_idx][kp_idx] > MIN_CONF:
                                    kx = int(keypoints_xy[p_idx][kp_idx][0])
                                    ky = int(keypoints_xy[p_idx][kp_idx][1])
                                    if cv2.pointPolygonTest(poly, (kx, ky), False) >= 0:
                                        in_poly = True
                                        break
                        if in_poly:
                            tentative[i] = True
                    else:
                        if has_keypoints:
                            for kp_idx in CRITICAL_KP_IDX:
                                if keypoints_conf[p_idx][kp_idx] > MIN_CONF:
                                    kx = float(keypoints_xy[p_idx][kp_idx][0])
                                    ky = float(keypoints_xy[p_idx][kp_idx][1])
                                    if cv2.pointPolygonTest(poly, (kx, ky), False) >= 0:
                                        tentative[i] = True
                                        break
                        else:
                            if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                                tentative[i] = True

        # BG subtractor fallback — only when YOLO missed everyone.
        # Running it while YOLO already found people causes shadow / artifact
        # false positives in empty zones.
        if fg is not None and not yolo_found_anyone:
            for i, state in self._area_states.items():
                if not tentative[i]:
                    tentative[i] = self._fg_person_in_zone(fg, state["polygon"])

        return tentative

    def _apply_special_stabilization(self, tentative: dict, current_time: float) -> dict:
        """Lock mechanism for is_special areas."""
        for i, state in self._area_states.items():
            if not state["is_special"]:
                continue
            if tentative[i]:
                state["last_valid_pose_time"] = current_time
                # Only commit the lock once a real detection streak has been
                # established (streak >= threshold). A single ghost frame has
                # streak ≈ 0 and must NOT lock the zone.
                streak = (current_time - state["first_seen_in_streak"]
                          if state["first_seen_in_streak"] is not None else 0.0)
                if streak >= state["threshold"]:
                    state["is_locked"] = True
            elif (state["is_locked"] and
                  current_time - state["last_valid_pose_time"] < SPECIAL_LOCK_TTL):
                tentative[i] = True
            else:
                state["is_locked"] = False
        return tentative

    def _update_timers(self, tentative: dict, delta: float, current_time: float):
        """Update duration counters and color indicators for each area."""
        for i, state in self._area_states.items():
            loss_tol = (min(SPECIAL_LOSS_CAP, self.occlusion_tolerance * 1.5)
                        if state["is_special"] else self.occlusion_tolerance)

            if state["is_special"]:
                if tentative[i]:
                    state["last_tentative_seen"] = current_time
                    if state["first_seen_in_streak"] is None:
                        state["first_seen_in_streak"] = current_time
                elif (current_time - state["last_tentative_seen"]) > loss_tol:
                    # Preserve the streak during the full loss-tolerance window so
                    # that re-detections after brief YOLO misses don't need to
                    # rebuild from scratch to go green again.
                    state["first_seen_in_streak"] = None

                streak = (current_time - state["first_seen_in_streak"]
                          if state["first_seen_in_streak"] else 0.0)
                # Require BOTH an established streak AND current tentative presence.
                # Without this, a ghost streak timer grows to threshold autonomously
                # after the ghost is gone, triggering false greens.
                is_active = streak >= state["threshold"] and tentative[i]

                if is_active:
                    state["total_seconds"] += delta
                    state["last_seen"]      = current_time
                    state["_color"]         = (0, 255, 0)       # green
                elif (current_time - state["last_seen"]) <= loss_tol:
                    state["total_seconds"] += delta
                    state["_color"]         = (0, 255, 255)     # yellow
                else:
                    state["_color"]         = (0, 0, 255)       # red

            else:
                if tentative[i]:
                    if state["first_seen_time"] is None:
                        state["first_seen_time"] = current_time
                    state["last_seen"] = current_time

                    duration = current_time - state["first_seen_time"]
                    if duration >= state["threshold"]:
                        state["total_seconds"] += delta
                        state["_color"]         = (0, 255, 0)   # green
                    else:
                        state["_color"]         = (0, 255, 255) # yellow

                elif (state["first_seen_time"] is not None and
                      (current_time - state["last_seen"]) < loss_tol):
                    state["_color"] = (255, 255, 255)            # white
                    if (current_time - state["first_seen_time"]) >= state["threshold"]:
                        state["total_seconds"] += delta
                else:
                    state["_color"]          = (0, 0, 255)       # red
                    state["first_seen_time"] = None

    # ==================== UPDATE ====================

    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        # bboxes / tracking data not used — Type B module uses its own YOLO model
        self._check_daily_reset()
        now = _time.time()

        delta = (now - self._last_frame_time) if self._last_frame_time else 0.0
        delta = min(delta, 1.0)
        self._last_frame_time = now

        # Feed every frame into the BG subtractor to build its model, but suppress
        # its output during warmup (early frames flag the whole scene as foreground).
        raw_fg = self._compute_fg_mask(frame)
        if self._bg_warmup_left > 0:
            self._bg_warmup_left -= 1
            fg = None
        else:
            fg = raw_fg

        # ── Flash / brightness-spike detection ───────────────────────────────────
        # Compare current frame brightness against a rolling average. An absolute
        # deviation above FLASH_DELTA_THRESH signals a flash. Both gray flashes
        # (moderate increase) and white flashes (large increase) are caught.
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        if len(self._brightness_history) >= 5:
            rolling_avg = sum(self._brightness_history) / len(self._brightness_history)
            deviation   = abs(brightness - rolling_avg)
            if deviation > FLASH_DELTA_THRESH:
                self._flash_suppress_until = now + FLASH_SUPPRESS_SEC
                self._flash_settle_until   = now + FLASH_SUPPRESS_SEC + FLASH_SETTLE_SEC
                for _s in self._area_states.values():
                    if not _s.get("is_special"):
                        continue
                    recently_active = (
                        _s.get("last_seen", 0) > 0 and
                        now - _s["last_seen"] < SPECIAL_LOSS_CAP
                    )
                    if _s.get("is_locked") or recently_active:
                        # Re-commit the lock and restart its TTL from the flash moment.
                        # Covers: (a) active lock → prevent mid-suppression expiry;
                        #         (b) lock just expired but zone was still counting →
                        #             force-lock so the worker isn't dropped.
                        _s["is_locked"]            = True
                        _s["last_valid_pose_time"] = now
                        # Keep the confirmed worker's streak clock alive.
                        if _s.get("first_seen_in_streak") is not None:
                            _s["last_tentative_seen"] = now
                    else:
                        # No confirmed worker — clear any partial streak immediately.
                        # YOLO runs on the 1-2 brightening frames before the rolling-
                        # average threshold triggers; those ghost detections may have
                        # started a first_seen_in_streak. Without this clear, the ghost
                        # streak survives the full 7 s window (loss-tolerance 7.5 s >
                        # 7 s window) and instantly locks the zone when YOLO resumes.
                        _s["first_seen_in_streak"] = None
                self._add_log(
                    f"Flash detected (brightness={brightness:.1f}, "
                    f"avg={rolling_avg:.1f}, Δ={deviation:.1f}) "
                    f"— suppressing {FLASH_SUPPRESS_SEC}s + settle {FLASH_SETTLE_SEC}s",
                    color=(0, 200, 255),
                )
        self._brightness_history.append(brightness)

        # ── Suppression window: skip YOLO entirely ────────────────────────────────
        if now < self._flash_suppress_until:
            tentative = {i: False for i in range(len(self._area_states))}
            # Refresh lock TTL and streak clock every suppression frame so that the
            # full SPECIAL_LOCK_TTL window is available from the moment YOLO resumes,
            # not from when the flash was first detected. Without this, 4 s of
            # suppression consumes half the 8 s TTL, leaving only 4 s for YOLO to
            # re-acquire a desk that is harder to detect post-flash.
            for _s in self._area_states.values():
                if not _s.get("is_special"):
                    continue
                if _s.get("is_locked"):
                    _s["last_valid_pose_time"] = now
                    if _s.get("first_seen_in_streak") is not None:
                        _s["last_tentative_seen"] = now

        # ── Normal / settle window: run YOLO ─────────────────────────────────────
        else:
            with GPU_LOCK, torch.no_grad():
                results = self._model(frame, classes=[0], conf=MIN_CONF, verbose=False)

            # During the post-flash settle window suppress BG fallback. The BG
            # subtractor model was trained on bright flash frames; once brightness
            # returns to normal it labels the entire scene as foreground.
            in_settle = now < self._flash_settle_until
            tentative = self._detect_in_areas(results, None if in_settle else fg)

            # During settle, block new zone activations from post-flash ghosts.
            # Only zones locked before the flash (confirmed real workers) stay active.
            if in_settle:
                for i, state in self._area_states.items():
                    if tentative[i] and not state.get("is_locked"):
                        tentative[i] = False

        tentative = self._apply_special_stabilization(tentative, now)
        self._update_timers(tentative, delta, now)

        # Update draw() guard
        self._last_status = True

        if now - self._last_save_time >= SAVE_INTERVAL_SEC:
            self._save_state()
            self._last_save_time = now

    # ==================== DATA ====================

    def get_data(self) -> dict:
        data  = {}
        total = 0.0
        for state in self._area_states.values():
            mins                = round(state["total_seconds"] / 60, 1)
            data[state["name"]] = mins
            total              += state["total_seconds"]
        data["daily_total_minutes"] = round(total / 60, 1)
        return data

    # ==================== DRAW ====================

    def draw(self, frame):
        if self._last_status is None:   # update() not yet called
            return frame
        frame = self._draw_areas(frame)
        if self.show_panel:
            frame = self._draw_panel(frame)
        return frame

    def _draw_areas(self, frame):
        for state in self._area_states.values():
            color    = state.get("_color", (0, 0, 255))
            time_str = self._fmt_time(state["total_seconds"])
            cv2.polylines(frame, [state["polygon"]], True, color, 2)
            label = f"{state['name']}: {time_str}"
            lx    = int(state["polygon"][0][0][0])
            ly    = max(int(state["polygon"][0][0][1]) - 10, 20)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (lx + 2, ly - th - 4), (lx + tw + 8, ly + 4),
                          (0, 0, 0), -1)
            cv2.putText(frame, label, (lx + 5, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return frame

    def _draw_panel(self, frame):
        h, w  = frame.shape[:2]
        n     = len(self._area_states)
        box_h = 80 + n * 35
        px    = w - 310
        py    = 20

        cv2.rectangle(frame, (px, py), (w - 20, py + box_h), (0, 140, 255), -1)
        cv2.putText(frame, "WORK HOURS",
                    (px + 10, py + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        for i, state in self._area_states.items():
            time_str = self._fmt_time(state["total_seconds"])
            color    = state.get("_color", (255, 255, 255))
            cv2.putText(frame, f"{state['name']}: {time_str}",
                        (px + 10, py + 65 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        return frame

    # ==================== SHUTDOWN ====================

    def shutdown(self):
        self._save_state()

    def reset(self):
        for s in self._area_states.values():
            s["total_seconds"]        = 0.0
            s["first_seen_in_streak"] = None
            s["first_seen_time"]      = None
            s["is_locked"]            = False
            s["_color"]               = (0, 0, 255)
        self._save_state()


# ==================== STANDALONE TEST ====================

if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path

    load_dotenv(Path(__file__).parent.parent / ".env")

    video_source = os.getenv("office", "")
    if not video_source:
        raise SystemExit("Set the 'office' variable in your .env file")

    video = cv2.VideoCapture(video_source)
    if not video.isOpened():
        raise SystemExit(f"Cannot open video source: {video_source}")

    W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Stream resolution: {W}x{H}")

    # Relative zone definitions (0.0–1.0) — converted to absolute pixels below.
    # Fine-tune the x/y values if boxes drift off the chairs.
    ZONES_REL = [
        # Desk 1 — left desk work chair (upper-left; excludes the visitor chair)
        {"name": "Desk 1",
         "points": [(0.14, 0.04), (0.50, 0.04), (0.50, 0.46), (0.14, 0.46)],
         "threshold": 1.5, "is_special": True},
        # Desk 2 — centre desk (shifted up so bbox centre near frame top is inside)
        {"name": "Desk 2",
         "points": [(0.53, 0.00), (0.76, 0.00), (0.76, 0.32), (0.53, 0.32)],
         "threshold": 1.5, "is_special": True},
        # Desk 3 — right desk (tight zone; y_max=0.32 excludes standing visitors)
        {"name": "Desk 3",
         "points": [(0.82, 0.17), (0.93, 0.17), (0.93, 0.37), (0.82, 0.37)],
         "threshold": 1.5, "is_special": True},
    ]

    areas = [
        {**z, "points": [[int(x * W), int(y * H)] for x, y in z["points"]]}
        for z in ZONES_REL
    ]

    module = WorkHoursModule(
        name="test",
        model_path="yolo26l-pose.pt",
        areas=areas,
        occlusion_tolerance=5.0,
        use_bg_subtractor=True,
        min_person_area=3000,
        show_panel=True,
    )

    cv2.namedWindow("Work Hours Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Work Hours Test", W, H)

    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            print("End of stream")
            break

        module.update(None, None, None, None, frame, {})
        annotated = module.draw(frame.copy())

        cv2.imshow("Work Hours Test", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    module.shutdown()
    video.release()
    cv2.destroyAllWindows()

    print(f"\nProcessed {frame_count} frames")
    print("Final totals:", module.get_data())
