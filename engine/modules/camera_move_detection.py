# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:51:44 2026

@author: berka
"""

# -*- coding: utf-8 -*-
"""
camera_angle_guard.py
─────────────────────
Camera angle change detection (template matching + live RTSP).



Decision logic:
  - ROIs are placed at fixed, isolated points in the scene (permanent coordinates)
  - Each ROI is held as a template and searched for in the current frame
  - Consistent direction + sufficient ratio + triggered within 1 second window → ALARM

On alarm:
  - Snapshot is saved under ALARM_DIR
  - snapshot_path is returned in alarm_info from kamera_aci_kontrol();
    the dashboard broadcast layer uses this path.
"""

# ==================== IMPORTS ====================
import cv2
import numpy as np
import datetime
import os
import sys
import time as _time
from collections import deque


# ==================== SOURCE SETTINGS ====================
RTSP_URL = "rtsp://youreye:Y6ry3*-h63j2k@81.136.213.73:554/cam/playback?channel=12&subtype=0&starttime=2026_05_10_06_33_25&endtime=2026_05_11_22_10_00"

PIPELINE = (
    f'rtspsrc location="{RTSP_URL}" latency=100 protocols=tcp '
    '! rtph265depay ! h265parse ! nvh265dec ! videoconvert '
    '! video/x-raw,format=BGR '
    '! appsink drop=true sync=false max-buffers=2'
)




# ==================== IMAGE SETTINGS ====================
PROCESS_WIDTH  = 1280
PROCESS_HEIGHT = 720       # updated in main() according to original aspect ratio
TARGET_FPS     = 25


# ==================== ROI AND TEMPLATE SETTINGS ====================
ROI_SIZE      = 60         # ROI square size (px)
SEARCH_MARGIN = 40         # Template search area margin (px)


#                     --------------CAMERA ROIs---------------------

# Channel 1 ROI
FIXED_ROIS = [
    (114,628),
    (58,275),
    (83,29),
    (536,0),
    (1220,445),
    (1212,665),
      ]


# # Channel 2 ROI
# FIXED_ROIS = [
#     (142,655),
#     (65,520),
#     (24,390),
#     (0,260),
#     (0,148),
#     (1220,16),
#     (1214,649)   
#     ]

# # Channel 3 ROI
# FIXED_ROIS = [
#     (1217,655),
#     (1218,27),
#     (0,294),
#     (198,658),
#     (294,64),
#     (1216,657),
#     (30,478)
#     ]


# # Channel 4 ROI
# FIXED_ROIS = [
#     (1220, 9),
#     (616, 2),
#     (791, 23),
#     (239, 0),
#     (37, 2),
#     (34, 662),
#     (1216, 654),
#     ]


# # Channel 5 ROI
# FIXED_ROIS = [
#     (47,647),
#     (3,465),
#     (13,8),
#     (254,666),
#     (893,7),
#     (1216,657), 
#     ]


#     # Channel 6 ROI
# FIXED_ROIS = [
#         (1227,146),
#         (1015,18),
#         (634,0),
#         (277,43),
#         (3,119),
#         (57,276)
#     ]
    
#     # Channel 7 ROI
# FIXED_ROIS = [
#         (1226,86),
#         (780,0),
#         (75,649),
#         (13,387),
#         (83,31),
#         (1127,650),
#         (151,248)  
#     ]

# # Channel 8 ROI
# FIXED_ROIS = [
#     (61,606),
#     (971,646),
#     (1109,236),
#     (334,31),
#     (15,54),
#     (1062,33),
# ]

# # Channel 9 ROI
# FIXED_ROIS = [
#     (1140,613),
#     (902,656),
#     (894,88),
#     (35,645),
#     (146,98),
#     (1194,65),
# ]

# # Channel 10 ROI
# FIXED_ROIS = [
#     (1217,526),
#     (1040,645),
#     (633,656),
#     (34,655),
#     (18,57),
#     (1205,19),
#     (321,9),  
# ]

# # Channel 11 ROI
# FIXED_ROIS = [
#     (1209,642),
#     (1206,302),
#     (457,31),
#     (1084,9),
#     (9,189),
#     (207,649),
#     (744,0),  
# ]


# # Channel 13 ROI
# FIXED_ROIS = [
#     (1054,607),
#     (1160,397),
#     (1159,211),
#     (1073,100),
#     (630,0),
#     (0,435),
#     (450,0), 
# ]


# # Channel 14 ROI
# FIXED_ROIS = [
#     (14,44),
#     (341,584),
#     (12,554),
#     (257,15),
#     (285,206),
#     (305,359),
#     (1112,11), 
#     (1221,448)
# ]


# # Channel 15 ROI
# FIXED_ROIS = [
#     (866,596),
#     (946,399),
#     (989,224),
#     (1219,653),
#     (562,0),
#     (0,461),
# ]



# # Channel 16 ROI
# FIXED_ROIS = [
#     (48,654),
#     (985,664),
#     (1188,644),
#     (1138,479),
#     (1203,218),
#     (1198,13),
#     (0,2)
# ]





# ==================== SENSITIVITY SETTINGS ====================
MIN_SHIFT_PIXELS     = 12.0
MAX_SHIFT_STD        = 4.0
MIN_MATCH_CONFIDENCE = 0.6
MIN_VALID_ROIS       = 3


# ==================== DECISION SETTINGS ====================
TRIGGER_WINDOW_SEC  = 1.0
MIN_TRIGGERED_RATIO = 1.0
COOLDOWN_SEC        = 4.0
STUCK_RESET_SEC     = 1.0   # seconds before resetting if ROI consistently gives low confidence


# ==================== TEMPLATE UPDATE SETTINGS ====================
TEMPLATE_UPDATE_SEC  = 3.0
TEMPLATE_BLEND_ALPHA = 0.30


# ==================== ALARM RECORD SETTINGS ====================
ALARM_DIR = "camera_alarms"
os.makedirs(ALARM_DIR, exist_ok=True)


# ==================== PANEL SETTINGS ====================
PANEL_W        = 360
PANEL_MARGIN   = 15
PANEL_BG_COLOR = (0, 100, 200)
PANEL_ALPHA    = 0.82
TITLE_H        = 48
ROW_H          = 38
PADDING        = 16

COLOR_OK    = (0, 200, 80)
COLOR_ALARM = (0, 40, 220)
COLOR_LOST  = (130, 130, 130)


# ==================== SCREEN LOG SETTINGS ====================
MAX_LOG_LINES  = 8
LOG_FONT_SCALE = 0.52
LOG_THICKNESS  = 1
LOG_COLOR      = (220, 220, 220)
LOG_BG_COLOR   = (30, 30, 30)
LOG_X          = 10
LOG_Y_BOTTOM   = 40


# ==================== DAILY RESET SETTING ====================
RESET_HOUR   = 0
RESET_MINUTE = 0


# ==================== MODEL / GLOBAL STATE ====================

# Detector state — collected under one dict
detector_state = {
    "rois"                 : [],   # liste: {'rect': (x1,y1,x2,y2), 'template': float32}
    "roi_recent_trigger"   : {},   # idx → (timestamp, dx, dy)
    "roi_low_conf_since"   : {},   # idx → timestamp
    "last_template_update" : 0.0,
    "last_alarm_time"      : 0.0,
    "is_camera_moved"      : False,
    "last_decision_info"   : "",
    "last_shifts"          : [],   # her ROI icin (dx, dy, conf)
}

total_alarms      = 0
daily_alarm_count = 0
_last_reset_date  = None

screen_logs  = deque(maxlen=MAX_LOG_LINES * 3)
recording    = False
writer       = None
current_file = ""
frame_number = 0


# ==================== HELPER FUNCTIONS ====================

def add_log(msg, color=None):
    ts    = datetime.datetime.now().strftime("%H:%M:%S")
    color = color or LOG_COLOR
    screen_logs.append({"time": ts, "msg": msg, "color": color})
    print(f"[{ts}] {msg}")


def _check_daily_reset():
    """Automatic daily reset according to RESET_HOUR / RESET_MINUTE."""
    global _last_reset_date
    now    = datetime.datetime.now()
    today  = now.date()
    target = now.replace(hour=RESET_HOUR, minute=RESET_MINUTE,
                         second=0, microsecond=0)

    if _last_reset_date is None:
        _last_reset_date = today
        return

    if today != _last_reset_date and now >= target:
        reset_daily()


def reset_daily():
    """Reset daily alarm count."""
    global daily_alarm_count, _last_reset_date
    daily_alarm_count = 0
    _last_reset_date  = datetime.datetime.now().date()
    add_log(
        f"Daily reset -> {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')}",
        color=(255, 200, 0),
    )


def _pre_process(frame):
    """Grayscale + gaussian blur — suppresses encoding noise."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _next_fft_safe(n: int) -> int:
    """Returns smallest m >= n whose prime factors are only 2, 3, 5.
    cv2.matchTemplate uses FFT internally; result dimensions with other prime
    factors trigger an assertion failure in OpenCV's dxt.cpp."""
    if n <= 1:
        return max(1, n)
    while True:
        x = n
        for p in (2, 3, 5):
            while x % p == 0:
                x //= p
        if x == 1:
            return n
        n += 1


def _pad_search_for_fft(search: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
    """Pads search so (search_size - tmpl_size + 1) is FFT-safe on each axis."""
    sh, sw = search.shape[:2]
    th, tw = tmpl.shape[:2]
    rh = sh - th + 1
    rw = sw - tw + 1
    safe_rh = _next_fft_safe(rh)
    safe_rw = _next_fft_safe(rw)
    pad_h = safe_rh - rh
    pad_w = safe_rw - rw
    if pad_h > 0 or pad_w > 0:
        search = np.pad(search, ((0, pad_h), (0, pad_w)), mode="edge")
    return search


def _setup_fixed_rois(first_frame):
    """Build the permanent ROI list from FIXED_ROIS coordinates."""
    processed = _pre_process(first_frame)
    detector_state["rois"].clear()

    for cx, cy in FIXED_ROIS:
        x1 = max(0, cx - ROI_SIZE // 2)
        y1 = max(0, cy - ROI_SIZE // 2)
        x2 = min(PROCESS_WIDTH,  x1 + ROI_SIZE)
        y2 = min(PROCESS_HEIGHT, y1 + ROI_SIZE)
        template = processed[y1:y2, x1:x2].astype(np.float32)
        detector_state["rois"].append({
            "rect"    : (x1, y1, x2, y2),
            "template": template,
        })

    detector_state["last_template_update"] = _time.time()
    # add_log(f"{len(detector_state['rois'])} adet ROI yuklendi", color=(0, 255, 120))


def _blend_templates(processed):
    """Update templates with a soft blend (for persistent scene changes)."""
    for roi in detector_state["rois"]:
        x1, y1, x2, y2 = roi["rect"]
        new_patch = processed[y1:y2, x1:x2].astype(np.float32)
        if roi["template"].shape != new_patch.shape:
            roi["template"] = new_patch
            continue
        roi["template"] = cv2.addWeighted(
            roi["template"], 1.0 - TEMPLATE_BLEND_ALPHA,
            new_patch, TEMPLATE_BLEND_ALPHA, 0
        )
    detector_state["last_template_update"] = _time.time()


def _match_roi(processed, roi):
    """Template matching for a single ROI → (dx, dy, confidence)."""
    x1, y1, x2, y2 = roi["rect"]
    tmpl_u8 = np.clip(roi["template"], 0, 255).astype(np.uint8)

    sx1 = max(0, x1 - SEARCH_MARGIN)
    sy1 = max(0, y1 - SEARCH_MARGIN)
    sx2 = min(PROCESS_WIDTH,  x2 + SEARCH_MARGIN)
    sy2 = min(PROCESS_HEIGHT, y2 + SEARCH_MARGIN)
    search_img = processed[sy1:sy2, sx1:sx2]

    if (tmpl_u8.shape[0] > search_img.shape[0] or
        tmpl_u8.shape[1] > search_img.shape[1]):
        return 0.0, 0.0, 0.0

    search_img = _pad_search_for_fft(search_img, tmpl_u8)
    res = cv2.matchTemplate(search_img, tmpl_u8, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    found_x = sx1 + max_loc[0]
    found_y = sy1 + max_loc[1]
    dx = float(found_x - x1)
    dy = float(found_y - y1)
    return dx, dy, float(max_val)


# def _save_alarm_snapshot(frame, info):
#     """
#     Save the alarm snapshot image and metadata.
#     Returns: saved file path (used by the dashboard broadcast layer).
#     """
#     ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename  = f"alarm_{ts}_{info['alarm_id']:04d}.jpg"
#     full_path = os.path.join(ALARM_DIR, filename)

#     snap = frame.copy()
#     cv2.putText(snap, "CAMERA ANGLE CHANGED",
#                 (50, PROCESS_HEIGHT - 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
#     cv2.putText(snap,
#                 f"{ts}  yon=({info['mean_dx']:+.1f},{info['mean_dy']:+.1f})  "
#                 f"oran={info['triggered_ratio']:.0%}",
#                 (50, PROCESS_HEIGHT - 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
#     cv2.imwrite(full_path, snap)
#     return full_path


def draw_screen_logs(frame):
    visible = list(screen_logs)[-MAX_LOG_LINES:]
    if not visible:
        return frame
    h, _w   = frame.shape[:2]
    line_h  = int(cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX,
                                  LOG_FONT_SCALE, LOG_THICKNESS)[0][1] * 2.2)
    panel_h = line_h * len(visible) + 8
    panel_y = h - LOG_Y_BOTTOM - panel_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (LOG_X - 4, panel_y),
                  (LOG_X + 520, h - LOG_Y_BOTTOM + 4), LOG_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, entry in enumerate(visible):
        y   = panel_y + 6 + (i + 1) * line_h
        txt = f"[{entry['time']}] {entry['msg']}"
        cv2.putText(frame, txt, (LOG_X, y),
                    cv2.FONT_HERSHEY_SIMPLEX, LOG_FONT_SCALE,
                    entry["color"], LOG_THICKNESS, cv2.LINE_AA)
    return frame


def draw_status_panel(frame):
    h, w = frame.shape[:2]

    n_rows  = 3
    panel_h = TITLE_H + n_rows * ROW_H + PADDING

    px = w - PANEL_W - PANEL_MARGIN
    py = h - panel_h - PANEL_MARGIN

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + PANEL_W, py + panel_h),
                  PANEL_BG_COLOR, -1)
    cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1 - PANEL_ALPHA, 0, frame)
    cv2.rectangle(frame, (px, py), (px + PANEL_W, py + panel_h),
                  (0, 140, 255), 2)

    # Title
    cv2.rectangle(frame, (px, py), (px + PANEL_W, py + TITLE_H),
                  (0, 70, 160), -1)
    cv2.putText(frame, "Camera Angle Change",
                (px + PADDING, py + TITLE_H - 14),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    y = py + TITLE_H + ROW_H - 10

    # Status
    if detector_state["is_camera_moved"]:
        status_txt = "Status: MOVED"
        col        = (0, 0, 255)
    else:
        status_txt = "Status: STABLE"
        col        = (0, 255, 120)
    cv2.putText(frame, status_txt, (px + PADDING, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, col, 1, cv2.LINE_AA)

    y += ROW_H
    cv2.putText(frame, f"Total Alarms: {total_alarms}",
                (px + PADDING, y), cv2.FONT_HERSHEY_DUPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_rois(frame):
        """Draws ROI boxes and their states, scaled to current frame size."""
        rois = detector_state["rois"]
        shifts = detector_state["last_shifts"]
        
        # Anlık frame boyutuna göre ölçekleme faktörü
        fh, fw = frame.shape[:2]
        sx = fw / PROCESS_WIDTH
        sy = fh / PROCESS_HEIGHT

        for i, roi in enumerate(rois):
            x1, y1, x2, y2 = roi["rect"]
            
            # Koordinatları ölçekle
            rx1, ry1 = int(x1 * sx), int(y1 * sy)
            rx2, ry2 = int(x2 * sx), int(y2 * sy)

            if i < len(shifts):
                _dx, _dy, conf = shifts[i]
            else:
                conf = 0.0

            is_valid = conf >= MIN_MATCH_CONFIDENCE
            in_window = i in detector_state["roi_recent_trigger"]

            if not is_valid:
                color = COLOR_LOST
            elif in_window:
                color = COLOR_ALARM
            else:
                color = COLOR_OK

            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
            # Etiketi de ölçeklenmiş konuma yaz
            cv2.putText(frame, f"R{i+1}", (rx1 + 3, ry1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if detector_state["is_camera_moved"]:
            cv2.rectangle(frame, (0, 0), (fw, fh), (0, 0, 255), 8)

        return frame

def start_recording(fps, width, height):
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recordings/kayit_{ts}.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    w        = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    add_log(f"Recording started → {filename}", color=(0, 80, 255))
    return w, filename


# ==================== MAIN FUNCTION ====================

def kamera_aci_kontrol(img):
    """
    Takes a single raw frame and performs camera angle check.

    Returns:
        annotated_frame  : Frame with drawings applied
        alarm_info       : dict — alarm info for this frame
                           {'triggered': bool, 'snapshot_path': str|None,
                            'triggered_ratio': float, 'mean_dx': float,
                            'mean_dy': float, 'alarm_id': int|None}
        stats            : dict — total and daily alarm counts, status
                           {'total_alarms': int, 'daily_alarms': int,
                            'active_rois': int, 'is_motion': bool}
    """
    global frame_number, total_alarms, daily_alarm_count

    _check_daily_reset()
    frame_number += 1
    now       = _time.time()
    annotated = img.copy()
    processed = _pre_process(img)

    alarm_info = {
        "triggered"      : False,
        "snapshot_path"  : None,
        "triggered_ratio": 0.0,
        "mean_dx"        : 0.0,
        "mean_dy"        : 0.0,
        "alarm_id"       : None,
    }

    rois = detector_state["rois"]
    if not rois:
        annotated = draw_status_panel(annotated)
        annotated = draw_screen_logs(annotated)
        return annotated, alarm_info, {
            "total_alarms": total_alarms,
            "daily_alarms": daily_alarm_count,
            "active_rois" : 0,
            "is_motion"   : False,
        }

    # Periodic template update
    if (now - detector_state["last_template_update"]) >= TEMPLATE_UPDATE_SEC:
        _blend_templates(processed)

    # Template matching + individual triggering for each ROI
    shifts = []
    for i, roi in enumerate(rois):
        dx, dy, conf = _match_roi(processed, roi)
        shifts.append((dx, dy, conf))

        # Track low confidence — reset that ROI's template if it persists too long
        if conf < MIN_MATCH_CONFIDENCE:
            if i not in detector_state["roi_low_conf_since"]:
                detector_state["roi_low_conf_since"][i] = now
            elif (now - detector_state["roi_low_conf_since"][i]) >= STUCK_RESET_SEC:
                x1, y1, x2, y2 = roi["rect"]
                roi["template"] = processed[y1:y2, x1:x2].astype(np.float32)
                detector_state["roi_low_conf_since"].pop(i, None)
            continue
        else:
            detector_state["roi_low_conf_since"].pop(i, None)

        magnitude = float(np.sqrt(dx * dx + dy * dy))
        if magnitude >= MIN_SHIFT_PIXELS:
            detector_state["roi_recent_trigger"][i] = (now, dx, dy)

    detector_state["last_shifts"] = shifts

    # Remove out-of-window triggers — ROI automatically turns green
    expired = [
        i for i, (t, _, _) in detector_state["roi_recent_trigger"].items()
        if (now - t) > TRIGGER_WINDOW_SEC
    ]
    for i in expired:
        detector_state["roi_recent_trigger"].pop(i, None)

    # Active triggers
    active = [
        (i, dx, dy)
        for i, (t, dx, dy) in detector_state["roi_recent_trigger"].items()
        if i < len(rois)
    ]
    active_count    = len(active)
    total_count     = len(rois)
    triggered_ratio = active_count / total_count if total_count else 0.0

    # Direction consistency
    if active_count >= 2:
        dxs = np.array([t[1] for t in active])
        dys = np.array([t[2] for t in active])
        shift_std = float(np.sqrt(np.var(dxs) + np.var(dys)))
        mean_dx   = float(np.mean(dxs))
        mean_dy   = float(np.mean(dys))
    else:
        shift_std = 0.0
        mean_dx   = 0.0
        mean_dy   = 0.0

    # Decision
    is_motion = (
        total_count >= MIN_VALID_ROIS         and
        triggered_ratio >= MIN_TRIGGERED_RATIO and
        # shift_std <= MAX_SHIFT_STD             and
        active_count >= 2
    )
    detector_state["is_camera_moved"] = is_motion
    detector_state["last_decision_info"] = (
        f"P:{active_count}/{total_count} ({triggered_ratio:.0%}) "
        f"y=({mean_dx:+.1f},{mean_dy:+.1f}) std={shift_std:.1f}"
    )

    # Alarm
    if is_motion and (now - detector_state["last_alarm_time"]) >= COOLDOWN_SEC:
        detector_state["last_alarm_time"] = now
        total_alarms      += 1
        daily_alarm_count += 1

        # Use annotated frame for snapshot
        annotated_for_snap = draw_rois(annotated.copy())
        annotated_for_snap = draw_status_panel(annotated_for_snap)

        snap_meta = {
            "alarm_id"       : total_alarms,
            "triggered_ratio": triggered_ratio,
            "mean_dx"        : mean_dx,
            "mean_dy"        : mean_dy,
        }
        snap_path = _save_alarm_snapshot(annotated_for_snap, snap_meta)

        alarm_info.update({
            "triggered"      : True,
            "snapshot_path"  : snap_path,
            "triggered_ratio": triggered_ratio,
            "mean_dx"        : mean_dx,
            "mean_dy"        : mean_dy,
            "alarm_id"       : total_alarms,
        })

        add_log(
            f"ALARM #{total_alarms}: camera movement detected → {snap_path}",
            color=(0, 0, 255),
        )

        # Reset templates after alarm → new position becomes new baseline
        for roi in rois:
            x1, y1, x2, y2 = roi["rect"]
            roi["template"] = processed[y1:y2, x1:x2].astype(np.float32)
        detector_state["roi_recent_trigger"].clear()
        detector_state["last_template_update"] = now

    # Drawings
    annotated = draw_rois(annotated)
    annotated = draw_status_panel(annotated)
    annotated = draw_screen_logs(annotated)

    stats = {
        "total_alarms": total_alarms,
        "daily_alarms": daily_alarm_count,
        "active_rois" : len(rois),
        "is_motion"   : is_motion,
    }
    return annotated, alarm_info, stats


# ==================== MAIN PROGRAM ====================

def main():
    global recording, writer, current_file, PROCESS_HEIGHT

    cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)


    if not cap.isOpened():
        print("[ERROR] Could not open camera connection!")
        sys.exit(1)

    _width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    _height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _fps    = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS

    # PROCESS_HEIGHT'i orijinal aspect ratio'ya gore ayarla
    PROCESS_HEIGHT = int(_height * (PROCESS_WIDTH / _width))

    print(f"[START] Stream: {_width}x{_height} @ {_fps}fps")
    print(f"        Processing size: {PROCESS_WIDTH}x{PROCESS_HEIGHT}")
    print("Keys:  R=Record  |  Z=Daily reset  |  Q=Quit")

    os.makedirs("recordings", exist_ok=True)

    # Read first frame and set up fixed ROIs
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture first frame!")
        cap.release()
        sys.exit(1)
    first_resized = cv2.resize(first_frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    _setup_fixed_rois(first_resized)

    fps_counter = 0
    fps_time    = _time.time()
    fps_display = 0

    add_log("System started", color=(100, 255, 100))

    cv2.namedWindow("Camera Angle Control", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Angle Control", 1280, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                add_log("Frame capture failed!", color=(0, 0, 255))
                break

            current = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
            annotated, alarm_info, stats = kamera_aci_kontrol(current)

            # FPS
            fps_counter += 1
            if _time.time() - fps_time >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_time    = _time.time()
            cv2.putText(annotated, f"FPS: {fps_display}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            # REC indicator
            if recording:
                cv2.circle(annotated, (30, 60), 10, (0, 0, 255), -1)
                cv2.putText(annotated, "REC", (48, 68),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            # Record — write after all drawings
            if recording and writer is not None:
                writer.write(annotated)

            cv2.imshow("Camera Angle Control", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('r'), ord('R')):
                if not recording:
                    writer, current_file = start_recording(
                        _fps, PROCESS_WIDTH, PROCESS_HEIGHT
                    )
                    recording = True
                else:
                    writer.release()
                    writer    = None
                    recording = False
                    add_log(f"Recording stopped → {current_file}", color=(0, 80, 255))
            elif key in (ord('z'), ord('Z')):
                reset_daily()
            elif key in (ord('q'), ord('Q')):
                break

    finally:
        if recording and writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 50)
        print("           RESULT REPORT")
        print("=" * 50)
        print(f"  Total frames  : {frame_number}")
        print(f"  Total alarms  : {total_alarms}")
        print(f"  Today alarms  : {daily_alarm_count}")
        print("=" * 50)


if __name__ == "__main__":
    main()


# =============================================================================
#  MODULE SYSTEM INTEGRATION
#  CameraMoveDetector — instance-based, used by main.py
# =============================================================================

class CameraMoveDetector:
    """
    Camera angle change detection — separate instance per camera.
    Used in main.py with the same pattern as stream_alarm.

    Usage:
        detector = CameraMoveDetector(rois=[[114,628], [58,275], ...])
        # In main loop:
        info = detector.process(frame)
        if info["triggered"]:
            ...send alarm...
    """

    def __init__(self, rois: list,
                 process_width: int        = 1280,
                 process_height: int       = 720,
                 roi_size: int             = ROI_SIZE,
                 search_margin: int        = SEARCH_MARGIN,
                 min_shift_pixels: float   = MIN_SHIFT_PIXELS,
                 max_shift_std: float      = MAX_SHIFT_STD,
                 min_match_confidence: float = MIN_MATCH_CONFIDENCE,
                 min_valid_rois: int       = MIN_VALID_ROIS,
                 trigger_window_sec: float = TRIGGER_WINDOW_SEC,
                 min_triggered_ratio: float = MIN_TRIGGERED_RATIO,
                 cooldown_sec: float       = COOLDOWN_SEC,
                 template_update_sec: float = TEMPLATE_UPDATE_SEC,
                 template_blend_alpha: float = TEMPLATE_BLEND_ALPHA,
                 stuck_reset_sec: float    = STUCK_RESET_SEC,
                 show_panel: bool          = True):

        self._fixed_rois           = [tuple(r) for r in rois]
        self._pw                   = process_width
        self._ph                   = process_height
        self._roi_size             = roi_size
        self._search_margin        = search_margin
        self._min_shift            = min_shift_pixels
        self._max_std              = max_shift_std
        self._min_conf             = min_match_confidence
        self._min_valid            = min_valid_rois
        self._trigger_window       = trigger_window_sec
        self._min_ratio            = min_triggered_ratio
        self._cooldown             = cooldown_sec
        self._tpl_update_sec       = template_update_sec
        self._tpl_blend            = template_blend_alpha
        self._stuck_reset          = stuck_reset_sec
        self._show_panel           = show_panel

        self._state = {
            "rois"                 : [],
            "roi_recent_trigger"   : {},
            "roi_low_conf_since"   : {},
            "last_template_update" : 0.0,
            "last_alarm_time"      : 0.0,
            "is_camera_moved"      : False,
            "last_shifts"          : [],
            "last_triggered_ratio" : 0.0,
        }
        self._initialized = False
        self._total_alarms = 0

    # ── Set up ROI templates on first frame ──────────────────────

    def _setup(self, frame):
        processed = _pre_process(frame)
        self._state["rois"].clear()
        for cx, cy in self._fixed_rois:
            x1 = max(0, cx - self._roi_size // 2)
            y1 = max(0, cy - self._roi_size // 2)
            x2 = min(self._pw, x1 + self._roi_size)
            y2 = min(self._ph, y1 + self._roi_size)
            self._state["rois"].append({
                "rect"    : (x1, y1, x2, y2),
                "template": processed[y1:y2, x1:x2].astype(np.float32),
            })
        self._state["last_template_update"] = _time.time()
        self._initialized = True

    # ── Main processing ───────────────────────────────────────────

    def process(self, frame) -> dict:
        """
        Processes the frame.
        Returns: {"triggered": bool, "triggered_ratio": float,
                  "mean_dx": float, "mean_dy": float}
        """
        resized = cv2.resize(frame, (self._pw, self._ph))

        if not self._initialized:
            self._setup(resized)
            return {"triggered": False, "triggered_ratio": 0.0,
                    "mean_dx": 0.0, "mean_dy": 0.0}

        processed = _pre_process(resized)
        now       = _time.time()
        state     = self._state

        # Periyodik template blend
        if (now - state["last_template_update"]) >= self._tpl_update_sec:
            for roi in state["rois"]:
                x1, y1, x2, y2 = roi["rect"]
                new_patch = processed[y1:y2, x1:x2].astype(np.float32)
                if roi["template"].shape == new_patch.shape:
                    roi["template"] = cv2.addWeighted(
                        roi["template"], 1.0 - self._tpl_blend,
                        new_patch, self._tpl_blend, 0,
                    )
                else:
                    roi["template"] = new_patch
            state["last_template_update"] = now

        # Template matching for each ROI
        shifts = []
        for i, roi in enumerate(state["rois"]):
            x1, y1, x2, y2 = roi["rect"]
            tmpl_u8 = np.clip(roi["template"], 0, 255).astype(np.uint8)
            sx1 = max(0, x1 - self._search_margin)
            sy1 = max(0, y1 - self._search_margin)
            sx2 = min(self._pw, x2 + self._search_margin)
            sy2 = min(self._ph, y2 + self._search_margin)
            search = processed[sy1:sy2, sx1:sx2]

            if tmpl_u8.shape[0] > search.shape[0] or tmpl_u8.shape[1] > search.shape[1]:
                shifts.append((0.0, 0.0, 0.0))
                continue

            search = _pad_search_for_fft(search, tmpl_u8)
            res = cv2.matchTemplate(search, tmpl_u8, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            dx = float(sx1 + max_loc[0] - x1)
            dy = float(sy1 + max_loc[1] - y1)
            conf = float(max_val)
            shifts.append((dx, dy, conf))

            if conf < self._min_conf:
                if i not in state["roi_low_conf_since"]:
                    state["roi_low_conf_since"][i] = now
                elif (now - state["roi_low_conf_since"][i]) >= self._stuck_reset:
                    roi["template"] = processed[y1:y2, x1:x2].astype(np.float32)
                    state["roi_low_conf_since"].pop(i, None)
                continue
            else:
                state["roi_low_conf_since"].pop(i, None)

            if float(np.sqrt(dx*dx + dy*dy)) >= self._min_shift:
                state["roi_recent_trigger"][i] = (now, dx, dy)

        state["last_shifts"] = shifts

        # Clean up expired triggers
        expired = [i for i, (t, _, _) in state["roi_recent_trigger"].items()
                   if (now - t) > self._trigger_window]
        for i in expired:
            state["roi_recent_trigger"].pop(i, None)

        # Decision
        active = [(i, dx, dy) for i, (t, dx, dy) in state["roi_recent_trigger"].items()
                  if i < len(state["rois"])]
        total_count     = len(state["rois"])
        triggered_ratio = len(active) / total_count if total_count else 0.0

        mean_dx = mean_dy = shift_std = 0.0
        if len(active) >= 2:
            dxs = np.array([t[1] for t in active])
            dys = np.array([t[2] for t in active])
            shift_std = float(np.sqrt(np.var(dxs) + np.var(dys)))
            mean_dx   = float(np.mean(dxs))
            mean_dy   = float(np.mean(dys))

        is_moved = (
            total_count        >= self._min_valid  and
            triggered_ratio    >= self._min_ratio  and
            #shift_std          <= self._max_std    and
            len(active)        >= 2
        )
        state["is_camera_moved"]      = is_moved
        state["last_triggered_ratio"] = triggered_ratio

        triggered = False
        if is_moved and (now - state["last_alarm_time"]) >= self._cooldown:
            state["last_alarm_time"] = now
            self._total_alarms      += 1
            triggered                = True
            # Reset templates after alarm — new position becomes new baseline
            for roi in state["rois"]:
                x1, y1, x2, y2 = roi["rect"]
                roi["template"] = processed[y1:y2, x1:x2].astype(np.float32)
            state["roi_recent_trigger"].clear()
            state["last_template_update"] = now

        return {
            "triggered"      : triggered,
            "triggered_ratio": triggered_ratio,
            "mean_dx"        : mean_dx,
            "mean_dy"        : mean_dy,
        }

    # ── Drawing ───────────────────────────────────────────────────

    def draw(self, frame):
        """
        Draws ROI boxes and optional status panel onto the frame.
        Coordinates are scaled from processing resolution to actual frame size.
        Returns the annotated frame.
        """
        state = self._state
        rois  = state["rois"]
        if not rois:
            return frame

        fh, fw = frame.shape[:2]
        sx = fw / self._pw
        sy = fh / self._ph
        shifts = state["last_shifts"]

        for i, roi in enumerate(rois):
            x1, y1, x2, y2 = roi["rect"]
            rx1, ry1 = int(x1 * sx), int(y1 * sy)
            rx2, ry2 = int(x2 * sx), int(y2 * sy)

            conf      = shifts[i][2] if i < len(shifts) else 0.0
            is_valid  = conf >= self._min_conf
            in_window = i in state["roi_recent_trigger"]

            if not is_valid:
                color = (130, 130, 130)  # gray — low confidence
            elif in_window:
                color = (0, 40, 220)     # blue — triggered
            else:
                color = (0, 200, 80)     # green — OK

            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
            cv2.putText(frame, f"R{i+1}", (rx1 + 3, ry1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if state["is_camera_moved"]:
            cv2.rectangle(frame, (0, 0), (fw, fh), (0, 0, 255), 8)

        if self._show_panel:
            frame = self._draw_status_panel(frame)

        return frame

    def _draw_status_panel(self, frame):
        h, w      = frame.shape[:2]
        panel_w   = 290
        panel_h   = 82
        margin    = 12
        px        = w - panel_w - margin
        py        = h - panel_h - margin

        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (80, 80, 80), 1)

        is_moved   = self._state["is_camera_moved"]
        status_txt = "MOVED" if is_moved else "STABLE"
        status_col = (0, 0, 255) if is_moved else (0, 200, 80)
        ratio      = self._state["last_triggered_ratio"]

        cv2.putText(frame, f"Cam Move: {status_txt}",
                    (px + 8, py + 24), cv2.FONT_HERSHEY_DUPLEX,
                    0.58, status_col, 1, cv2.LINE_AA)
        cv2.putText(frame, f"ROI: {len(self._state['rois'])}   Alarms: {self._total_alarms}",
                    (px + 8, py + 50), cv2.FONT_HERSHEY_DUPLEX,
                    0.48, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Triggered: {ratio:.0%}",
                    (px + 8, py + 72), cv2.FONT_HERSHEY_DUPLEX,
                    0.44, (160, 160, 160), 1, cv2.LINE_AA)
        return frame