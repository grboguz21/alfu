"""
Kasa Takip Modülü
-----------------
Müşteri bekleme alanını ve kasiyer kasalarını izler.
Kasiyersiz bekleme ve ek kasa açılması gerektiğinde alarm üretir.

Config example:
    {
        "type":                   "kasa_takip",
        "name":                   "kasa_takip_cam1",
        "orta_coords":            [[0,890],[346,878],[516,74],[1322,44],[1542,1426],[4,1422],[10,890]],
        "kasa1_coords":           [[1520,456],[2110,394],[2026,8],[1448,0],[1520,458]],
        "kasa2_coords":           [[6,144],[420,272],[226,608],[4,594],[10,144]],
        "entry_wait_time":        3.0,
        "id_memory_duration":     4.0,
        "table_seen_threshold":   1.2,
        "cashier_empty_wait":     4.0,
        "cashier_alarm_delay":    15.0,
        "alarm_min_customers":    3,
        "extra_lane_threshold":   2,
        "extra_lane_confirm":     2.0,
        "customer_conf":          0.50,
        "cashier_conf":           0.60,
        "show_panel":             true
    }

get_data() output:
    {
        "Customer Count":         int,
        "Lane 1 Occupied":        bool,
        "Lane 2 Occupied":        bool,
        "Active Cashiers":        int,
        "no_cashier_alert":       bool,
        "extra_lane_alert":       bool
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

# Görsel sabitler
COLOR_CUSTOMER_AREA   = (0, 255, 255)   # sarı — müşteri alanı çerçevesi
COLOR_LANE            = (0, 255, 0)     # yeşil — kasa çerçevesi
COLOR_DOT_CONFIRMED   = (0, 0, 255)     # kırmızı — onaylı müşteri noktası
COLOR_DOT_PENDING     = (0, 255, 255)   # sarı — henüz onaylanmamış
COLOR_DOT_CASHIER     = (0, 255, 0)     # yeşil — kasiyer noktası
COLOR_PANEL_BG        = (0, 165, 255)   # turuncu — durum paneli
COLOR_ALARM_NO_CASH   = (0, 0, 200)     # kırmızı — kasiyersiz alarm
COLOR_ALARM_EXTRA     = (0, 80, 210)    # mavi-kırmızı — ek kasa alarmı


# ==================== MODULE ====================

class KasaTakipModule(BaseModule):

    def __init__(self,
                 name:                  str,
                 orta_coords:           list,
                 kasa1_coords:          list,
                 kasa2_coords:          list,
                 model_path:            str   = "models/yolo26l.pt",
                 entry_wait_time:       float = 3.0,
                 id_memory_duration:    float = 4.0,
                 table_seen_threshold:  float = 1.2,
                 cashier_empty_wait:    float = 4.0,
                 cashier_alarm_delay:   float = 15.0,
                 alarm_min_customers:   int   = 3,
                 extra_lane_threshold:  int   = 2,
                 extra_lane_confirm:    float = 2.0,
                 customer_conf:         float = 0.50,
                 cashier_conf:          float = 0.60,
                 show_panel:            bool  = True,
                 **_kwargs):

        self.name                 = name
        self.show_panel           = show_panel

        # Koordinatlar — orijinal kamera uzayında saklanır, resize yapılmaz
        self._orta_coords  = np.array(orta_coords,  np.int32)
        self._kasa1_coords = np.array(kasa1_coords, np.int32)
        self._kasa2_coords = np.array(kasa2_coords, np.int32)

        # Parametreler
        self._entry_wait_time      = entry_wait_time
        self._id_memory_duration   = id_memory_duration
        self._table_seen_threshold = table_seen_threshold
        self._cashier_empty_wait   = cashier_empty_wait
        self._cashier_alarm_delay  = cashier_alarm_delay
        self._alarm_min_customers  = alarm_min_customers
        self._extra_lane_threshold = extra_lane_threshold
        self._extra_lane_confirm   = extra_lane_confirm
        self._customer_conf        = customer_conf
        self._cashier_conf         = cashier_conf

        # YOLO modeli — Tür B
        self._model = YOLO(model_path)
        if torch.cuda.is_available():
            self._model.to('cuda')

        # İç durum
        # track_id → {first_seen, last_seen, last_red_time, center, confirmed}
        self._customer_track:   dict  = {}
        self._kasa1_last_seen:  float = 0.0
        self._kasa2_last_seen:  float = 0.0
        self._extra_lane_since        = None
        self._last_save_time:   float = 0.0
        self._last_reset_date         = None

        # draw() için anlık durum — update() çağrılmadan None
        self._last_status = None

        self._load_state()
        print(f"✅ KasaTakipModule ready [{name}]")

    # ==================== PERSISTENCE ====================

    def _state_path(self) -> str:
        return os.path.join(STATE_DIR, f"kasa_takip_{self.name}.json")

    def _save_state(self):
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            state = {
                "date":             (self._last_reset_date.isoformat()
                                     if self._last_reset_date else None),
                "kasa1_last_seen":  self._kasa1_last_seen,
                "kasa2_last_seen":  self._kasa2_last_seen,
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
            self._kasa1_last_seen = float(state.get("kasa1_last_seen", 0.0))
            self._kasa2_last_seen = float(state.get("kasa2_last_seen", 0.0))
            self._last_reset_date = datetime.date.fromisoformat(saved_date)
            print(f"[{self.name}] State loaded.")
        except Exception as e:
            print(f"[{self.name}] State load error: {e} — starting fresh.")

    # ==================== HELPERS ====================

    def _check_daily_reset(self):
        today = datetime.datetime.now().date()
        if self._last_reset_date is None:
            self._last_reset_date = today
            return
        if today != self._last_reset_date:
            self._customer_track   = {}
            self._kasa1_last_seen  = 0.0
            self._kasa2_last_seen  = 0.0
            self._extra_lane_since = None
            self._last_reset_date  = today
            self._save_state()
            print(f"[{self.name}] Daily reset → {today}")

    def _point_in_poly(self, coords, point) -> bool:
        return cv2.pointPolygonTest(coords, point, False) >= 0

    # ==================== UPDATE ====================

    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        # bboxes/object_ids pipeline'dan gelse de yok sayılır — Tür B
        self._check_daily_reset()
        now = _time.time()

        # --- YOLO inference (GPU_LOCK zorunlu) ---
        with GPU_LOCK, torch.no_grad():
            results = self._model.track(
                frame, classes=[0], persist=True, verbose=False,
                device=0 if torch.cuda.is_available() else 'cpu'
            )

        # Lock dışında işle
        current_kasa1_count = 0
        current_kasa2_count = 0
        this_frame_center   = set()

        det_boxes     = []
        det_track_ids = []
        det_confs     = []

        if results[0].boxes.id is not None:
            det_boxes     = results[0].boxes.xyxy.cpu().numpy()
            det_track_ids = results[0].boxes.id.int().cpu().numpy()
            det_confs     = results[0].boxes.conf.cpu().numpy()

        # Kasiyer nokta bilgisini draw() için sakla
        cashier_dots = []   # [(cx, cy), ...]

        for box, track_id, conf in zip(det_boxes, det_track_ids, det_confs):
            x1, y1, x2, y2 = box
            # Koordinatlar orijinal kamera uzayında — resize yok
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            if self._point_in_poly(self._orta_coords, center):
                if conf >= self._customer_conf:
                    this_frame_center.add(int(track_id))
                    tid = int(track_id)
                    if tid not in self._customer_track:
                        self._customer_track[tid] = {
                            'first_seen'   : now,
                            'last_seen'    : now,
                            'last_red_time': None,
                            'center'       : center,
                            'confirmed'    : False,
                        }
                    else:
                        self._customer_track[tid]['last_seen'] = now
                        self._customer_track[tid]['center']    = center

            elif self._point_in_poly(self._kasa1_coords, center):
                if conf >= self._cashier_conf and current_kasa1_count == 0:
                    current_kasa1_count = 1
                    cashier_dots.append(center)

            elif self._point_in_poly(self._kasa2_coords, center):
                if conf >= self._cashier_conf and current_kasa2_count == 0:
                    current_kasa2_count = 1
                    cashier_dots.append(center)

        # --- Müşteri takip & sayım ---
        to_delete         = []
        current_red_count = 0
        customer_dots     = []   # [(cx, cy, confirmed), ...]

        for tid, data in self._customer_track.items():
            if now - data['last_seen'] > self._id_memory_duration:
                to_delete.append(tid)
                continue

            visible = tid in this_frame_center

            if (now - data['first_seen']) >= self._entry_wait_time:
                data['confirmed'] = True

            if data['confirmed'] and visible:
                data['last_red_time'] = now

            count_in_table = (
                data['confirmed']
                and data['last_red_time'] is not None
                and (now - data['last_red_time']) <= self._table_seen_threshold
            )
            if count_in_table:
                current_red_count += 1

            if visible:
                customer_dots.append((data['center'], data['confirmed']))

        for tid in to_delete:
            del self._customer_track[tid]

        # --- Kasiyer durumu ---
        if current_kasa1_count > 0:
            self._kasa1_last_seen = now
        if current_kasa2_count > 0:
            self._kasa2_last_seen = now

        kasa1_occupied  = (now - self._kasa1_last_seen) <= self._cashier_empty_wait
        kasa2_occupied  = (now - self._kasa2_last_seen) <= self._cashier_empty_wait
        active_cashiers = sum([kasa1_occupied, kasa2_occupied])
        last_cashier    = max(self._kasa1_last_seen, self._kasa2_last_seen)

        # --- Alarm mantığı ---
        cashier_timer_expired = (
            last_cashier > 0
            and (now - last_cashier) >= (self._cashier_empty_wait + self._cashier_alarm_delay)
        )
        alarm_no_cashier = (
            current_red_count >= self._alarm_min_customers
            and active_cashiers == 0
            and cashier_timer_expired
        )

        if current_red_count > self._extra_lane_threshold and active_cashiers == 1:
            if self._extra_lane_since is None:
                self._extra_lane_since = now
        else:
            self._extra_lane_since = None

        alarm_extra_lane = (
            self._extra_lane_since is not None
            and (now - self._extra_lane_since) >= self._extra_lane_confirm
        )

        # --- Geri sayım hesapla (draw için) ---
        remaining1 = None
        if not kasa1_occupied and self._kasa1_last_seen > 0:
            elapsed1   = max(0.0, now - self._kasa1_last_seen - self._cashier_empty_wait)
            r1         = self._cashier_alarm_delay - elapsed1
            if 0 < r1 <= self._cashier_alarm_delay:
                remaining1 = r1

        remaining2 = None
        if not kasa2_occupied and self._kasa2_last_seen > 0:
            elapsed2   = max(0.0, now - self._kasa2_last_seen - self._cashier_empty_wait)
            r2         = self._cashier_alarm_delay - elapsed2
            if 0 < r2 <= self._cashier_alarm_delay:
                remaining2 = r2

        # draw() için durum
        self._last_status = {
            "current_red_count": current_red_count,
            "kasa1_occupied":    kasa1_occupied,
            "kasa2_occupied":    kasa2_occupied,
            "active_cashiers":   active_cashiers,
            "alarm_no_cashier":  alarm_no_cashier,
            "alarm_extra_lane":  alarm_extra_lane,
            "customer_dots":     customer_dots,
            "cashier_dots":      cashier_dots,
            "remaining1":        remaining1,
            "remaining2":        remaining2,
        }

        if now - self._last_save_time >= SAVE_INTERVAL_SEC:
            self._save_state()
            self._last_save_time = now

    # ==================== DATA ====================

    def get_data(self) -> dict:
        if self._last_status is None:
            return {
                "Customer Count":   0,
                "Lane 1 Occupied":  False,
                "Lane 2 Occupied":  False,
                "Active Cashiers":  0,
                "no_cashier_alert": False,
                "extra_lane_alert": False,
            }
        s = self._last_status
        return {
            "Customer Count":   s["current_red_count"],
            "Lane 1 Occupied":  s["kasa1_occupied"],
            "Lane 2 Occupied":  s["kasa2_occupied"],
            "Active Cashiers":  s["active_cashiers"],
            "no_cashier_alert": s["alarm_no_cashier"],
            "extra_lane_alert": s["alarm_extra_lane"],
        }

    # ==================== DRAW ====================

    def draw(self, frame):
        if self._last_status is None:
            return frame

        s = self._last_status

        # Alan çerçeveleri — orijinal koordinatlar, resize yok
        cv2.polylines(frame, [self._orta_coords],  True, COLOR_CUSTOMER_AREA, 4)
        cv2.polylines(frame, [self._kasa1_coords], True, COLOR_LANE,          4)
        cv2.polylines(frame, [self._kasa2_coords], True, COLOR_LANE,          4)

        # Alan etiketleri
        self._draw_area_label(frame, self._orta_coords,  "CUSTOMER AREA", COLOR_CUSTOMER_AREA,
                              custom_point=(610, 80))
        self._draw_area_label(frame, self._kasa1_coords, "LANE_1",        COLOR_LANE)
        self._draw_area_label(frame, self._kasa2_coords, "LANE_2",        COLOR_LANE)

        # Müşteri noktaları
        for (center, confirmed) in s["customer_dots"]:
            color = COLOR_DOT_CONFIRMED if confirmed else COLOR_DOT_PENDING
            cv2.circle(frame, center, 12, color,       -1)
            cv2.circle(frame, center, 12, (0, 0, 0),    2)

        # Kasiyer noktaları
        for center in s["cashier_dots"]:
            cv2.circle(frame, center, 12, COLOR_DOT_CASHIER, -1)
            cv2.circle(frame, center, 12, (0, 0, 0),          2)

        # Geri sayım baloncukları
        if s["remaining1"] is not None:
            self._draw_countdown(frame, self._kasa1_coords,
                                 s["remaining1"], self._cashier_alarm_delay)
        if s["remaining2"] is not None:
            self._draw_countdown(frame, self._kasa2_coords,
                                 s["remaining2"], self._cashier_alarm_delay)

        if self.show_panel:
            frame = self._draw_panel(frame, s)

        frame = self._draw_alarm_banners(frame, s)
        return frame

    # ==================== DRAW HELPERS ====================

    def _draw_area_label(self, frame, coords, text, bg, custom_point=None):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, kl = 0.8, 2
        (mw, mh), bl = cv2.getTextSize(text, font, fs, kl)
        if custom_point:
            ex, ey = custom_point
        else:
            x, y, _, _ = cv2.boundingRect(coords)
            ey = y + 40 if y < 40 else y + 20
            ex = x + 20
        cv2.rectangle(frame, (ex - 5, ey - mh - 10), (ex + mw + 5, ey + bl), bg, -1)
        cv2.putText(frame, text, (ex, ey - 5), font, fs, (0, 0, 0), kl)

    def _draw_countdown(self, frame, coords, seconds_left, total):
        x, y, bw, bh = cv2.boundingRect(coords)
        cx = x + bw + 60
        cy = y + bh // 2

        cv2.circle(frame, (cx, cy), 40, (30, 30, 30), -1)
        cv2.circle(frame, (cx, cy), 40, (0, 0, 0),    2)

        ratio   = max(0.0, min(1.0, seconds_left / total))
        end_ang = int(360 * ratio)

        if ratio > 0.5:
            arc_color = (0, 220, 0)
        elif ratio > 0.2:
            arc_color = (0, 200, 255)
        else:
            arc_color = (0, 0, 255)

        for angle in range(0, end_ang, 2):
            a1 = angle - 90
            cv2.ellipse(frame, (cx, cy), (35, 35), 0, a1, a1 + 2, arc_color, 4)

        secs = int(np.ceil(seconds_left))
        txt  = str(max(secs, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs   = 1.0 if secs >= 10 else 1.2
        (tw, th), _ = cv2.getTextSize(txt, font, fs, 2)
        cv2.putText(frame, txt, (cx - tw // 2, cy + th // 2), font, fs, (255, 255, 255), 2)

    def _draw_panel(self, frame, s):
        h, w  = frame.shape[:2]
        rx1, ry1, rx2, ry2 = w - 600, 20, w - 20, 200
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), COLOR_PANEL_BG, -1)
        cv2.putText(frame, f"Customer Area: {s['current_red_count']}",
                    (rx1 + 20, ry1 + 50),  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, f"Lane 1: {'Occupied' if s['kasa1_occupied'] else 'Empty'}",
                    (rx1 + 20, ry1 + 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, f"Lane 2: {'Occupied' if s['kasa2_occupied'] else 'Empty'}",
                    (rx1 + 20, ry1 + 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        return frame

    def _draw_alarm_banners(self, frame, s):
        h, w      = frame.shape[:2]
        banner_h  = 70
        n_banners = int(s["alarm_no_cashier"]) + int(s["alarm_extra_lane"])
        if n_banners == 0:
            return frame

        banner_top = h - banner_h * n_banners
        idx        = 0

        if s["alarm_no_cashier"]:
            self._draw_single_banner(frame, banner_top + idx * banner_h,
                                     "!! ALARM: CUSTOMERS WAITING - NO CASHIER !!",
                                     COLOR_ALARM_NO_CASH)
            idx += 1

        if s["alarm_extra_lane"]:
            self._draw_single_banner(frame, banner_top + idx * banner_h,
                                     "!! ALARM: OPEN AN ADDITIONAL LANE !!",
                                     COLOR_ALARM_EXTRA)
        return frame

    def _draw_single_banner(self, frame, y, msg, bg_color):
        h, w     = frame.shape[:2]
        banner_h = 70
        cv2.rectangle(frame, (0, y), (w, y + banner_h), bg_color, -1)
        font  = cv2.FONT_HERSHEY_SIMPLEX
        fs, thick = 1.6, 4
        (mw, mh), _ = cv2.getTextSize(msg, font, fs, thick)
        cv2.putText(frame, msg, ((w - mw) // 2, y + banner_h - 18),
                    font, fs, (255, 255, 255), thick)

    # ==================== SHUTDOWN ====================

    def shutdown(self):
        self._save_state()
        print(f"[{self.name}] Shutdown — state saved.")

    def reset(self):
        self._customer_track   = {}
        self._kasa1_last_seen  = 0.0
        self._kasa2_last_seen  = 0.0
        self._extra_lane_since = None
        self._last_status      = None
