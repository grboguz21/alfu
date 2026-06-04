"""
Kasiyer Süresi Modülü
---------------------
Tanımlı polygon ROI'ler içinde kişi tespiti yaparak kasiyerlerin
aktif çalışma sürelerini saniye cinsinden biriktirir.
Ana pipeline'ın (YOLO + tracker) bboxes çıktısını kullanır (Tür A).

Config example:
    {
        "type":        "kasiyer_suresi",
        "name":        "kasiyer_suresi_cam1",
        "kasalar": [
            {
                "id":     "KASA-01",
                "coords": [[1520,456],[2110,394],[2026,8],[1448,0],[1520,458]]
            },
            {
                "id":     "KASA-02",
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
        "Kasa 1 Active Minutes": float,
        "Kasa 1 Is Active":      bool,
        "Kasa 2 Active Minutes": float,
        "Kasa 2 Is Active":      bool
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

# Varsayılan görüntü boyutları
DEFAULT_ORIGINAL_W = 2560
DEFAULT_ORIGINAL_H = 1440
DEFAULT_DISPLAY_W  = 1280

# Renkler (BGR)
COLOR_PALETTE = [
    (235, 206, 135),   # Sarımsı - Kasa 1
    (255, 105, 180),   # Pembe   - Kasa 2
    (0,   255, 165),   # Yeşil   - Kasa 3
    (255, 165,   0),   # Turuncu - Kasa 4
]
COLOR_PERSON_OUT  = (150, 240, 0)    # Fosforlu Yeşil - alan dışı kişi
COLOR_WHITE       = (255, 255, 255)
COLOR_DARK        = (15,  15,  15)
COLOR_DARK_BORDER = (40,  40,  40)
COLOR_PANEL_BG    = (15, 108, 242)   # Panel arka plan (turuncu-mavi)


# ==================== MODULE ====================

class KasiyerSuresiModule(BaseModule):
    """
    Tür A — Pipeline sonuçlarını kullanan modül.
    Ana YOLO modelinin tespit ettiği kişi bbox'larını kullanır,
    her kasa için ayrı Shapely polygon kontrolü yapar.
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

        # Ölçek hesabı
        self._orig_w   = original_w
        self._orig_h   = original_h
        self._disp_w   = display_w
        self._ratio    = display_w / float(original_w)
        self._disp_h   = int(original_h * self._ratio)

        # Kasa tanımlarını hazırla
        if kasalar is None:
            kasalar = []
        self._kasalar = self._build_kasalar(kasalar)

        # İç durum — her kasaya ayrı sayaç
        for k in self._kasalar:
            k["total_seconds"] = 0.0
            k["is_active"]     = False
            k["session_start"] = None

        # draw() için son durum — update() çağrılmadan None
        self._last_status = None
        self._last_save_time = 0.0
        self._last_reset_date = None

        self._load_state()
        print(f"✅ KasiyerSuresiModule ready [{name}] — {len(self._kasalar)} kasa")

    # ==================== SETUP ====================

    def _build_kasalar(self, kasalar_cfg: list) -> list:
        """Config listesinden Shapely polygon ve scaled koordinatları üretir."""
        result = []
        for idx, cfg in enumerate(kasalar_cfg):
            kasa_id = cfg.get("id", f"KASA-{idx+1:02d}")
            coords  = [tuple(p) for p in cfg.get("coords", [])]
            if len(coords) < 3:
                print(f"[{self.name}] Uyarı: {kasa_id} için yetersiz koordinat, atlanıyor.")
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
                "date":   (self._last_reset_date.isoformat()
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
            print(f"[{self.name}] State dosyası yok, sıfırdan başlanıyor.")
            return
        try:
            with open(path, encoding="utf-8") as f:
                state = json.load(f)
            saved_date = state.get("date")
            today      = datetime.datetime.now().date().isoformat()
            if saved_date != today:
                print(f"[{self.name}] State eski tarihte ({saved_date}), sıfırdan başlanıyor.")
                return

            # Kaydedilen süreleri eşleştir
            saved_kasalar = {s["id"]: s for s in state.get("kasalar", [])}
            for k in self._kasalar:
                if k["id"] in saved_kasalar:
                    k["total_seconds"] = float(
                        saved_kasalar[k["id"]].get("total_seconds", 0.0)
                    )

            self._last_reset_date = datetime.date.fromisoformat(saved_date)
            totals = [f"{k['id']}={k['total_seconds']/60:.1f}dk" for k in self._kasalar]
            print(f"[{self.name}] State yüklendi: {', '.join(totals)}")
        except Exception as e:
            print(f"[{self.name}] State yükleme hatası: {e} — sıfırdan başlanıyor.")

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
            print(f"[{self.name}] Günlük sıfırlama → {today}")

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    # ==================== UPDATE ====================

    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        self._check_daily_reset()   # Her zaman ilk satır
        now = _time.time()

        # Hangi kasalar bu karede dolu?
        dolu_kasalar = {k["id"]: False for k in self._kasalar}

        if len(bboxes) > 0:
            for bbox, cls_id in zip(bboxes, class_ids):
                # Sadece "person" sınıfı (COCO: 0)
                if int(cls_id) != 0:
                    continue

                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                merkez = Point((x1 + x2) / 2, (y1 + y2) / 2)

                for k in self._kasalar:
                    if k["poly"].contains(merkez):
                        dolu_kasalar[k["id"]] = True
                        break   # Bir kişi birden fazla kasaya sayılmasın

        # Zaman birikimi
        for k in self._kasalar:
            dolu_mu = dolu_kasalar[k["id"]]
            if dolu_mu:
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

        # draw() için son durumu sakla
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
        # Ayrıca merkez noktaları draw() için sakla
        self._last_points = []
        if len(bboxes) > 0:
            for bbox, cls_id in zip(bboxes, class_ids):
                if int(cls_id) != 0:
                    continue
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                orig_cx = int((x1 + x2) / 2)
                orig_cy = int((y1 + y2) / 2)
                merkez  = Point(orig_cx, orig_cy)

                point_color = COLOR_PERSON_OUT
                for k in self._kasalar:
                    if k["poly"].contains(merkez):
                        point_color = k["border_color"]
                        break

                disp_cx = int(orig_cx * self._ratio)
                disp_cy = int(orig_cy * self._ratio)
                self._last_points.append((disp_cx, disp_cy, point_color))

        # Periyodik kayıt
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
        if self._last_status is None:   # update() henüz çağrılmadı
            return frame

        now = _time.time()

        # 1. Polygon sınırlarını çiz
        for k in self._kasalar:
            status = self._last_status[k["id"]]
            pts    = np.array(status["coords_scaled"], np.int32)
            cv2.polylines(frame, [pts], True, status["border_color"], 2, cv2.LINE_AA)

            # Etiket kapsülü (sol üst köşe)
            total = status["total_seconds"]
            if status["is_active"] and status["session_start"]:
                total += now - status["session_start"]

            metin       = f" {k['id']} | {self._fmt_time(total)} "
            yazi_x, yazi_y = status["coords_scaled"][0]
            yazi_y_adj  = max(yazi_y - 12, 25)
            (tw, th), _ = cv2.getTextSize(metin, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)

            # Arka plan kutusu
            cv2.rectangle(
                frame,
                (yazi_x - 2, yazi_y_adj - th - 5),
                (yazi_x + tw + 2, yazi_y_adj + 5),
                COLOR_DARK, -1,
            )
            # Renkli sol kenar çizgisi
            cv2.rectangle(
                frame,
                (yazi_x - 2, yazi_y_adj - th - 5),
                (yazi_x + 1, yazi_y_adj + 5),
                status["border_color"], -1,
            )
            # İnce siyah çerçeve
            cv2.rectangle(
                frame,
                (yazi_x - 2, yazi_y_adj - th - 5),
                (yazi_x + tw + 2, yazi_y_adj + 5),
                COLOR_DARK_BORDER, 1, cv2.LINE_AA,
            )
            cv2.putText(
                frame, metin,
                (yazi_x + 2, yazi_y_adj),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1, cv2.LINE_AA,
            )

        # 2. Kişi merkez noktaları
        if hasattr(self, "_last_points"):
            for (cx, cy, color) in self._last_points:
                cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)

        # 3. Sağ üst bilgi paneli
        if self.show_panel:
            frame = self._draw_panel(frame, now)

        return frame

    def _draw_panel(self, frame, now: float):
        panel_w  = 195
        line_h   = 25
        padding  = 10
        panel_h  = padding * 2 + line_h * len(self._kasalar)

        h, w     = frame.shape[:2]
        px1      = w - panel_w - 20
        py1      = 20
        px2      = w - 20
        py2      = py1 + panel_h

        # Yarı saydam arka plan
        overlay = frame.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), COLOR_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)
        cv2.rectangle(frame, (px1, py1), (px2, py2), COLOR_DARK_BORDER, 1, cv2.LINE_AA)

        for idx, k in enumerate(self._kasalar):
            total = k["total_seconds"]
            if k["is_active"] and k["session_start"]:
                total += now - k["session_start"]
            metin = f"{k['id']}: {self._fmt_time(total)}"
            y_pos = py1 + padding + line_h * idx + 18
            cv2.putText(
                frame, metin,
                (px1 + 12, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1, cv2.LINE_AA,
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
        print(f"[{self.name}] Shutdown tamamlandı, state kaydedildi.")

    # ==================== RESET ====================

    def reset(self):
        for k in self._kasalar:
            k["total_seconds"] = 0.0
            k["is_active"]     = False
            k["session_start"] = None
        self._save_state()
        print(f"[{self.name}] Manuel sıfırlama yapıldı.")
