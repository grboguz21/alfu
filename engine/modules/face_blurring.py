import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .base import BaseModule
from engine.shared_memory import GPU_LOCK


class FaceBlurringModule(BaseModule):

    def __init__(self, name: str,
                 model_path: str,
                 confidence: float = 0.45,
                 process_every_n: int = 2,
                 show_panel: bool = False):
        self.name            = name
        self.confidence      = confidence
        self.process_every_n = process_every_n
        self.show_panel      = show_panel

        self._model = YOLO(model_path)

        self._frame_counter      = 0
        self._current_face_count = 0
        self._daily_face_count   = 0
        self._last_reset_date    = None
        self._last_boxes         = []

        print(f"✅ Face Blurring Modülü hazır [{name}]")
        print(f"   ├── Model          : {model_path}")
        print(f"   ├── Confidence     : {confidence}")
        print(f"   ├── Process every  : {process_every_n} frame")
        print(f"   └── Yöntem         : kırmızı boya")

    def _check_daily_reset(self):
        today = datetime.datetime.now().date()
        if self._last_reset_date is None:
            self._last_reset_date = today
            return
        if today != self._last_reset_date:
            self._daily_face_count = 0
            self._last_reset_date  = today

    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        self._check_daily_reset()

        self._frame_counter += 1
        if self._frame_counter % self.process_every_n != 0:
            return

        with GPU_LOCK, torch.no_grad():
            results = self._model(frame, conf=self.confidence, verbose=False)
        boxes = []
        if results and results[0].boxes is not None and len(results[0].boxes):
            boxes = results[0].boxes.xyxy.cpu().numpy().tolist()

        if len(boxes) != self._current_face_count:
            print(f"[{self.name}] Yüz tespiti: {len(boxes)} yüz")

        self._last_boxes         = boxes
        self._current_face_count = len(boxes)
        self._daily_face_count  += len(boxes)

    def get_data(self) -> dict:
        return {
            "Bulunan Yuz Sayisi":  self._current_face_count,
            "Toplam Yuz (gunluk)": self._daily_face_count,
        }

    def draw(self, frame):
            h, w = frame.shape[:2]
            for (x1, y1, x2, y2) in self._last_boxes:
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                
                if x2 > x1 and y2 > y1:
                    # 1. Yüz bölgesini (ROI) kesiyoruz
                    face_roi = frame[y1:y2, x1:x2]
                    
                    # 2. Yüzün boyutuna göre dinamik blur şiddeti (kernel) belirliyoruz
                    # (Sıfır çıkmaması ve tek sayı olması için bitwise OR '| 1' kullandık)
                    kernel_w = int((x2 - x1) / 3) | 1
                    kernel_h = int((y2 - y1) / 3) | 1
                    kernel_w, kernel_h = max(3, kernel_w), max(3, kernel_h)
                    
                    # 3. Blurlanmış yüzü orijinal karedeki yerine yapıştırıyoruz
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(face_roi, (kernel_w, kernel_h), 0)

            if self.show_panel:
                self._draw_panel(frame)
            return frame

    def _draw_panel(self, frame):
        h, w = frame.shape[:2]
        px, py = 20, h - 100
        cv2.rectangle(frame, (px, py), (px + 300, py + 70), (20, 20, 20), -1)
        cv2.rectangle(frame, (px, py), (px + 300, py + 70), (0, 0, 255), 2)
        cv2.putText(frame,
                    f"Yuz: {self._current_face_count} (gunluk: {self._daily_face_count})",
                    (px + 10, py + 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return frame