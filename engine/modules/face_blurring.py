import datetime
import cv2
import numpy as np

from .base import BaseModule


class FaceBlurringModule(BaseModule):
    """
    Yüz tespiti ve blur.
    
    İki mod:
    1. Servis modu (önerilen): face_service_client ile çalışır, model yüklenmez
    2. Klasik mod: model_path ile kendi modelini yükler (eski davranış)
    
    Config örneği (servis modu):
        {
            "type": "face_blurring",
            "name": "yuz_blur",
            "confidence": 0.45,
            "process_every_n": 2
        }
    
    Config örneği (klasik mod):
        {
            "type": "face_blurring", 
            "name": "yuz_blur",
            "model_path": "models/yolov8l_100e.engine",
            "confidence": 0.45,
            "process_every_n": 2
        }
    """

    def __init__(self, name: str,
                 model_path: str = None,
                 confidence: float = 0.45,
                 process_every_n: int = 2,
                 show_panel: bool = False,
                 face_service_client=None,   # ← servis modu
                 **_kwargs):
        self.name            = name
        self.confidence      = confidence
        self.process_every_n = process_every_n
        self.show_panel      = show_panel
        self._client         = face_service_client
        self._model          = None

        if face_service_client is not None:
            # Servis modu — model yüklenmez
            print(f"✅ Face Blurring Modülü hazır [{name}] (servis modu)")
        elif model_path:
            # Klasik mod — model bu process'te yüklenir
            import torch
            from ultralytics import YOLO
            from engine.shared_memory import GPU_LOCK
            self._model    = YOLO(model_path)
            self._GPU_LOCK = GPU_LOCK
            self._torch    = torch
            print(f"✅ Face Blurring Modülü hazır [{name}] (klasik mod)")
            print(f"   ├── Model: {model_path}")
        else:
            print(f"⚠️  Face Blurring [{name}]: ne client ne model_path var — blur devre dışı")

        self._frame_counter      = 0
        self._current_face_count = 0
        self._daily_face_count   = 0
        self._last_reset_date    = None
        self._last_boxes         = []

    # ------------------------------------------------------------------

    def _check_daily_reset(self):
        today = datetime.datetime.now().date()
        if self._last_reset_date is None:
            self._last_reset_date = today
            return
        if today != self._last_reset_date:
            self._daily_face_count = 0
            self._last_reset_date  = today

    # ------------------------------------------------------------------

    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        self._check_daily_reset()

        self._frame_counter += 1
        if self._frame_counter % self.process_every_n != 0:
            return

        boxes = []

        if self._client is not None:
            # Servis modu
            self._client.push_frame(frame)
            result = self._client.get_result()
            if result is not None:
                det_bboxes, det_class_ids, det_scores = result
                for bbox, score in zip(det_bboxes, det_scores):
                    if float(score) >= self.confidence:
                        boxes.append(bbox[:4].tolist())

        elif self._model is not None:
            # Klasik mod
            with self._GPU_LOCK, self._torch.no_grad():
                results = self._model(frame, conf=self.confidence, verbose=False)
            if results and results[0].boxes is not None and len(results[0].boxes):
                boxes = results[0].boxes.xyxy.cpu().numpy().tolist()

        if boxes or self._client is not None:
            # Servis modunda her zaman güncelle (boş olsa da)
            self._last_boxes         = boxes
            self._current_face_count = len(boxes)
            self._daily_face_count  += len(boxes)
        elif self._model is not None:
            self._last_boxes         = boxes
            self._current_face_count = len(boxes)
            self._daily_face_count  += len(boxes)

    # ------------------------------------------------------------------

    def get_data(self) -> dict:
        return {
            "Bulunan Yuz Sayisi":  self._current_face_count,
            "Toplam Yuz (gunluk)": self._daily_face_count,
        }

    # ------------------------------------------------------------------

    def draw(self, frame):
        h, w = frame.shape[:2]
        for (x1, y1, x2, y2) in self._last_boxes:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 > x1 and y2 > y1:
                face_roi = frame[y1:y2, x1:x2]
                kernel_w = max(3, int((x2 - x1) / 3) | 1)
                kernel_h = max(3, int((y2 - y1) / 3) | 1)
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(face_roi, (kernel_w, kernel_h), 0)
        if self.show_panel:
            self._draw_panel(frame)
        return frame

    def _draw_panel(self, frame):
        h, w   = frame.shape[:2]
        px, py = 20, h - 100
        cv2.rectangle(frame, (px, py), (px + 300, py + 70), (20, 20, 20), -1)
        cv2.rectangle(frame, (px, py), (px + 300, py + 70), (0, 0, 255),  2)
        cv2.putText(frame,
                    f"Yuz: {self._current_face_count} (gunluk: {self._daily_face_count})",
                    (px + 10, py + 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def shutdown(self):
        self._model = None