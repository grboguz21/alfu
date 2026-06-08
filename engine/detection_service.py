"""
DetectionService
----------------
Tek YOLO modeli, tüm kameralar paylaşır.
subprocess ile başlatılır — tamamen izole process, temiz CUDA context.

Haberleşme: multiprocessing.Queue (spawn öncesi oluşturulur, pickle ile geçer)
"""

import os
import sys
import time
import multiprocessing
import numpy as np


_REQUEST_MAXSIZE  = 64
_RESPONSE_MAXSIZE = 8


# ---------------------------------------------------------------------------
# DetectionService
# ---------------------------------------------------------------------------

class DetectionService:

    def __init__(self,
                 model_path : str,
                 device     : int   = 0,
                 batch_size : int   = 8,
                 img_size   : int   = 416,
                 conf       : float = 0.25):
        self.model_path = model_path
        self.device     = device
        self.batch_size = batch_size
        self.img_size   = img_size
        self.conf       = conf

        self._request_queue   = multiprocessing.Queue(maxsize=_REQUEST_MAXSIZE)
        self._response_queues : dict = {}
        self._names_queue     = multiprocessing.Queue(maxsize=1)
        self._process         = None
        self._model_names     = None

    def make_client(self, cam_id: str,
                    filter_classes: list = None,
                    conf: float = None) -> "DetectionClient":
        q = multiprocessing.Queue(maxsize=_RESPONSE_MAXSIZE)
        self._response_queues[cam_id] = q
        return DetectionClient(
            cam_id         = cam_id,
            request_queue  = self._request_queue,
            response_queue = q,
            filter_classes = filter_classes,
            conf           = conf if conf is not None else self.conf,
        )

    def start(self):
        self._process = multiprocessing.Process(
            target = _service_worker,
            args   = (
                self.model_path,
                self.device,
                self.batch_size,
                self.img_size,
                self.conf,
                self._request_queue,
                self._response_queues,
                self._names_queue,
            ),
            daemon = False,
            name   = "DetectionService",
        )
        self._process.start()
        print(f"[DetectionService] PID {self._process.pid} başlatıldı — {self.model_path}")

        # Model hazır olana kadar bekle (max 120sn — TRT yavaş yüklenebilir)
        try:
            self._model_names = self._names_queue.get(timeout=120)
            print(f"[DetectionService] ✅ Model hazır — {len(self._model_names)} sınıf")
        except Exception:
            print("[DetectionService] ❌ Model yüklenemedi (timeout)")
            self._model_names = {}

    def stop(self):
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
            print("[DetectionService] Durduruldu.")

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def model_names(self) -> dict:
        return self._model_names or {}


# ---------------------------------------------------------------------------
# Worker — ayrı process, temiz CUDA context
# ---------------------------------------------------------------------------

def _service_worker(model_path, device, batch_size, img_size, conf,
                    request_queue, response_queues, names_queue):

    # Temiz CUDA başlangıcı
    os.environ["CUDA_MODULE_LOADING"]  = "LAZY"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    import torch
    import numpy as np
    from ultralytics import YOLO

    print(f"[DetectionService] Worker PID={os.getpid()} model yükleniyor: {model_path}")

    try:
        model = YOLO(model_path, task="detect")
    except Exception as e:
        print(f"[DetectionService] Model yüklenemedi: {e}", flush=True)
        return

    print(f"[DetectionService] Model yüklendi — {len(model.names)} sınıf")

    try:
        names_queue.put(model.names, timeout=5)
    except Exception:
        pass

    # Warm-up — batch=1 ile
    try:
        dummy = [np.zeros((img_size, img_size, 3), dtype=np.uint8)]
        _infer(model, dummy, device, img_size, conf, None)
        print("[DetectionService] Warm-up tamamlandı. İstekler bekleniyor...")
    except Exception as e:
        print(f"[DetectionService] Warm-up hatası: {e}", flush=True)

    # Ana döngü
    while True:
        requests = []
        try:
            item = request_queue.get(timeout=1.0)
            requests.append(item)
        except Exception:
            continue

        while len(requests) < batch_size:
            try:
                requests.append(request_queue.get_nowait())
            except Exception:
                break

        if not requests:
            continue

        cam_ids    = [r[0] for r in requests]
        frames     = [r[1] for r in requests]
        filter_cls = requests[0][2]
        conf_ov    = requests[0][3]
        real_count = len(frames)

        # Batch pad
        if real_count < batch_size:
            frames = frames + [frames[-1]] * (batch_size - real_count)

        try:
            r_bboxes, r_class_ids, r_scores = _infer(
                model, frames, device, img_size, conf_ov or conf, filter_cls
            )
        except Exception as e:
            print(f"[DetectionService] Inference hatası: {e}", flush=True)
            empty = (
                np.empty((0, 4), dtype=int),
                np.empty((0,),   dtype=int),
                np.empty((0,),   dtype=float),
            )
            for cam_id in cam_ids:
                q = response_queues.get(cam_id)
                if q:
                    _safe_put(q, empty)
            continue

        for i, cam_id in enumerate(cam_ids):
            q = response_queues.get(cam_id)
            if q is None:
                continue
            _safe_put(q, (r_bboxes[i], r_class_ids[i], r_scores[i]))


def _infer(model, frames, device, img_size, conf, filter_classes):
    """Frame frame inference — engine batch=1 ile derlenmişse güvenli."""
    import torch
    r_bboxes, r_class_ids, r_scores = [], [], []
    for frame in frames:
        with torch.no_grad():
            results = model.predict(
                source   = frame,
                save     = False,
                save_txt = False,
                imgsz    = img_size,
                conf     = conf,
                nms      = True,
                classes  = filter_classes,
                device   = device,
                half     = True,
                verbose  = False,
            )
        r = results[0]
        r_bboxes.append(np.array(r.boxes.xyxy.cpu(),  dtype="int"))
        r_class_ids.append(np.array(r.boxes.cls.cpu(), dtype="int"))
        r_scores.append(np.array(r.boxes.conf.cpu(),  dtype="float").round(2))
    return r_bboxes, r_class_ids, r_scores


def _safe_put(q, item):
    if q.full():
        try:
            q.get_nowait()
        except Exception:
            pass
    try:
        q.put_nowait(item)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# DetectionClient — kamera process içinde
# ---------------------------------------------------------------------------

class DetectionClient:

    def __init__(self, cam_id, request_queue, response_queue,
                 filter_classes=None, conf=0.25):
        self.cam_id         = cam_id
        self._req_q         = request_queue
        self._res_q         = response_queue
        self.filter_classes = filter_classes
        self.conf           = conf
        self._frame_counter = 0
        self._last_result   = None
        self._model_names   = {}

    def push_frame(self, frame: np.ndarray, process_every_n: int = 1):
        self._frame_counter += 1
        if self._frame_counter % process_every_n != 0:
            return
        _safe_put(self._req_q,
                  (self.cam_id, frame.copy(), self.filter_classes, self.conf))

    def get_result(self):
        while not self._res_q.empty():
            try:
                self._last_result = self._res_q.get_nowait()
            except Exception:
                break
        return self._last_result