import time
import numpy as np
import platform
if platform.system() == "Darwin":
    import os
    print("MAC OS: Setting MPS Fallback")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random
import colorsys
from threading import Thread
from engine.shared_memory import SharedMemory, DETECTION_BUFFER_SIZE, GPU_LOCK

random.seed(0)


class ObjectDetection:
    def __init__(self, weights_path, batch_size=32, img_size=416, device=0,
                 filter_classes: list = None, detection_fps: float = None,
                 tracker=None, detection_client=None):
        print("Object Detection: Loading YOLO model")
        self.weights_path     = weights_path
        self.colors           = self.random_colors(81)
        self.img_size         = img_size
        self.batch_size       = batch_size
        self.device           = device
        self.filter_classes   = filter_classes
        self.detection_fps    = detection_fps
        self.processed_frames_count = 0
        self.thread           = None
        self._stop_requested  = False
        self._tracker         = tracker
        self._client          = detection_client  # ← DetectionService client

        if detection_client is not None:
            # Merkezi servis modu — model yüklenmez
            self.model   = None
            self.classes = {}
            print("Object Detection: Merkezi DetectionService kullanılıyor")
        else:
            # Klasik mod — model bu process'te yüklenir
            from ultralytics import YOLO
            self.model   = YOLO(self.weights_path, task="detect")
            self.classes = self.model.names
            print("Object Detection: Model yüklendi")

    def set_tracker(self, tracker):
        self._tracker = tracker

    def _reload_model(self):
        if self._client is not None:
            return True  # servis modunu reload etmeye gerek yok
        import torch
        from ultralytics import YOLO
        print("OD: CUDA error detected, reloading model...", flush=True)
        try:
            torch.cuda.empty_cache()
            self.model = YOLO(self.weights_path, task="detect")
            print("OD: Model reloaded successfully", flush=True)
            return True
        except Exception as reload_err:
            print(f"OD: Model reload failed: {reload_err}", flush=True)
            return False

    def start_thread(self):
        print("Object Detection: Starting Detection Thread")
        print("Object Detection: Batch size {}".format(self.batch_size))
        self.thread = Thread(target=self.process_image, args=())
        self.thread.daemon = True
        self.thread.start()
        print("Object Detection: Detection Thread Started")

    def process_image(self):
        if self._client is None:
            # Klasik mod warm-up
            print("Object Detection: Warm up model")
            self.detect_batch([np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                               for _ in range(self.batch_size)])
        print("Object Detection: Started Processing Frames")

        _min_interval  = (1.0 / self.detection_fps) if self.detection_fps else 0.0
        _last_detect_t = 0.0

        if self.detection_fps:
            print(f"Object Detection: FPS limit {self.detection_fps} FPS "
                  f"({_min_interval*1000:.0f}ms/frame)")

        while not self._stop_requested:
            if _min_interval > 0:
                elapsed = time.time() - _last_detect_t
                if elapsed < _min_interval:
                    time.sleep(min(_min_interval - elapsed, 0.005))
                    continue

            if SharedMemory.cap_buffer.qsize() > 0:
                rets   = []
                frames = []
                frame_indexes = []

                while len(frames) < self.batch_size and not SharedMemory.cap_buffer.empty():
                    ret, frame = SharedMemory.cap_buffer.get()

                    if not ret and frame is None:
                        if len(frames) > 0:
                            if len(frames) < self.batch_size:
                                padding_needed = self.batch_size - len(frames)
                                last_frame = frames[-1].copy() if isinstance(frames[-1], np.ndarray) else frames[-1]
                                frames.extend([last_frame.copy() for _ in range(padding_needed)])
                                frame_indexes = list(range(len(rets)))

                            try:
                                r_bboxes, r_class_ids, r_scores = self._detect(frames)
                            except Exception as e:
                                print(f"OD: Detection error in EOS path (skipped): {e}", flush=True)
                                if "CUDA" in str(e):
                                    self._reload_model()
                                r_bboxes    = [np.empty((0,4),dtype=int)]  * len(rets)
                                r_class_ids = [np.empty((0,),dtype=int)]   * len(rets)
                                r_scores    = [np.empty((0,),dtype=float)] * len(rets)

                            if len(frame_indexes) > 0:
                                real_count  = len(frame_indexes)
                                r_bboxes    = r_bboxes[:real_count]
                                r_class_ids = r_class_ids[:real_count]
                                r_scores    = r_scores[:real_count]

                            self._run_tracker_and_put(
                                rets, frames[:len(rets)], r_bboxes, r_class_ids, r_scores, eos=False
                            )

                        SharedMemory.tracking_buffer.put([False, None, [], [], [], []])
                        print("OD+OT: End of stream detected")
                        return

                    if ret and frame is not None:
                        rets.append(ret)
                        frames.append(frame.copy())

                frames = [f for f in frames if f is not None and
                          isinstance(f, np.ndarray) and f.size > 0]

                if len(frames) > 0:
                    frame_indexes = []
                    if len(frames) < self.batch_size:
                        frame_indexes  = list(range(len(frames)))
                        padding_needed = self.batch_size - len(frames)
                        last_frame     = frames[-1].copy()
                        for _ in range(padding_needed):
                            frames.append(last_frame.copy())

                    try:
                        r_bboxes, r_class_ids, r_scores = self._detect(frames)
                    except Exception as e:
                        print(f"OD: Detection error (frame skipped): {e}", flush=True)
                        if "CUDA" in str(e):
                            success = self._reload_model()
                            if not success:
                                print("OD: Cannot recover, stopping detection thread.", flush=True)
                                self._stop_requested = True
                                return
                        time.sleep(0.001)
                        continue

                    if len(frame_indexes) > 0:
                        real_count  = len(frame_indexes)
                        r_bboxes    = r_bboxes[:real_count]
                        r_class_ids = r_class_ids[:real_count]
                        r_scores    = r_scores[:real_count]

                    self._run_tracker_and_put(
                        rets, frames[:len(rets)], r_bboxes, r_class_ids, r_scores, eos=False
                    )

                    self.processed_frames_count += len(rets)
                    _last_detect_t = time.time()

            time.sleep(0.001)

    # ------------------------------------------------------------------
    # _detect — client varsa servise sor, yoksa local batch
    # ------------------------------------------------------------------

    def _detect(self, frames):
        if self._client is not None:
            return self._detect_via_service(frames)
        return self.detect_batch(frames)

    def _detect_via_service(self, frames):
        """Her frame'i servise gönder, sonuçları topla."""
        real_count = len(frames)

        for frame in frames:
            self._client.push_frame(frame)

        # Her frame için ayrı sonuç bekle — servis batch'i bölebilir
        # Basit yaklaşım: tek bir toplu sonuç bekle
        result = None
        for _ in range(200):  # max 2sn bekle
            result = self._client.get_result()
            if result is not None:
                break
            time.sleep(0.01)

        if result is None:
            empty_b = [np.empty((0, 4), dtype=int)]   * real_count
            empty_c = [np.empty((0,),   dtype=int)]   * real_count
            empty_s = [np.empty((0,),   dtype=float)] * real_count
            return empty_b, empty_c, empty_s

        bboxes, class_ids, scores = result
        # Tüm frame'lere aynı sonucu ver (servis en son frame'i işledi)
        r_bboxes    = [bboxes]    * real_count
        r_class_ids = [class_ids] * real_count
        r_scores    = [scores]    * real_count
        return r_bboxes, r_class_ids, r_scores

    # ------------------------------------------------------------------

    def _run_tracker_and_put(self, rets, frames, r_bboxes, r_class_ids, r_scores, eos=False):
        from engine.shared_memory import TRACKING_BUFFER_SIZE

        if self._tracker is None:
            while SharedMemory.detection_buffer.qsize() + 1 >= DETECTION_BUFFER_SIZE:
                time.sleep(0.01)
            safe_frames = [f.copy() for f in frames if f is not None]
            SharedMemory.detection_buffer.put([rets, safe_frames, r_bboxes, r_class_ids, r_scores])
            return

        for ret, frame, r_bbox, r_class_id, r_score in zip(rets, frames, r_bboxes, r_class_ids, r_scores):
            while SharedMemory.tracking_buffer.qsize() >= TRACKING_BUFFER_SIZE:
                time.sleep(0.01)

            safe_frame = frame.copy() if frame is not None else None

            try:
                r_bboxes_ids = self._tracker.update(r_bbox, r_score, r_class_id, safe_frame)
            except Exception as e:
                print(f"OT: Tracker error (skipped): {e}", flush=True)
                r_bboxes_ids = np.empty((0, 7))

            if r_bboxes_ids is not None and hasattr(r_bboxes_ids, 'size') and r_bboxes_ids.size > 0:
                bboxes    = r_bboxes_ids[:, :4].copy()
                obj_ids   = r_bboxes_ids[:, 4].copy()
                scores    = r_bboxes_ids[:, 6].copy()
                class_ids = r_bboxes_ids[:, 5].copy()
            else:
                bboxes    = np.empty((0, 4))
                obj_ids   = np.empty((0,))
                scores    = np.empty((0,))
                class_ids = np.empty((0,))

            SharedMemory.tracking_buffer.put([
                ret,
                safe_frame.copy() if safe_frame is not None else None,
                bboxes,
                class_ids,
                scores,
                obj_ids,
            ])

    def random_colors(self, N, bright=False):
        brightness = 255 if bright else 180
        hsv = [(i / N + 1, 1, brightness) for i in range(N + 1)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def detect_batch(self, frames, imgsz=None, conf=0.25, nms=True, classes=None, device=None):
        if self.model is None:
            n = len(frames)
            return ([np.empty((0,4),dtype=int)]  * n,
                    [np.empty((0,),dtype=int)]   * n,
                    [np.empty((0,),dtype=float)] * n)
        imgsz          = imgsz if imgsz is not None else self.img_size
        filter_classes = classes if classes else self.filter_classes
        device         = device if device else self.device
        with GPU_LOCK:
            results = self.model.predict(
                source   = frames,
                save     = False,
                save_txt = False,
                imgsz    = imgsz,
                conf     = conf,
                nms      = nms,
                classes  = filter_classes,
                device   = device,
                half     = True,
                verbose  = False,
            )
        r_bboxes, r_class_ids, r_scores = [], [], []
        for result in results:
            r_bboxes.append(np.array(result.boxes.xyxy.cpu(), dtype="int"))
            r_class_ids.append(np.array(result.boxes.cls.cpu(), dtype="int"))
            r_scores.append(np.array(result.boxes.conf.cpu(), dtype="float").round(2))
        return r_bboxes, r_class_ids, r_scores

    def release(self):
        self._stop_requested = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.model = None
        if self._client is None:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def detect(self, frame, imgsz=416, conf=0.25, nms=True, classes=None, device=None):
        filter_classes = classes if classes else None
        device         = device if device else self.device
        with GPU_LOCK:
            results = self.model.predict(
                source   = frame,
                save     = False,
                save_txt = False,
                imgsz    = imgsz,
                conf     = conf,
                nms      = nms,
                classes  = filter_classes,
                half     = True,
                device   = device,
                verbose  = False,
            )
        result    = results[0]
        bboxes    = np.array(result.boxes.xyxy.cpu(), dtype="int")
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        scores    = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
        return bboxes, class_ids, scores