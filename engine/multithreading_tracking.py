from engine.object_detection import ObjectDetection
from queue import Queue
import cv2
from engine.object_tracking import MultiObjectTracking
import time
from threading import Thread
import numpy as np
from engine.shared_memory import SharedMemory, TRACKING_BUFFER_SIZE, GPU_LOCK


class Cap:
    def __init__(self, resize=None, reconnect_delay_sec: int = 5,
                 on_disconnect=None, on_reconnect=None,
                 alarm_after_attempts: int = 2):
        assert isinstance(resize, float) or resize is None, "Resize must be a float"
        self.thread              = None
        self.cap                 = None
        self.resize              = resize
        self.frame_count         = 0
        self.reconnect_delay_sec = reconnect_delay_sec
        self.on_disconnect       = on_disconnect
        self.on_reconnect        = on_reconnect
        self.alarm_after_attempts = alarm_after_attempts
        self._source             = None
        self._backend            = cv2.CAP_ANY
        self._had_successful_frame = False
        self._disconnect_notified = False
        self._stop_requested     = False

    def stop(self):
        self._stop_requested = True

    def start_separate_thread(self):
        print("CAP: Starting Cap Thread")
        self.thread = Thread(target=self.grab_frames, args=())
        self.thread.daemon = True
        self.thread.start()
        print("CAP: Cap Thread Started")

    def load_video_capture(self, cam, backend=cv2.CAP_ANY):
        self._source  = cam
        self._backend = backend
        self.cap = cv2.VideoCapture(cam, backend)
        assert self.cap.isOpened(), f"CAP: Cannot open source: {cam}"
        return self.cap

    def _reconnect(self) -> bool:
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self._source, self._backend)
            return self.cap.isOpened()
        except Exception:
            return False

    def get_frame(self):
        ret, frame = self.cap.read()
        if self.resize is not None and ret and frame is not None:
            frame = cv2.resize(frame, None, fx=self.resize, fy=self.resize,
                               interpolation=cv2.INTER_AREA)
        self.frame_count += 1
        if ret and frame is not None:
            frame = frame.copy()
        return ret, frame

    def grab_frames(self):
        attempt = 0
        while not self._stop_requested:
            ret, frame = self.get_frame()

            if not ret:
                if not self.reconnect_delay_sec:
                    print("CAP: No more frames")
                    self.cap.release()
                    print("CAP: Releasing Cap")
                    SharedMemory.cap_buffer.put([False, None])
                    break

                attempt += 1
                delay = min(self.reconnect_delay_sec * attempt, 60)
                print(f"CAP: Stream disconnected — reconnecting in {delay}s "
                      f"(attempt {attempt})...")

                time.sleep(delay)

                if self._reconnect():
                    print(f"CAP: Reconnected! (attempt {attempt})")
                    if self._disconnect_notified and self.on_reconnect:
                        try:
                            self.on_reconnect()
                        except Exception:
                            pass
                    attempt = 0
                    self._disconnect_notified = False
                else:
                    print(f"CAP: Could not connect, will retry...")
                    if (attempt >= self.alarm_after_attempts
                            and not self._disconnect_notified
                            and self.on_disconnect and self._had_successful_frame
                            and not self._stop_requested):
                        try:
                            self.on_disconnect()
                        except Exception:
                            pass
                        self._disconnect_notified = True
                continue

            attempt = 0
            self._had_successful_frame = True

            if frame is None:
                continue

            if SharedMemory.cap_buffer.full():
                try:
                    SharedMemory.cap_buffer.get_nowait()
                except Exception:
                    pass
            SharedMemory.cap_buffer.put([ret, frame])


class MultiThreadingTracker:
    def __init__(self):
        self.cap                     = None
        self.od                      = None
        self.ot                      = None
        self.current_frame_detection = None
        self.resize                  = None

    def start_cap_thread(self, cam, resize=None, reconnect_delay_sec: int = 5,
                         on_disconnect=None, on_reconnect=None,
                         alarm_after_attempts: int = 2):
        self.cap = Cap(resize, reconnect_delay_sec=reconnect_delay_sec,
                       on_disconnect=on_disconnect, on_reconnect=on_reconnect,
                       alarm_after_attempts=alarm_after_attempts)
        if isinstance(cam, str) and cam.startswith("rtspsrc"):
            self.cap.load_video_capture(cam, cv2.CAP_GSTREAMER)
        else:
            self.cap.load_video_capture(cam)
        self.cap.start_separate_thread()

    def start_detection_thread(self, weights_path, device=0, batch_size=1, img_size=416,
                               filter_classes: list = None, detection_fps: float = None,
                               detection_client=None):  # ← YENİ parametre
        self.od = ObjectDetection(
            weights_path,
            batch_size       = batch_size,
            img_size         = img_size,
            device           = device,
            filter_classes   = filter_classes,
            detection_fps    = detection_fps,
            detection_client = detection_client,  # ← geçir
        )
        self.od.start_thread()

    def get_class_name(self, class_id):
        return self.od.classes[int(class_id)]

    def get_color(self, class_id):
        return self.od.colors[int(class_id)]

    def start_tracking_thread(self, tracker="ocsort", max_age=30, min_hits=3, iou_threshold=0.3):
        mot = MultiObjectTracking()
        tracker_instance = mot.ocsort(
            max_age       = max_age,
            min_hits      = min_hits,
            iou_threshold = iou_threshold,
        )
        self.ot = tracker_instance
        if self.od is not None:
            self.od.set_tracker(tracker_instance)
        print(f"Tracker: ocsort (max_age={max_age}, min_hits={min_hits}, "
              f"iou_threshold={iou_threshold}) — single thread mode")

    def get_frame(self):
        if self.cap is None:
            assert False, "Cap not initialized"

        elif self.cap is not None and self.od is None:
            while SharedMemory.cap_buffer.empty():
                time.sleep(0.01)
            return SharedMemory.cap_buffer.get()

        elif self.cap is not None and self.od is not None and self.ot is None:
            if SharedMemory.current_batch_buffer.empty():
                while SharedMemory.detection_buffer.empty():
                    time.sleep(0.001)
                rets, frames, r_bboxes, r_class_ids, r_scores = SharedMemory.detection_buffer.get()
                for ret, frame, r_bbox, r_class_id, r_score in zip(rets, frames, r_bboxes, r_class_ids, r_scores):
                    SharedMemory.current_batch_buffer.put([ret, frame, r_bbox, r_class_id, r_score])

        elif self.cap is not None and self.od is not None and self.ot is not None:
            if SharedMemory.tracking_buffer.empty():
                wait_start     = time.time()
                absolute_start = time.time()
                while SharedMemory.tracking_buffer.empty():
                    time.sleep(0.001)
                    cap_thread_alive = (self.cap.thread is not None and
                                        self.cap.thread.is_alive())
                    od_thread_alive  = (self.od is not None and
                                        self.od.thread is not None and
                                        self.od.thread.is_alive())
                    if (time.time() - wait_start > 2
                            and not cap_thread_alive
                            and SharedMemory.cap_buffer.empty()):
                        return False, None
                    if cap_thread_alive and not od_thread_alive:
                        return False, None
                    if cap_thread_alive and od_thread_alive:
                        wait_start = time.time()
                    if time.time() - absolute_start > 30:
                        print("WARNING: tracking_buffer timeout, resetting...", flush=True)
                        return False, None

        if not SharedMemory.tracking_buffer.empty():
            ret, frame, bboxes, class_ids, scores, obj_ids = SharedMemory.tracking_buffer.get()
            SharedMemory.current_batch_buffer.put([ret, frame, bboxes, class_ids, scores, obj_ids])
        else:
            return False, None

        while not SharedMemory.current_batch_buffer.empty():
            self.current_frame_detection = SharedMemory.current_batch_buffer.get()
            return self.current_frame_detection[0], self.current_frame_detection[1]

        return False, None

    def get_detection(self):
        if self.od is None:
            assert False, "Object Detection not initialized"
        if self.ot is not None:
            assert False, "Object Tracking is initialized, use get_tracking() instead."
        return self.current_frame_detection[2:]

    def get_tracking(self):
        if self.ot is None:
            assert False, "Object Tracking not initialized"
        return self.current_frame_detection[2:]

    def release(self):
        if self.cap is not None:
            self.cap.stop()
            self.cap.cap.release()
            print("CAP: Releasing Cap")
        if self.od is not None:
            self.od.release()
            print("OD: Releasing Object Detection")
        cv2.destroyAllWindows()
        print("All resources released.")