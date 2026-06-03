import threading
from queue import Queue

# Buffers
CAP_BUFFER_SIZE         = 4
DETECTION_BUFFER_SIZE   = 4
CURRENT_BATCH_BUFFER_SIZE = 4
TRACKING_BUFFER_SIZE    = 4

# ✅ Thread'ler için threading.Lock kullan, multiprocessing.Lock değil
GPU_LOCK = threading.Lock()

class SharedMemory:
    cap_buffer           = Queue(maxsize=CAP_BUFFER_SIZE)
    detection_buffer     = Queue(maxsize=DETECTION_BUFFER_SIZE)
    tracking_buffer      = Queue(maxsize=TRACKING_BUFFER_SIZE)
    current_batch_buffer = Queue(maxsize=CURRENT_BATCH_BUFFER_SIZE)