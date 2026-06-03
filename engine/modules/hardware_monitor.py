"""
Hardware Monitor Module
-----------------------
Logs CPU, RAM, GPU and disk usage to terminal periodically.
Does not send data to API.

Requirements: psutil, pynvml
    pip install psutil pynvml

Config example:
    {
        "type": "hardware_monitor",
        "name": "donanim_izleme",
        "gpu_index": 0,
        "log_interval_sec": 30
    }
"""

import time
import datetime
import psutil
from .base import BaseModule

try:
    from pynvml import (nvmlInit, nvmlDeviceGetHandleByIndex,
                        nvmlDeviceGetName, nvmlDeviceGetUtilizationRates,
                        nvmlDeviceGetMemoryInfo, nvmlDeviceGetTemperature,
                        NVML_TEMPERATURE_GPU)
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False


class HardwareMonitorModule(BaseModule):
    """Logs system hardware usage to terminal."""

    def __init__(self, name: str, gpu_index: int = 0, log_interval_sec: int = 30):
        self.name             = name
        self.log_interval_sec = log_interval_sec
        self._gpu_handle      = None
        self._last_log_time   = 0.0

        if _NVML_AVAILABLE:
            try:
                # nvmlInit must be called inside worker process (after fork)
                nvmlInit()
                self._gpu_handle = nvmlDeviceGetHandleByIndex(gpu_index)
                gpu_name = nvmlDeviceGetName(self._gpu_handle)
                print(f"[{self.name}] GPU found: {gpu_name}")
            except Exception as e:
                print(f"[{self.name}] GPU initialization failed: {e}")
        else:
            print(f"[{self.name}] pynvml not found — pip install nvidia-ml-py")

    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names):
        now = time.time()
        if now - self._last_log_time >= self.log_interval_sec:
            self._log()
            self._last_log_time = now

    def _log(self):
        mem  = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        ts   = datetime.datetime.now().strftime("%H:%M:%S")

        parts = [
            f"CPU: {psutil.cpu_percent(interval=None):.1f}%",
            f"RAM: {mem.percent:.1f}% ({mem.used // 1024**3}/{mem.total // 1024**3} GB)",
            f"Disk: {disk.percent:.1f}%",
        ]

        if self._gpu_handle:
            try:
                util     = nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem_info = nvmlDeviceGetMemoryInfo(self._gpu_handle)
                temp     = nvmlDeviceGetTemperature(self._gpu_handle, NVML_TEMPERATURE_GPU)
                gpu_used  = mem_info.used  // 1024**2
                gpu_total = mem_info.total // 1024**2
                parts += [
                    f"GPU: {util.gpu}%",
                    f"VRAM: {gpu_used}/{gpu_total} MB ({mem_info.used / mem_info.total * 100:.1f}%)",
                    f"GPU Temp: {temp}°C",
                ]
            except Exception as e:
                parts.append(f"GPU error: {e}")

        print(f"[{ts}][{self.name}] " + " | ".join(parts))

    def get_data(self) -> dict:
        return {}

    def draw(self, frame):
        return frame
