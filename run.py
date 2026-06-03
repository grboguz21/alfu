import subprocess
import sys
import os
import signal
import queue
import time
import gc
from threading import Thread
from datetime import datetime
from pathlib import Path

# NVIDIA CUDA Kilitlenme Koruma Değişkenleri
os.environ["CUDA_MODULE_LOADING"] = "EAGER"
os.environ["PYTHONUNBUFFERED"] = "1"

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def cleanup_old_logs(keep: int = 10):
    logs = sorted(LOG_DIR.glob("run_*.log"), key=lambda p: p.stat().st_mtime)
    for old in logs[:-keep]:
        old.unlink(missing_ok=True)

command = [sys.executable, "main.py"] + sys.argv[1:]

_shutdown_requested = False

def _on_sigint(signum, _):
    global _shutdown_requested
    _shutdown_requested = True

signal.signal(signal.SIGINT,  _on_sigint)
signal.signal(signal.SIGTERM, _on_sigint)

RESTART_DELAY = 5  # saniye
restart_count = 0
exit_code     = 0

def stream_reader(pipe, log_queue):
    try:
        for line in iter(pipe.readline, ''):
            if not line:
                break
            log_queue.put(line)
    except Exception:
        pass
    finally:
        pipe.close()

while not _shutdown_requested:
    cleanup_old_logs(keep=10)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file  = LOG_DIR / f"run_{timestamp}.log"

    if restart_count == 0:
        print(f"[run.py] Log file: {log_file}")
        print(f"[run.py] Command: {' '.join(command)}\n")
    else:
        print(f"[run.py] Restart #{restart_count} — Log: {log_file}")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Start: {datetime.now().isoformat()}\n")
        f.write(f"Command: {' '.join(command)}\n")
        f.write(f"Restart count: {restart_count}\n")
        f.write("-" * 60 + "\n")
        f.flush()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ},
            bufsize=1,
        )

        log_queue     = queue.Queue()
        reader_thread = Thread(
            target=stream_reader,
            args=(process.stdout, log_queue),
            daemon=True,
        )
        reader_thread.start()

        try:
            while True:
                running = process.poll()

                while not log_queue.empty():
                    try:
                        line = log_queue.get_nowait()
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        f.write(line)
                        f.flush()
                    except queue.Empty:
                        break

                if _shutdown_requested:
                    print("\n[run.py] Shutdown requested, stopping...")
                    process.send_signal(signal.SIGINT)
                    break

                if running is not None and log_queue.empty():
                    break

                time.sleep(0.01)

        except KeyboardInterrupt:
            _shutdown_requested = True
            print("\n[run.py] Ctrl+C received, shutting down...")
            process.send_signal(signal.SIGINT)

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("[run.py] Timeout, forcing shutdown...")
            process.kill()
            process.wait()

        exit_code = process.returncode
        f.write("-" * 60 + "\n")
        f.write(f"End: {datetime.now().isoformat()} | Exit code: {exit_code}\n")

    print(f"\n[run.py] Process ended. Exit code: {exit_code}")

    # Normal çıkış veya kullanıcı durdurduysa restart etme
    if exit_code == 0 or _shutdown_requested:
        print("[run.py] Clean exit, not restarting.")
        break

    # # Maksimum restart sayısına ulaşıldıysa dur
    # if restart_count >= MAX_RESTARTS:
    #     print(f"[run.py] Max restarts ({MAX_RESTARTS}) reached, giving up.")
    #     break

    restart_count += 1
    print(f"[run.py] Crashed (exit {exit_code}). Restarting in {RESTART_DELAY}s... "
        f"(attempt #{restart_count})")
    # Bellek temizliği — bir sonraki process için
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        print("[run.py] CUDA cache cleared.")
    except Exception:
        pass

    time.sleep(RESTART_DELAY)

sys.exit(exit_code)