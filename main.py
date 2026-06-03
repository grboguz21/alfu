# YOLO multiprocessing

import argparse
import cv2
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path


import torch
import multiprocessing
from engine.multithreading_tracking import MultiThreadingTracker
from engine.minio_manager import MinIOManager
from engine.modules import ModuleRunner
from engine.rtsp_probe import build_gst_pipeline
from report_manager import ReportManager


def load_config(path: str = "config.json") -> dict:
    config_path = Path(path)
    assert config_path.exists(), f"Config file not found: {path}"
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    cameras_dir = config_path.parent / "cameras"
    cameras = []
    if cameras_dir.is_dir():
        for cam_file in sorted(cameras_dir.glob("*.json")):
            with open(cam_file, encoding="utf-8") as f:
                cameras.append(json.load(f))
    cfg["cameras"] = cameras
    return cfg

def track_video(video_path=None,
                engine_path=None,
                tracker="ocsort",
                batch_size=1,
                device=0,
                img_size=640,
                max_age=30,
                min_hits=3,
                iou_threshold=0.3,
                gst_pipeline=None,
                minio_folder="Kamera",
                display=False,
                ):

    mtt = MultiThreadingTracker()

    cfg = load_config()

    cam_cfg = next((c for c in cfg["cameras"] if c["minio_folder"] == minio_folder), {})

    m = cfg["minio"]
    minio = MinIOManager(
        endpoint=m["endpoint"],
        access_key=m["access_key"],
        secret_key=m["secret_key"],
        branch_name=m["branch_name"],
        secure=m.get("secure", False),
        bucket_name=m.get("bucket_name", "ai-outputs")
    )

    a = cfg["api"]
    report = ReportManager(
        gateway_base=a["gateway_base"],
        api_key=a["api_key"],
        branch_id=a["branch_id"],
        log_file=f"queue_log_{minio_folder}.json",
    )

    _frame_resize = cam_cfg.get("frame_resize", 1.0)
    modules = ModuleRunner(cam_cfg.get("modules", []), frame_resize=_frame_resize)

    _reports_cfg = cam_cfg.get("reports") or []
    if not _reports_cfg and cam_cfg.get("report_interval_seconds"):
        _reports_cfg = [{
            "name":             cam_cfg.get("report_name", "istasyon_doluluk"),
            "module_id":        cam_cfg.get("module_id"),
            "interval_seconds": cam_cfg["report_interval_seconds"],
            "fields":           cam_cfg.get("report_fields"),
        }]

    for r in _reports_cfg:
        if r.get("type") == "alarm":
            report.add_alarm(
                name=r["name"],
                cooldown_seconds=r["interval_seconds"],
                camera_id=cam_cfg.get("camera_id"),
                module_id=r.get("module_id") or cam_cfg.get("module_id"),
                data={},
            )
        else:
            report.add_periodic_report(
                name=r["name"],
                interval_seconds=r["interval_seconds"],
                data_func=modules.get_data,
                camera_id=cam_cfg.get("camera_id"),
                module_id=r.get("module_id") or cam_cfg.get("module_id"),
            )

    for alarm_cfg in cam_cfg.get("alarms", []):
        report.add_alarm(
            name=alarm_cfg["name"],
            cooldown_seconds=alarm_cfg.get("cooldown_seconds", 30),
            camera_id=cam_cfg.get("camera_id"),
            module_id=alarm_cfg.get("module_id") or cam_cfg.get("module_id"),
            once_per_day=alarm_cfg.get("once_per_day", False),
            data=alarm_cfg.get("data", {}),
        )

    source = gst_pipeline if gst_pipeline is not None else video_path
    assert source is not None, "video_path or gst_pipeline must be provided"
    reconnect_delay_sec = cam_cfg.get("reconnect_delay_sec", 5)

    stream_alarm_cfg = cam_cfg.get("stream_alarm", {})
    if gst_pipeline and stream_alarm_cfg.get("enabled", True):
        report.add_alarm(
            name="stream_kesintisi",
            cooldown_seconds=stream_alarm_cfg.get("cooldown_seconds", 60),
            camera_id=cam_cfg.get("camera_id"),
            module_id=stream_alarm_cfg.get("module_id") or cam_cfg.get("module_id"),
            data=stream_alarm_cfg.get("data", {"Status": "Camera Connection Lost"}),
        )

    move_alarm_cfg = cam_cfg.get("camera_move_alarm", {})
    move_detector  = None
    if move_alarm_cfg.get("enabled", False) and move_alarm_cfg.get("rois"):
        from engine.modules.camera_move_detection import CameraMoveDetector
        report.add_alarm(
            name="camera_move",
            cooldown_seconds=move_alarm_cfg.get("cooldown_seconds", 60),
            camera_id=cam_cfg.get("camera_id"),
            module_id=cam_cfg.get("module_id"),
            data=move_alarm_cfg.get("data", {"Status": "Camera Angle Changed"}),
        )
        _rois = move_alarm_cfg["rois"]
        move_detector = CameraMoveDetector(
            rois       = _rois,
            show_panel = move_alarm_cfg.get("show_panel", True),
        )
        print(f"[{minio_folder}] Camera movement detection active ({len(move_alarm_cfg['rois'])} ROI)")

    write_camera_health(minio_folder, False)

    def _on_disconnect():
        write_camera_health(minio_folder, False)
        print(f"[{minio_folder}] Stream disconnected, sending alarm...")
        report.send_alarm(name="stream_kesintisi")

    def _on_reconnect():
        write_camera_health(minio_folder, True)
        print(f"[{minio_folder}] Stream reconnected.")

    mtt.start_cap_thread(
        source,
        resize=cam_cfg.get("frame_resize"),
        reconnect_delay_sec=reconnect_delay_sec,
        on_disconnect=_on_disconnect if gst_pipeline else None,
        on_reconnect=_on_reconnect  if gst_pipeline else None,
    )

    has_detection = bool(engine_path)
    if has_detection:
        mtt.start_detection_thread(engine_path, batch_size=cam_cfg.get("batch_size", batch_size), device=device,
                                   img_size=img_size, filter_classes=cam_cfg.get("detection_classes", None),
                                   detection_fps=cam_cfg.get("detection_fps", None))
        mtt.start_tracking_thread(tracker=tracker, max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    else:
        print(f"[{minio_folder}] engine_path empty - detection/tracking skipped")

    def shutdown(signum, _):
        print(f"\n[main] Signal received ({signum}), shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    trigger_alarms = [a for a in cam_cfg.get("alarms", []) if a.get("trigger_on")]

    _first_frame   = True
    _window_name   = f"OPET - {minio_folder}"
    _is_cuda_error = False

    try:
        _consecutive_empty  = 0
        _last_health_write  = 0

        while True:
            ret, frame = mtt.get_frame()
            if not ret or frame is None:
                _consecutive_empty += 1
                # Detection thread öldüyse 10 saniye sonra çık
                if has_detection and _consecutive_empty > 20:
                    od_alive = (mtt.od is not None and
                                mtt.od.thread is not None and
                                mtt.od.thread.is_alive())
                    if not od_alive:
                        print(f"[{minio_folder}] Detection thread died, shutting down...")
                        break
                time.sleep(0.5)
                continue
            _consecutive_empty = 0

            if time.time() - _last_health_write > 30:
                write_camera_health(minio_folder, True)
                _last_health_write = time.time()

            if _first_frame:
                fh, fw = frame.shape[:2]
                print(f"[{minio_folder}] Stream started: {fw}x{fh}")
                _first_frame = False

            if has_detection:
                bboxes, class_ids, scores, object_ids = mtt.get_tracking()
                class_names = mtt.od.classes
            else:
                bboxes, class_ids, scores, object_ids = [], [], [], []
                class_names = {}

            modules.update(bboxes, class_ids, scores, object_ids, frame, class_names)

            if move_detector is not None:
                move_info = move_detector.process(frame)

            frame = modules.draw(frame)

            try:
                sent = report.check_reports()
                data = modules.get_data()

                for r in _reports_cfg:
                    fields      = r.get("fields")
                    report_data = ({k: data[k] for k in fields if k in data} if fields else data)

                    if r.get("type") == "alarm":
                        if report.can_send_alarm(r["name"]):
                            minio_path = minio.upload_alert(frame, minio_folder, prefix=f"{r['name']}_")
                            report.send_alarm(name=r["name"], data=report_data, media_path=minio_path)
                    elif sent.get(r["name"]):
                        minio_path = minio.upload_report(frame, minio_folder)
                        report.send_report(name=r["name"], data=report_data, media_path=minio_path)

                for alarm in trigger_alarms:
                    if data.get(alarm["trigger_on"]) and report.can_send_alarm(alarm["name"]):
                        media_path = None
                        if alarm.get("with_snapshot", False):
                            media_path = minio.upload_alert(frame, minio_folder, prefix=f"{alarm['name']}_")
                        extra = {k: data[k] for k in alarm.get("include_fields", []) if k in data}
                        report.send_alarm(name=alarm["name"], media_path=media_path, data=extra if extra else None)

                if move_detector is not None:
                    frame = move_detector.draw(frame)
                    if move_info["triggered"] and report.can_send_alarm("camera_move"):
                        media_path = minio.upload_alert(frame, minio_folder, prefix="camera_move_")
                        report.send_alarm(name="camera_move", media_path=media_path)

            except Exception as report_err:
                print(f"⚠️ [NETWORK/API ERROR] Rapor veya alarm gonderilirken hata olustu ama kod devam ediyor: {report_err}")

            if display:
                cv2.imshow(_window_name, frame)
                if cv2.waitKey(1) == 27:
                    break

    except (KeyboardInterrupt, SystemExit):
        print(f"[{minio_folder}] Shutdown signal received.")
    except Exception as e:
        import traceback
        print(f"[{minio_folder}] Fatal error in main loop: {e}", flush=True)
        traceback.print_exc()
        _is_cuda_error = "CUDA" in str(e) or "AcceleratorError" in type(e).__name__
    finally:
        write_camera_health(minio_folder, False)
        modules.shutdown()
        def _force_exit():
            os._exit(1 if _is_cuda_error else 0)
        timer = threading.Timer(5.0, _force_exit)
        timer.daemon = True
        timer.start()
        mtt.release()
        timer.cancel()
        if _is_cuda_error:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            os._exit(1)


PING_URL               = "https://hc-ping.com/5bc3d95c-9398-4749-9905-09582f0b6120"
CAMERA_HEALTH_PING_URL = "https://hc-ping.com/f51650b4-3a42-45a3-8095-a898e40a3b46"
HEALTH_LOG             = "logs/camera_health.log"

def send_heartbeat_direct():
    try:
        import urllib.request
        urllib.request.urlopen(PING_URL, timeout=3)
        print("[HEALTHCHECK] Heartbeat basariyla gonderildi.")
    except Exception as e:
        print(f"[HEALTHCHECK] Ping gonderilemedi: {e}")


def write_camera_health(minio_folder: str, connected: bool):
    """Her kamera süreci kendi sağlık durumunu state/ altına yazar."""
    health_file = Path("state") / f"health_{minio_folder}.json"
    try:
        with open(health_file, "w") as f:
            json.dump({"connected": connected, "timestamp": time.time()}, f)
    except Exception:
        pass


def send_camera_health_ping(enabled_folders: list):
    """
    Tüm kameralar sağlıklıysa CAMERA_HEALTH_PING_URL'e ping atar,
    değilse atlamaz. Sonucu HEALTH_LOG'a kaydeder.
    max_age: HEARTBEAT_INTERVAL * 2 saniye içinde güncellenmemiş dosya stale sayılır.
    """
    max_age   = HEARTBEAT_INTERVAL * 2
    now       = time.time()
    unhealthy = []
    for folder in enabled_folders:
        health_file = Path("state") / f"health_{folder}.json"
        if not health_file.exists():
            unhealthy.append(f"{folder}:no_file")
            continue
        try:
            data = json.load(open(health_file))
            age  = now - data.get("timestamp", 0)
            if not data.get("connected", False):
                unhealthy.append(f"{folder}:disconnected")
            elif age > max_age:
                unhealthy.append(f"{folder}:stale({age:.0f}s)")
        except Exception as e:
            unhealthy.append(f"{folder}:read_error({e})")

    if not unhealthy:
        try:
            import urllib.request
            urllib.request.urlopen(CAMERA_HEALTH_PING_URL, timeout=3)
        except Exception as e:
            print(f"[CAMERA_HEALTH] Ping hatasi: {e}")
    else:
        ts   = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] SKIP — sagliksiz kameralar: {', '.join(unhealthy)}\n"
        Path("logs").mkdir(exist_ok=True)
        with open(HEALTH_LOG, "a", encoding="utf-8") as f:
            f.write(line)
        print(f"[CAMERA_HEALTH] {line.strip()}")


def startup_pings(enabled_folders: list):
    """
    main.py açılınca bir kez çalışır:
    1. Hemen genel ping atar.
    2. Tüm kameralar bağlanana kadar 5sn'de bir kontrol eder (max 120s).
    3. Bağlantı sağlanınca kamera health ping atar.
    """
    send_heartbeat_direct()

    deadline = time.time() + 120
    while time.time() < deadline:
        time.sleep(5)
        try:
            all_ok = all(
                json.load(open(Path("state") / f"health_{f}.json")).get("connected", False)
                for f in enabled_folders
                if (Path("state") / f"health_{f}.json").exists()
            ) and all(
                (Path("state") / f"health_{f}.json").exists()
                for f in enabled_folders
            )
        except Exception:
            all_ok = False

        if all_ok:
            try:
                import urllib.request
                urllib.request.urlopen(CAMERA_HEALTH_PING_URL, timeout=3)
                print("[CAMERA_HEALTH] Startup: tum kameralar baglandi, ping atildi.")
            except Exception as e:
                print(f"[CAMERA_HEALTH] Startup ping hatasi: {e}")
            return

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] STARTUP TIMEOUT — 120s icinde kameralar baglanamadi\n"
    Path("logs").mkdir(exist_ok=True)
    with open(HEALTH_LOG, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"[CAMERA_HEALTH] {line.strip()}")


if __name__ == "__main__":

    HEARTBEAT_INTERVAL = 60

    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--display", action="store_true", help="Show camera view in window")
    args = parser.parse_args()

    cfg = load_config()

    jobs = []
    for cam in cfg["cameras"]:
        if not cam.get("enabled", True):
            continue
        job = {
            "engine_path":  cam["engine_path"],
            "minio_folder": cam["minio_folder"],
            "display":      args.display,
        }
        if cam.get("gst_pipeline"):
            job["gst_pipeline"] = cam["gst_pipeline"]
        elif cam.get("rtsp_url"):
            job["gst_pipeline"] = build_gst_pipeline(
                rtsp_url=cam["rtsp_url"],
                framerate=cam.get("gst_framerate", 30),
                width=cam.get("gst_width"),
                height=cam.get("gst_height"),
                latency=cam.get("gst_latency", 300),
            )
        elif cam.get("video_path"):
            job["video_path"] = cam["video_path"]

        jobs.append(job)

    if len(jobs) == 1:
        _single_folders = [jobs[0]["minio_folder"]]

        threading.Thread(target=startup_pings, args=(_single_folders,), daemon=True).start()

        def heartbeat_thread():
            time.sleep(HEARTBEAT_INTERVAL)
            while True:
                send_heartbeat_direct()
                send_camera_health_ping(_single_folders)
                time.sleep(HEARTBEAT_INTERVAL)

        hb = threading.Thread(target=heartbeat_thread, daemon=True)
        hb.start()

        track_video(**jobs[0])

    else:
        _shutdown = multiprocessing.Event()

        def _on_sigint(signum, _):
            _shutdown.set()

        signal.signal(signal.SIGINT,  _on_sigint)
        signal.signal(signal.SIGTERM, _on_sigint)

        _MAX_RESTARTS           = 10
        _RESTART_DELAY          = 15
        _all_folders            = [j["minio_folder"] for j in jobs]
        last_heartbeat_time     = time.time()
        last_camera_health_time = time.time()

        procs = {}
        for job in jobs:
            p = multiprocessing.Process(target=track_video, kwargs=job, daemon=False)
            p.start()
            procs[job["minio_folder"]] = {"process": p, "job": job, "restarts": 0}

        threading.Thread(target=startup_pings, args=(_all_folders,), daemon=True).start()

        try:
            while not _shutdown.is_set():
                if time.time() - last_heartbeat_time >= HEARTBEAT_INTERVAL:
                    send_heartbeat_direct()
                    last_heartbeat_time = time.time()

                if time.time() - last_camera_health_time >= HEARTBEAT_INTERVAL:
                    send_camera_health_ping(_all_folders)
                    last_camera_health_time = time.time()

                for name, entry in list(procs.items()):
                    p = entry["process"]
                    if p.is_alive():
                        continue

                    if entry["restarts"] < _MAX_RESTARTS:
                        print(f"[supervisor] {name} düştü (Exit: {p.exitcode}). Yeniden başlatılıyor...")
                        time.sleep(_RESTART_DELAY)
                        new_p = multiprocessing.Process(target=track_video, kwargs=entry["job"], daemon=False)
                        new_p.start()
                        entry["process"]  = new_p
                        entry["restarts"] += 1
                    else:
                        print(f"[supervisor] {name} maksimum yeniden başlatma sınırına ulaştı.")
                        del procs[name]

                if not procs:
                    print("[supervisor] Tüm süreçler durdu.")
                    break

                time.sleep(1)

        except KeyboardInterrupt:
            _shutdown.set()
        finally:
            print("[supervisor] Kapatılıyor...")
            for entry in procs.values():
                if entry["process"].is_alive():
                    entry["process"].terminate()
            print("[main] Tüm kamera süreçleri güvenle kapatıldı.")