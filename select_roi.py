"""
ROI Drawing Tool
Usage:
    python select_roi.py                                    # polygon modu, config'deki ilk kamera
    python select_roi.py --cam-name ingiltere-kamera-2      # polygon modu, belirli kamera
    python select_roi.py --mode camera_move                 # camera_move anchor noktaları
    python select_roi.py --video test.mp4                   # video dosyası
    python select_roi.py --camera 0                         # webcam

Modlar:
    polygon      (varsayılan) → Kapalı alan / bölge çizer. zone_points / polygon formatı çıktı verir.
    camera_move              → Bağımsız anchor noktaları seçer. camera_move_alarm rois formatı çıktı verir.

Kontroller:
    Sol tık      → Nokta ekle
    Sağ tık      → Son noktayı sil
    Enter / Space → Bitir ve koordinatları yazdır
    R            → Sıfırla
    Q / Esc      → Çık
"""

import cv2
import numpy as np
import json
import argparse

# camera_move modunda her anchor noktanın görsel ROI boyutu (piksel)
CAMERA_MOVE_ROI_SIZE = 60

points        = []
frame_display = None
frame_clean   = None
_mode         = "polygon"   # global, main() tarafından set edilir


def mouse_callback(event, x, y, flags, param):
    global points, frame_display, frame_clean
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        redraw()
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()
            redraw()
    elif event == cv2.EVENT_MOUSEMOVE:
        tmp = frame_display.copy()
        if _mode == "polygon" and points:
            cv2.line(tmp, points[-1], (x, y), (200, 200, 200), 1)
        elif _mode == "camera_move":
            half = CAMERA_MOVE_ROI_SIZE // 2
            cv2.rectangle(tmp, (x - half, y - half), (x + half, y + half), (200, 200, 200), 1)
        cv2.imshow("ROI Draw", tmp)


def redraw():
    global frame_display, frame_clean
    frame_display = frame_clean.copy()
    h, w = frame_display.shape[:2]

    if _mode == "polygon":
        if len(points) > 1:
            cv2.polylines(frame_display, [np.array(points, np.int32)], False, (0, 255, 0), 2)
        for i, p in enumerate(points):
            cv2.circle(frame_display, p, 5, (0, 0, 255), -1)
            cv2.putText(frame_display, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        hint = "Sol:Ekle  Sag:Sil  Enter:Bitir  R:Sifirla"

    else:  # camera_move
        half = CAMERA_MOVE_ROI_SIZE // 2
        for i, p in enumerate(points):
            cv2.rectangle(frame_display,
                          (p[0] - half, p[1] - half),
                          (p[0] + half, p[1] + half),
                          (0, 200, 80), 2)
            cv2.circle(frame_display, p, 4, (0, 0, 255), -1)
            cv2.putText(frame_display, f"R{i + 1}", (p[0] + half // 2, p[1] - half // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        hint = "[camera_move] Sol:Anchor ekle  Sag:Son sil  Enter:Bitir  R:Sifirla"

    cv2.putText(frame_display, f"Noktalar: {len(points)} | {hint}",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("ROI Draw", frame_display)


def print_result(frame_shape):
    h, w = frame_shape[:2]
    print("\n" + "=" * 55)
    print(f"Goruntu boyutu: {w}x{h}")
    print(f"Nokta sayisi  : {len(points)}")

    pts_json = json.dumps([list(p) for p in points], indent=8)

    if _mode == "polygon":
        print("\n--- polygon / zone_points formati (kamera JSON) ---")
        print(f'"zone_points": {pts_json}')
        print(f'"polygon":     {pts_json}')
    else:
        print("\n--- camera_move_alarm rois formati (kamera JSON) ---")
        print(f'"rois": {pts_json}')

    print("=" * 55 + "\n")


def get_rtsp_from_config(camera_name):
    from pathlib import Path
    cameras_dir = Path("cameras")
    all_cams = []

    if cameras_dir.is_dir():
        for cam_file in sorted(cameras_dir.glob("*.json")):
            try:
                with open(cam_file, encoding="utf-8") as f:
                    all_cams.append(json.load(f))
            except Exception:
                pass

    for cam in all_cams:
        if cam.get("minio_folder") == camera_name:
            url = (cam.get("rtsp_url") or
                   cam.get("gst_pipeline") or
                   cam.get("video_path"))
            frame_resize = float(cam.get("frame_resize", 1.0))
            return url, frame_resize

    print(f"Kamera bulunamadi: '{camera_name}'")
    if all_cams:
        names = [c.get("minio_folder", "?") for c in all_cams]
        print(f"Mevcut kameralar: {', '.join(names)}")
    return None, 1.0


def main():
    global frame_display, frame_clean, points, _mode

    parser = argparse.ArgumentParser()
    parser.add_argument("--video",    type=str, default=None)
    parser.add_argument("--camera",   type=int, default=None)
    parser.add_argument("--cam-name", type=str, default=None)
    parser.add_argument("--frame",    type=int, default=30, help="Kacinci kare alinsin")
    parser.add_argument("--mode",     type=str, default="polygon",
                        choices=["polygon", "camera_move"],
                        help="polygon: alan/bolge ciz | camera_move: anchor noktasi sec")
    args = parser.parse_args()

    _mode = args.mode

    frame_resize = 1.0
    source = None
    if args.video:
        source = args.video
    elif args.camera is not None:
        source = args.camera
    else:
        cam_name = args.cam_name
        if cam_name is None:
            from pathlib import Path
            cams = sorted(Path("cameras").glob("*.json")) if Path("cameras").is_dir() else []
            if cams:
                with open(cams[0], encoding="utf-8") as f:
                    cam_name = json.load(f).get("minio_folder", "Kamera-5")
            else:
                cam_name = "Kamera-5"
        source, frame_resize = get_rtsp_from_config(cam_name)
        if source:
            print(f"{cam_name} RTSP: {source[:60]}...")
            print(f"frame_resize: {frame_resize}")
        else:
            print("Kaynak bulunamadi. --video veya --camera belirtin.")
            return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Kaynak acilamadi: {source}")
        return

    for _ in range(args.frame):
        ret, frame = cap.read()
        if not ret:
            break
    cap.release()

    if not ret or frame is None:
        print("Kare okunamadi.")
        return

    # Frame'i pipeline ile aynı boyuta getir
    if frame_resize != 1.0:
        frame = cv2.resize(frame, None, fx=frame_resize, fy=frame_resize,
                           interpolation=cv2.INTER_AREA)

    h, w = frame.shape[:2]
    print(f"Goruntu boyutu (pipeline): {w}x{h}")
    print(f"Mod: {_mode}")

    frame_clean   = frame.copy()
    frame_display = frame.copy()

    cv2.namedWindow("ROI Draw", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI Draw", min(w, 1280), min(h, 720))
    cv2.setMouseCallback("ROI Draw", mouse_callback)
    redraw()

    while True:
        key = cv2.waitKey(50) & 0xFF

        if key in (13, 32):  # Enter veya Space
            min_pts = 3 if _mode == "polygon" else 1
            if len(points) >= min_pts:
                print_result(frame.shape)

                # Sonucu göster
                frame_display = frame_clean.copy()
                if _mode == "polygon":
                    cv2.fillPoly(frame_display, [np.array(points, np.int32)], (0, 255, 0))
                    cv2.addWeighted(frame_display, 0.3, frame_clean, 0.7, 0, frame_display)
                    cv2.polylines(frame_display, [np.array(points, np.int32)], True, (0, 255, 0), 2)
                    for i, p in enumerate(points):
                        cv2.circle(frame_display, p, 5, (0, 0, 255), -1)
                        cv2.putText(frame_display, str(i + 1), (p[0] + 8, p[1] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    msg = "Polygon tamamlandi! Q=Cik R=Sifirla"
                else:
                    half = CAMERA_MOVE_ROI_SIZE // 2
                    for i, p in enumerate(points):
                        cv2.rectangle(frame_display,
                                      (p[0] - half, p[1] - half),
                                      (p[0] + half, p[1] + half),
                                      (0, 200, 80), 2)
                        cv2.circle(frame_display, p, 4, (0, 0, 255), -1)
                        cv2.putText(frame_display, f"R{i + 1}",
                                    (p[0] + half // 2, p[1] - half // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    msg = f"{len(points)} anchor noktasi secildi! Q=Cik R=Sifirla"

                cv2.putText(frame_display, msg,
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                cv2.imshow("ROI Draw", frame_display)
            else:
                needed = "3 nokta" if _mode == "polygon" else "en az 1 nokta"
                print(f"En az {needed} gerekli.")

        elif key in (ord('r'), ord('R')):
            points = []
            redraw()

        elif key in (ord('q'), ord('Q'), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
