import json
import subprocess

# Sanal ortamdaki ffmpeg/ffprobe binary'lerini PATH'e ekler
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass   # static_ffmpeg yoksa sistem PATH'indeki ffprobe denenir

CODEC_GST = {
    "h264": ("rtph264depay", "h264parse", "nvh264dec"),
    "hevc": ("rtph265depay", "h265parse", "nvh265dec"),
    "h265": ("rtph265depay", "h265parse", "nvh265dec"),
}


def detect_rtsp_codec(rtsp_url: str, timeout: int = 10) -> str:
    """Returns the video codec of an RTSP stream using ffprobe ('h264' or 'hevc')."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-rtsp_transport", "tcp",
        "-print_format", "json",
        "-show_streams",
        rtsp_url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        data = json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffprobe timeout ({timeout}s): {rtsp_url}")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise RuntimeError(f"ffprobe could not run: {e}")

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            codec = stream["codec_name"].lower()
            if codec not in CODEC_GST:
                raise RuntimeError(f"Unsupported codec: {codec}")
            return codec

    raise RuntimeError(f"No video stream found in RTSP: {rtsp_url}")


def build_gst_pipeline(rtsp_url: str, framerate: int = 30,
                       width: int = None, height: int = None,
                       latency: int = 300,
                       timeout: int = 3000000,
                       tcp_timeout: int = 3000000) -> str:
    """Detects codec for rtsp_url and builds a GStreamer pipeline string."""
    codec = detect_rtsp_codec(rtsp_url)
    depay, parse, dec = CODEC_GST[codec]
    print(f"[rtsp_probe] {rtsp_url.split('@')[-1]} → {codec.upper()} detected")

    caps = f"video/x-raw,framerate={framerate}/1"
    scale = f"videoscale ! video/x-raw,width={width},height={height} ! " if (width and height) else ""

    return (
        f"rtspsrc location={rtsp_url} latency={latency} protocols=tcp "
        f"timeout={timeout} tcp-timeout={tcp_timeout} ! "
        f"{depay} ! {parse} ! {dec} ! "
        f"videorate ! {caps} ! "
        f"{scale}"
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"queue max-size-buffers=1 leaky=2 ! "
        f"appsink drop=true sync=false max-buffers=1"
    )
