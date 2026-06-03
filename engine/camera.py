import cv2

"""
Used for webcam and RTSP video streams.
"""

class Cap:
    def __init__(self, source=0):
        """
        source: int for local camera (0, 1..), str for RTSP stream
               e.g. "rtsp://admin:password@XxX.xXx.x.xXx:xXx/stream"
        """
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            print(f"Error: Could not connect to source {source}.")

    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                print("Frame could not be read (stream may be disconnected).")
                return None
        return None

    def release(self):
        self.cap.release()

# Example usage:
# camera = Cap(0)
# rtsp_camera = Cap("rtsp://user:password@ip_address:port/channel")