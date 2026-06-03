import cv2
from datetime import datetime

"""
Used to display FPS on screen.
"""


class FPSViewer:
    def __init__(self, fps_average_n_frames=100):

        self.text_size = 0.8
        self.text_thickness = 2  
        self.font = cv2.FONT_HERSHEY_SIMPLEX


        self.frame_times = []
        self.average_fps = 0
        self.frame_count = 0

        self.fps_average_n_frames = fps_average_n_frames

        self.waiting_times = []
        self.average_elapsed_time = 0

    def update_fps(self):
        self.frame_count += 1
        current_time = datetime.now()
        self.frame_times.append(current_time)

        # Calculate FPS based on the last n_of frames
        if len(self.frame_times) > self.fps_average_n_frames:
            elapsed_time = self.frame_times[-1] - self.frame_times[0]
            self.average_fps = round(self.fps_average_n_frames / elapsed_time.total_seconds())

            # Remove the oldest frame time
            self.frame_times.pop(0)
        return round(self.average_fps, 1)

    # visualize
    def show_label(self, frame, x, y, label_text, value, text_size=0.6):

        full_text = f"{label_text}: {value}"
        font = cv2.FONT_HERSHEY_DUPLEX
        thickness = 1
        
        (text_w, text_h), _ = cv2.getTextSize(full_text, font, text_size, thickness)

        overlay = frame.copy()
        padding = 12
        p1 = (x, y - text_h - padding)
        p2 = (x + text_w + (padding * 2), y + padding)
        cv2.rectangle(overlay, p1, p2, (10, 10, 10), -1)
        
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, full_text, (x + padding + 1, y - 1), 
                    font, text_size, (0, 0, 0), thickness + 1, cv2.LINE_AA)

        cv2.putText(frame, full_text, (x + padding, y - 2), 
                    font, text_size, (255, 255, 255), thickness, cv2.LINE_AA)
        
        return frame

    # show fps on the screen
    def show_fps(self, frame):
        self.update_fps()

        self.show_label(frame, 20, 40, "FPS", self.average_fps)
        return frame