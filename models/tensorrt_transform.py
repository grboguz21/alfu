# # from ultralytics import YOLO
# # import ultralytics.utils.checks as checks

# # checks.check_requirements = lambda *args, **kwargs: None

# # model = YOLO("yolo26l.pt") # forklift_best.pt
# # model.export(
# #     format="engine",
# #     imgsz=640,        # max size
# #     half=True,
# #     device=0,
# #     dynamic=True,     # dynamic shape aktif
# #     name="yolo26l_dynamic"
# # )



# ---

from ultralytics import YOLO
import ultralytics.utils.checks as checks

checks.check_requirements = lambda *args, **kwargs: None

model = YOLO("yolo26l-pose.pt")
model.export(
    format="engine",
    imgsz=640,
    half=True,
    device=0,
    dynamic=True,
    name="yolo26l-pose_dynamic"
)