import os
from ultralytics import YOLO

checkpoint = "runs/detect/train-3/weights/last.pt"

if os.path.exists(checkpoint):
    model = YOLO(checkpoint)
    model.train(resume=True)
else:
    model = YOLO("yolov8n.pt")
    model.train(
        data="roboflow_dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16
    )