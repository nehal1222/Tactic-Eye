from ultralytics import YOLO

# Load your trained model
model = YOLO("models/best.pt")

# Predict on an image
model.predict(
    source="roboflow_dataset/valid/images/08fd33_3_1_png.rf.6f25c835bf6d1828dcf584e5969b1f58.jpg",
    save=True,
    conf=0.4
)