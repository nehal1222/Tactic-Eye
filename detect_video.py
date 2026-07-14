from ultralytics import YOLO
from trackers import ByteTrackTracker
import cv2
import supervision as sv
import numpy as np
from sklearn.cluster import KMeans

VIDEO_PATH = "sports/examples/soccer/data/0bfacc_0.mp4"
WARMUP_FRAMES = 30

model = YOLO("models/best.pt")
PLAYER_CLASS_ID = list(model.names.values()).index("player")
BALL_CLASS_ID = list(model.names.values()).index("ball")

# index 0 = Team A, index 1 = Team B, index 2 = referee/goalkeeper, index 3 = ball
TEAM_PALETTE = sv.ColorPalette(
    colors=[sv.Color.RED, sv.Color.BLUE, sv.Color(255, 215, 0), sv.Color.BLACK]
)


def get_jersey_color(frame, box):
    x1, y1, x2, y2 = box.astype(int)
    bw, bh = x2 - x1, y2 - y1
    # center chest region: avoids head, arms, shorts, and the grass margin around the box
    cx1, cx2 = x1 + int(bw * 0.25), x1 + int(bw * 0.75)
    cy1, cy2 = y1 + int(bh * 0.2), y1 + int(bh * 0.55)
    torso = frame[cy1:cy2, cx1:cx2]
    if torso.size == 0:
        torso = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    green_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 40)
    non_green = torso[~green_mask]
    if len(non_green) < 10:  # crop was almost entirely grass, fall back to full crop
        non_green = torso.reshape(-1, 3)
    return non_green.reshape(-1, 3).mean(axis=0)


# --- Pass 1: fit team-color clusters on the first frames ---
cap = cv2.VideoCapture(VIDEO_PATH)
warmup_colors = []
for _ in range(WARMUP_FRAMES):
    ok, frame = cap.read()
    if not ok:
        break
    results = model.predict(frame, conf=0.4, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])
    for box in detections.xyxy[detections.class_id == PLAYER_CLASS_ID]:
        warmup_colors.append(get_jersey_color(frame, box))
cap.release()

kmeans = KMeans(n_clusters=2, n_init=10).fit(warmup_colors)

# --- Pass 2: run detection + tracking + team labeling over the full video ---
cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter("outputs/detected.mp4", fourcc, fps, (w, h))

# thresholds must be <= our YOLO conf filter (0.4), otherwise ByteTrack treats
# every detection as "low confidence" and never activates a track (permanent -1 id)
tracker = ByteTrackTracker(track_activation_threshold=0.3, high_conf_det_threshold=0.3)
box_annotator = sv.BoxAnnotator(color=TEAM_PALETTE)
label_annotator = sv.LabelAnnotator(color=TEAM_PALETTE)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    results = model.predict(frame, conf=0.4, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])
    detections = tracker.update(detections)

    labels = []
    color_lookup = np.full(len(detections), 2, dtype=int)
    for i, (tid, cls, box) in enumerate(
        zip(detections.tracker_id, detections.class_id, detections.xyxy)
    ):
        name = model.names[cls]
        if cls == PLAYER_CLASS_ID:
            team = kmeans.predict([get_jersey_color(frame, box)])[0]
            color_lookup[i] = team
            name = f"Team {'A' if team == 0 else 'B'}"
        elif cls == BALL_CLASS_ID:
            color_lookup[i] = 3
        labels.append(f"#{tid} {name}")

    annotated = box_annotator.annotate(frame.copy(), detections, custom_color_lookup=color_lookup)
    annotated = label_annotator.annotate(annotated, detections, labels, custom_color_lookup=color_lookup)
    writer.write(annotated)

cap.release()
writer.release()
