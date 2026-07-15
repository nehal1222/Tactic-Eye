from ultralytics import YOLO
from trackers import ByteTrackTracker
import cv2
import supervision as sv
import numpy as np
from sklearn.cluster import KMeans

from pitch_config import SoccerPitchConfiguration
from view_transform import ViewTransformer
from pitch_draw import draw_pitch, draw_points_on_pitch

VIDEO_PATH = "sports/examples/soccer/data/0bfacc_0.mp4"
WARMUP_FRAMES = 30
MIN_KEYPOINTS_FOR_HOMOGRAPHY = 6  # too few points gives a wildly inaccurate fit
KEYPOINT_CONFIDENCE_THRESHOLD = 0.5  # low-confidence keypoints still get coordinates, just wrong ones
CONFIG = SoccerPitchConfiguration()
PITCH_VERTICES = np.array(CONFIG.vertices, dtype=np.float32)

player_model = YOLO("models/best.pt")
pitch_model = YOLO("sports/examples/soccer/data/football-pitch-detection.pt")
PLAYER_CLASS_ID = list(player_model.names.values()).index("player")
BALL_CLASS_ID = list(player_model.names.values()).index("ball")

TEAM_COLORS = {0: sv.Color.RED, 1: sv.Color.BLUE}
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
    results = player_model.predict(frame, conf=0.4, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])
    for box in detections.xyxy[detections.class_id == PLAYER_CLASS_ID]:
        warmup_colors.append(get_jersey_color(frame, box))
cap.release()

kmeans = KMeans(n_clusters=2, n_init=10).fit(warmup_colors)

# --- Pass 2: detect pitch keypoints + players, transform to top-down radar ---
cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter("outputs/radar.mp4", fourcc, fps, (w, h))

# thresholds must be <= our YOLO conf filter (0.4), otherwise ByteTrack treats
# every detection as "low confidence" and never activates a track (permanent -1 id)
tracker = ByteTrackTracker(track_activation_threshold=0.3, high_conf_det_threshold=0.3)
box_annotator = sv.BoxAnnotator(color=TEAM_PALETTE)
label_annotator = sv.LabelAnnotator(color=TEAM_PALETTE)

last_transformer = None

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    pitch_result = pitch_model.predict(frame, verbose=False)[0]
    keypoints = sv.KeyPoints.from_ultralytics(pitch_result)

    results = player_model.predict(frame, conf=0.4, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])
    detections = tracker.update(detections)

    team_by_index = {}
    labels = []
    color_lookup = np.full(len(detections), 2, dtype=int)
    for i, (tid, cls, box) in enumerate(
        zip(detections.tracker_id, detections.class_id, detections.xyxy)
    ):
        name = player_model.names[cls]
        if cls == PLAYER_CLASS_ID:
            team = kmeans.predict([get_jersey_color(frame, box)])[0]
            team_by_index[i] = team
            color_lookup[i] = team
            name = f"Team {'A' if team == 0 else 'B'}"
        elif cls == BALL_CLASS_ID:
            color_lookup[i] = 3
        labels.append(f"#{tid} {name}")

    annotated = box_annotator.annotate(frame.copy(), detections, custom_color_lookup=color_lookup)
    annotated = label_annotator.annotate(annotated, detections, labels, custom_color_lookup=color_lookup)

    if keypoints.xy is not None and len(keypoints.xy) > 0:
        mask = (
            (keypoints.xy[0][:, 0] > 1)
            & (keypoints.xy[0][:, 1] > 1)
            & (keypoints.keypoint_confidence[0] > KEYPOINT_CONFIDENCE_THRESHOLD)
        )
        if mask.sum() >= MIN_KEYPOINTS_FOR_HOMOGRAPHY:
            last_transformer = ViewTransformer(
                source=keypoints.xy[0][mask].astype(np.float32),
                target=PITCH_VERTICES[mask],
            )

    if last_transformer is not None:
        anchors = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        pitch_xy = last_transformer.transform_points(anchors)

        radar = draw_pitch(CONFIG)
        for i, xy in enumerate(pitch_xy):
            team = team_by_index.get(i)
            color = TEAM_COLORS.get(team, sv.Color(255, 215, 0))
            radar = draw_points_on_pitch(
                CONFIG, xy[np.newaxis, :], face_color=color, pitch=radar
            )

        radar = sv.resize_image(radar, (w // 3, h // 3))
        rh, rw = radar.shape[:2]
        annotated[h - rh :, 0:rw] = radar

    writer.write(annotated)

cap.release()
writer.release()
