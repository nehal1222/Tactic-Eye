import base64
import os
import tempfile

import cv2
import gdown
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import stripe
import streamlit as st
import supervision as sv
from sklearn.cluster import KMeans
from trackers import ByteTrackTracker
from ultralytics import YOLO

import db
from pitch_draw import draw_pitch, draw_points_on_pitch
from view_transform import ViewTransformer
from pitch_config import SoccerPitchConfiguration

# Validated categorical/status palette (checked with dataviz skill's validate_palette.js:
# all CVD/contrast checks pass for both light and dark surfaces)
TEAM_A_COLOR = "#e34948"   # categorical red
TEAM_B_COLOR = "#2a78d6"   # categorical blue
CHART_INK = "#52514e"
CHART_GRIDLINE = "#e1e0d9"

PAGE_CSS = """
<style>
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb;
  --page-plane: #f9f9f7;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --border: rgba(11,11,11,0.10);
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --page-plane: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --border: rgba(255,255,255,0.10);
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19;
  --page-plane: #0d0d0d;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --border: rgba(255,255,255,0.10);
}
@keyframes heroShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
.hero-banner {
  position: relative;
  overflow: hidden;
  padding: 28px 32px;
  border-radius: 14px;
  margin-bottom: 8px;
  background: linear-gradient(120deg, #2a78d6, #184f95, #2a78d6, #3987e5);
  background-size: 300% 300%;
  animation: heroShift 12s ease-in-out infinite, fadeInUp 0.5s ease-out;
  color: white;
}
.hero-banner::before {
  content: "";
  position: absolute;
  inset: 0;
  opacity: 0.5;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.35'/%3E%3C/svg%3E");
  mix-blend-mode: overlay;
  pointer-events: none;
}
.hero-banner h1, .hero-banner p, .hero-banner .logo-row { position: relative; }
.logo-row { display: flex; align-items: center; gap: 10px; }
.hero-banner h1 { margin: 0; font-size: 1.9rem; font-weight: 700; }
.hero-banner p { margin: 6px 0 0 0; opacity: 0.9; font-size: 0.95rem; }
.legend-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px; border-radius: 999px; margin-right: 8px;
  font-size: 0.85rem; border: 1px solid var(--border);
  animation: fadeInUp 0.4s ease-out;
}
.legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

/* Streamlit widget polish */
div[data-testid="stExpander"] {
  border-radius: 12px;
  border: 1px solid var(--border);
  transition: box-shadow 0.2s ease, transform 0.2s ease;
}
div[data-testid="stExpander"]:hover {
  box-shadow: 0 6px 20px rgba(0,0,0,0.08);
}
.stButton button, .stDownloadButton button {
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton button:hover, .stDownloadButton button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 14px rgba(42,120,214,0.25);
}
div[data-testid="stMetric"] {
  animation: fadeInUp 0.4s ease-out;
}
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
  border-radius: 8px 8px 0 0;
  padding: 8px 16px;
}
.apple-section {
  padding: 56px 24px;
  text-align: center;
  border-bottom: 1px solid var(--border);
  animation: fadeInUp 0.5s ease-out;
}
.apple-section:last-child { border-bottom: none; }
.apple-section .eyebrow {
  font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: #2a78d6; margin-bottom: 10px;
}
.apple-section h2 {
  font-size: clamp(1.8rem, 4vw, 2.6rem);
  font-weight: 700;
  margin: 0 0 14px 0;
  color: var(--text-primary);
}
.apple-section p {
  font-size: 1.05rem;
  color: var(--text-secondary);
  max-width: 640px;
  margin: 0 auto;
  line-height: 1.65;
}
.stat-row {
  display: flex; justify-content: center; gap: 48px; flex-wrap: wrap;
  margin: 40px 0 8px 0;
}
.stat-row .stat-value {
  font-size: 2.4rem; font-weight: 700; color: var(--text-primary);
}
.stat-row .stat-label {
  font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px;
}
.photo-hero {
  position: relative;
  height: 380px;
  border-radius: 14px;
  overflow: hidden;
  margin-bottom: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  animation: fadeInUp 0.5s ease-out;
}
.photo-hero::before {
  content: "";
  position: absolute;
  inset: 0;
  background-image: var(--hero-img);
  background-size: cover;
  background-position: center;
  filter: brightness(0.55);
}
.photo-hero .photo-hero-content { position: relative; color: white; padding: 0 24px; }
.photo-hero .photo-hero-content .eyebrow { color: #cde2fb; }
.photo-hero h1 {
  font-size: clamp(2.2rem, 5vw, 3.4rem);
  font-weight: 700;
  margin: 8px 0 0 0;
  text-shadow: 0 2px 12px rgba(0,0,0,0.35);
}
.photo-hero p {
  font-size: 1.1rem;
  margin: 12px 0 0 0;
  opacity: 0.95;
  max-width: 560px;
}
</style>
"""

CONTACT_EMAIL = "inehalsinha@gmail.com"

PIPELINE_FEATURES = [
    ("Custom Object Detection", "YOLOv8 trained from scratch (100 epochs) on a Roboflow dataset — detects players, goalkeepers, referees, and the ball. Not a pretrained/off-the-shelf model."),
    ("Multi-Object Tracking", "ByteTrack assigns each detection a persistent ID across frames, so players are tracked through the clip rather than re-detected independently every frame."),
    ("Team Classification", "Unsupervised KMeans clustering on jersey color (with HSV-based grass exclusion) automatically splits players into two teams — no manual labeling."),
    ("Tactical Radar", "A separate pitch-keypoint model drives a live homography, projecting every player's position onto a flat, top-down tactical view in real time."),
]

VIDEO_SOURCES = [
    ("Roboflow Universe", "https://universe.roboflow.com/", "Search \"football\" or \"soccer\" for open datasets with match video/images (this project's training data came from here)."),
    ("Kaggle: DFL Bundesliga Data Shootout", "https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout", "The original match footage source behind the sample clips bundled with this app."),
    ("roboflow/sports (GitHub)", "https://github.com/roboflow/sports", "The repo this project's homography/radar code builds on — its setup script links directly to more sample clips."),
]

MODEL_METRICS = pd.DataFrame([
    {"Class": "All (overall)", "Precision": 0.857, "Recall": 0.774, "mAP50": 0.774},
    {"Class": "Player", "Precision": 0.882, "Recall": 0.961, "mAP50": 0.974},
    {"Class": "Goalkeeper", "Precision": 0.868, "Recall": 1.000, "mAP50": 0.972},
    {"Class": "Referee", "Precision": 0.679, "Recall": 0.875, "mAP50": 0.845},
    {"Class": "Ball", "Precision": 1.000, "Recall": 0.260, "mAP50": 0.303},
])

HERO_IMAGE_PATH = "docs/hero_bg.jpg"
PLAYER_MODEL_PATH = "models/best.pt"
PITCH_MODEL_PATH = "sports/examples/soccer/data/football-pitch-detection.pt"
SAMPLE_VIDEOS = {
    "0bfacc_0.mp4": "sports/examples/soccer/data/0bfacc_0.mp4",
    "2e57b9_0.mp4": "sports/examples/soccer/data/2e57b9_0.mp4",
    "08fd33_0.mp4": "sports/examples/soccer/data/08fd33_0.mp4",
    "573e61_0.mp4": "sports/examples/soccer/data/573e61_0.mp4",
    "121364_0.mp4": "sports/examples/soccer/data/121364_0.mp4",
}
# Google Drive file IDs for assets too large to commit to git — downloaded on
# demand so a fresh clone/deploy (e.g. Streamlit Community Cloud) works without
# manual setup steps.
GDOWN_ASSET_IDS = {
    PITCH_MODEL_PATH: "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf",
    "sports/examples/soccer/data/0bfacc_0.mp4": "12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF",
    "sports/examples/soccer/data/2e57b9_0.mp4": "19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf",
    "sports/examples/soccer/data/08fd33_0.mp4": "1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-",
    "sports/examples/soccer/data/573e61_0.mp4": "1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU",
    "sports/examples/soccer/data/121364_0.mp4": "1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu",
}
WARMUP_FRAMES = 30
MIN_KEYPOINTS_FOR_HOMOGRAPHY = 6
KEYPOINT_CONFIDENCE_THRESHOLD = 0.5
PAYMENTS_ENABLED = False  # Stripe integration is built but on hold — see dashboard.py history
SINGLE_TIER_MAX_FRAMES = 900
FREE_TIER_MAX_FRAMES = 150
PREMIUM_TIER_MAX_FRAMES = 1800
PREMIUM_PRICE_USD_CENTS = 499

CONFIG = SoccerPitchConfiguration()
PITCH_VERTICES = np.array(CONFIG.vertices, dtype=np.float32)
TEAM_COLORS = {0: sv.Color.RED, 1: sv.Color.BLUE}
# index 0 = Team A, index 1 = Team B, index 2 = referee/goalkeeper, index 3 = ball
TEAM_PALETTE = sv.ColorPalette(
    colors=[sv.Color.RED, sv.Color.BLUE, sv.Color(255, 215, 0), sv.Color.BLACK]
)

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY")
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

db.init_db()

st.set_page_config(page_title="TacticEye", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="viz-root">
      <div class="hero-banner">
        <div class="logo-row">
          <svg width="34" height="34" viewBox="0 0 36 36" aria-hidden="true">
            <circle cx="18" cy="18" r="16" fill="none" stroke="white" stroke-width="2" opacity="0.9"/>
            <circle cx="18" cy="18" r="9" fill="white"/>
            <circle cx="18" cy="18" r="4" fill="#2a78d6"/>
          </svg>
          <h1>TacticEye</h1>
        </div>
        <p>Custom-trained YOLOv8 detection · ByteTrack · team classification · homography-based tactical radar</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def ensure_asset(path):
    """Downloads a large asset from Google Drive on first use if it's not already
    on disk — lets a fresh clone/deploy work without manual gdown commands."""
    if os.path.exists(path):
        return
    file_id = GDOWN_ASSET_IDS.get(path)
    if file_id is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with st.spinner(f"Downloading {os.path.basename(path)} (one-time setup)..."):
        gdown.download(id=file_id, output=path, quiet=False)


@st.cache_resource
def load_models():
    ensure_asset(PITCH_MODEL_PATH)
    return YOLO(PLAYER_MODEL_PATH), YOLO(PITCH_MODEL_PATH)


def _style_fig(fig, height=280):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CHART_INK, size=13),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor=CHART_GRIDLINE)
    fig.update_yaxes(showgrid=True, gridcolor=CHART_GRIDLINE, zeroline=False)
    return fig


def get_jersey_color(frame, box):
    x1, y1, x2, y2 = box.astype(int)
    bw, bh = x2 - x1, y2 - y1
    cx1, cx2 = x1 + int(bw * 0.25), x1 + int(bw * 0.75)
    cy1, cy2 = y1 + int(bh * 0.2), y1 + int(bh * 0.55)
    torso = frame[cy1:cy2, cx1:cx2]
    if torso.size == 0:
        torso = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    green_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 40)
    non_green = torso[~green_mask]
    if len(non_green) < 10:
        non_green = torso.reshape(-1, 3)
    return non_green.reshape(-1, 3).mean(axis=0)


def process_video(video_path, max_frames):
    """Yields the output path first, then (frame_index, annotated_frame_bgr) per frame."""
    player_model, pitch_model = load_models()
    player_class_id = list(player_model.names.values()).index("player")
    ball_class_id = list(player_model.names.values()).index("ball")

    # --- fit team-color clusters on the first frames ---
    cap = cv2.VideoCapture(video_path)
    warmup_colors = []
    for _ in range(min(WARMUP_FRAMES, max_frames)):
        ok, frame = cap.read()
        if not ok:
            break
        results = player_model.predict(frame, conf=0.4, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        for box in detections.xyxy[detections.class_id == player_class_id]:
            warmup_colors.append(get_jersey_color(frame, box))
    cap.release()
    kmeans = KMeans(n_clusters=2, n_init=10).fit(warmup_colors)

    # --- main pass: detect, track, assign teams, compute homography, draw radar ---
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    yield out_path  # caller grabs this first, before consuming frames

    tracker = ByteTrackTracker(track_activation_threshold=0.3, high_conf_det_threshold=0.3)
    box_annotator = sv.BoxAnnotator(color=TEAM_PALETTE)
    label_annotator = sv.LabelAnnotator(color=TEAM_PALETTE)
    last_transformer = None

    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
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
            if cls == player_class_id:
                team = kmeans.predict([get_jersey_color(frame, box)])[0]
                team_by_index[i] = team
                color_lookup[i] = team
                name = f"Team {'A' if team == 0 else 'B'}"
            elif cls == ball_class_id:
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
                radar = draw_points_on_pitch(CONFIG, xy[np.newaxis, :], face_color=color, pitch=radar)

            radar = sv.resize_image(radar, (w // 3, h // 3))
            rh, rw = radar.shape[:2]
            annotated[h - rh:, 0:rw] = radar

        writer.write(annotated)
        frame_idx += 1
        yield frame_idx, annotated

    cap.release()
    writer.release()


def handle_stripe_return():
    """After a successful Stripe Checkout redirect, verify payment server-side
    before unlocking premium (never trust the redirect alone)."""
    session_id = st.query_params.get("session_id")
    if not session_id or not STRIPE_SECRET_KEY:
        return
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        if session.payment_status == "paid":
            db.set_premium(True, stripe_customer_id=session.customer)
            st.session_state["just_upgraded"] = True
    except stripe.error.StripeError:
        pass
    st.query_params.clear()


if PAYMENTS_ENABLED:
    handle_stripe_return()
    if st.session_state.get("just_upgraded"):
        st.success("Payment confirmed — Premium unlocked!")
        del st.session_state["just_upgraded"]

with st.sidebar:
    st.header("Settings")
    source_choice = st.radio("Video source", ["Sample clip", "Upload your own"])

    if source_choice == "Sample clip":
        video_label = st.selectbox("Choose a clip", list(SAMPLE_VIDEOS.keys()))
        video_path = SAMPLE_VIDEOS[video_label]
        ensure_asset(video_path)
    else:
        uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        video_path = None
        video_label = uploaded.name if uploaded is not None else None
        if uploaded is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(uploaded.read())
            video_path = tmp.name

        with st.expander("Where to find more videos"):
            for name, url, desc in VIDEO_SOURCES:
                st.markdown(f"**[{name}]({url})**  \n{desc}")

    if PAYMENTS_ENABLED:
        premium = db.is_premium()
        tier_max = PREMIUM_TIER_MAX_FRAMES if premium else FREE_TIER_MAX_FRAMES
        st.caption(f"{'Premium' if premium else 'Free tier'} — up to {tier_max} frames per run")
    else:
        tier_max = SINGLE_TIER_MAX_FRAMES

    max_frames = st.slider(
        "Frames to process", min_value=30, max_value=tier_max, value=min(150, tier_max), step=30,
        help="Lower = faster preview. Sample clips are ~30fps.",
    )

    run_clicked = st.button("Run analysis", type="primary", disabled=video_path is None, use_container_width=True)

    if PAYMENTS_ENABLED and not premium:
        st.markdown("---")
        st.subheader("Upgrade to Premium")
        st.caption(f"Unlock up to {PREMIUM_TIER_MAX_FRAMES} frames per run — ${PREMIUM_PRICE_USD_CENTS / 100:.2f}")
        if not STRIPE_SECRET_KEY or not STRIPE_PUBLISHABLE_KEY:
            st.info(
                "Stripe isn't configured yet. Set the STRIPE_SECRET_KEY and "
                "STRIPE_PUBLISHABLE_KEY environment variables (test-mode keys from "
                "your Stripe Dashboard) to enable upgrades."
            )
        else:
            if st.button("Upgrade to Premium"):
                checkout_session = stripe.checkout.Session.create(
                    mode="payment",
                    line_items=[{
                        "price_data": {
                            "currency": "usd",
                            "product_data": {"name": "TacticEye — Premium"},
                            "unit_amount": PREMIUM_PRICE_USD_CENTS,
                        },
                        "quantity": 1,
                    }],
                    success_url="http://localhost:8501/?session_id={CHECKOUT_SESSION_ID}",
                    cancel_url="http://localhost:8501/",
                )
                st.link_button("Continue to payment", checkout_session.url)

tab_home, tab_features, tab_analyze, tab_history, tab_analytics, tab_waitlist, tab_contact = st.tabs(
    ["Home", "Features", "Analyze", "History", "Analytics", "Waitlist", "Contact"]
)

with tab_home:
    hero_b64 = load_image_b64(HERO_IMAGE_PATH) if os.path.exists(HERO_IMAGE_PATH) else None
    st.markdown(
        f"""
        <div class="viz-root">
          {f'''<div class="photo-hero" style="--hero-img: url('data:image/jpeg;base64,{hero_b64}');">
            <div class="photo-hero-content">
              <div class="eyebrow">Computer Vision · Sports Analytics</div>
              <h1>TacticEye</h1>
              <p>See the game the way a coach does — player detection, team
              identification, and a live tactical radar, from a custom-trained model.</p>
            </div>
          </div>''' if hero_b64 else ''}
          <div class="apple-section">
            <p>Upload match footage and get player detection, team identification, and a
            live tactical radar view — powered by a custom-trained detector, not a
            generic off-the-shelf model.</p>
            <div class="stat-row">
              <div><div class="stat-value">4</div><div class="stat-label">Detected classes</div></div>
              <div><div class="stat-value">15.8K</div><div class="stat-label">Training annotations</div></div>
              <div><div class="stat-value">77%</div><div class="stat-label">Overall mAP50</div></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Head to the **Analyze** tab in the sidebar's flow to run it on a real clip.")

with tab_analyze:
    st.markdown(
        f"""
        <div class="viz-root">
          <span class="legend-chip"><span class="legend-dot" style="background:{TEAM_A_COLOR};"></span>Team A</span>
          <span class="legend-chip"><span class="legend-dot" style="background:{TEAM_B_COLOR};"></span>Team B</span>
          <span class="legend-chip"><span class="legend-dot" style="background:#ffd700;"></span>Referee / Goalkeeper</span>
          <span class="legend-chip"><span class="legend-dot" style="background:#000000;"></span>Ball</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if run_clicked and video_path is not None:
        progress_bar = st.progress(0.0, text="Processing video...")
        frame_placeholder = st.empty()

        gen = process_video(video_path, max_frames)
        output_path = next(gen)  # the generator yields the output file path first
        frames_done = 0
        for frame_idx, annotated in gen:
            frames_done = frame_idx
            frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            progress_bar.progress(frame_idx / max_frames, text=f"Processing video... {int(frame_idx / max_frames * 100)}%")

        progress_bar.empty()
        db.add_history(video_label or "uploaded video", frames_done, output_path)

        # persist results in session_state: local variables here vanish on the next
        # Streamlit rerun (e.g. switching tabs), which broke the download button
        # right after it first appeared
        st.session_state["last_result"] = {"output_path": output_path}

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        if os.path.exists(result["output_path"]):
            st.success("Analysis complete — playback above was live frame-by-frame; download below for the encoded video file.")
            with open(result["output_path"], "rb") as f:
                st.download_button(
                    "Download result", f, file_name="tactical_analysis.mp4", mime="video/mp4",
                    key="download_result_btn",
                )
        else:
            st.warning("The processed video file is no longer available (temp file was cleaned up). Run the analysis again to redownload.")

with tab_history:
    history_rows = db.get_history()
    if not history_rows:
        st.caption("No analyses run yet — run one from the Analyze tab.")
    else:
        hist_display = pd.DataFrame([dict(r) for r in history_rows])
        hist_display["timestamp"] = pd.to_datetime(hist_display["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
        hist_display = hist_display.rename(columns={
            "timestamp": "Date", "video_name": "Video", "frames_processed": "Frames",
        })[["Date", "Video", "Frames"]]
        st.dataframe(hist_display, use_container_width=True, hide_index=True)

with tab_analytics:
    history_all = db.get_all_history()
    waitlist_all = db.get_all_waitlist()

    if not history_all and not waitlist_all:
        st.caption("No data yet — run an analysis or two to populate the dashboard.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Total analyses", len(history_all))
        col2.metric("Waitlist signups", len(waitlist_all))

        hist_df = pd.DataFrame([dict(r) for r in history_all]) if history_all else None
        wl_df = pd.DataFrame([dict(r) for r in waitlist_all]) if waitlist_all else None

        if hist_df is not None:
            hist_df["date"] = pd.to_datetime(hist_df["timestamp"]).dt.date
            st.subheader("Analyses per day")
            daily = hist_df.groupby("date").size().reset_index(name="count")
            fig = go.Figure(go.Bar(
                x=daily["date"].astype(str), y=daily["count"],
                marker_color=TEAM_B_COLOR, text=daily["count"], textposition="outside",
            ))
            st.plotly_chart(_style_fig(fig), use_container_width=True)

        if wl_df is not None:
            wl_df["date"] = pd.to_datetime(wl_df["timestamp"]).dt.date
            st.subheader("Waitlist signups over time (cumulative)")
            cum = wl_df.groupby("date").size().cumsum().reset_index(name="total")
            fig3 = go.Figure(go.Scatter(
                x=cum["date"].astype(str), y=cum["total"], mode="lines+markers",
                line=dict(color=TEAM_B_COLOR, width=2), marker=dict(size=8),
            ))
            st.plotly_chart(_style_fig(fig3), use_container_width=True)

        st.markdown("---")
        st.subheader("Export for Power BI / Excel")
        st.caption("Download CSVs and open them in Power BI Desktop (Get Data → Text/CSV) to build your own reports.")
        exp_col1, exp_col2 = st.columns(2)
        exp_col1.download_button(
            "history.csv", hist_df.to_csv(index=False) if hist_df is not None else "",
            file_name="history.csv", mime="text/csv", disabled=hist_df is None,
        )
        exp_col2.download_button(
            "waitlist.csv", wl_df.to_csv(index=False) if wl_df is not None else "",
            file_name="waitlist.csv", mime="text/csv", disabled=wl_df is None,
        )

with tab_waitlist:
    st.subheader("Premium Waitlist")
    if PAYMENTS_ENABLED:
        st.caption("Manage your Premium subscription above in the sidebar.")
    else:
        st.caption(f"Payments aren't live yet — join the waitlist and we'll email you when Premium launches. ({db.get_waitlist_count()} people so far)")
        with st.form("waitlist_form", clear_on_submit=True):
            wl_name = st.text_input("Name (optional)")
            wl_email = st.text_input("Email")
            submitted = st.form_submit_button("Join the waitlist")
            if submitted:
                if "@" not in wl_email or "." not in wl_email.split("@")[-1]:
                    st.error("Please enter a valid email address.")
                elif db.add_to_waitlist(wl_email.strip(), wl_name.strip() or None):
                    st.success("You're on the list!")
                else:
                    st.info("That email is already on the waitlist.")

with tab_features:
    sections_html = "".join(
        f"""<div class="apple-section">
              <div class="eyebrow">Step {i + 1:02d}</div>
              <h2>{title}</h2>
              <p>{desc}</p>
            </div>"""
        for i, (title, desc) in enumerate(PIPELINE_FEATURES)
    )
    st.markdown(f'<div class="viz-root">{sections_html}</div>', unsafe_allow_html=True)

    st.subheader("Model performance (held-out test set)")
    overall = MODEL_METRICS[MODEL_METRICS["Class"] == "All (overall)"].iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Precision", f"{overall['Precision']:.0%}")
    m2.metric("Recall", f"{overall['Recall']:.0%}")
    m3.metric("mAP50", f"{overall['mAP50']:.0%}")

    st.dataframe(
        MODEL_METRICS.style.format({"Precision": "{:.0%}", "Recall": "{:.0%}", "mAP50": "{:.0%}"}),
        use_container_width=True, hide_index=True,
    )

with tab_contact:
    st.markdown(
        f"""
        <div class="viz-root">
          <div class="apple-section">
            <div class="eyebrow">Get in touch</div>
            <h2>Questions or feedback?</h2>
            <p>Reach out at <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
