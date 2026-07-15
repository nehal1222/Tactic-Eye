import base64
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import stripe
import streamlit as st
import streamlit.components.v1 as components
import supervision as sv
from sklearn.cluster import KMeans
from trackers import ByteTrackTracker
from ultralytics import YOLO

import db
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# Validated categorical/status palette (checked with dataviz skill's validate_palette.js:
# all CVD/contrast checks pass for both light and dark surfaces)
TEAM_A_COLOR = "#e34948"   # categorical red
TEAM_B_COLOR = "#2a78d6"   # categorical blue
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
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
.hero-banner {
  padding: 28px 32px;
  border-radius: 14px;
  margin-bottom: 8px;
  background: linear-gradient(135deg, #2a78d6 0%, #1c5cab 100%);
  color: white;
}
.hero-banner h1 { margin: 0; font-size: 1.9rem; font-weight: 700; }
.hero-banner p { margin: 6px 0 0 0; opacity: 0.9; font-size: 0.95rem; }
.legend-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px; border-radius: 999px; margin-right: 8px;
  font-size: 0.85rem; border: 1px solid var(--border);
}
.legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
</style>
"""

DINO_GAME_HTML = """
<div style="text-align:center;">
  <canvas id="dinoGame" width="600" height="180"
    style="background:#f7f7f7; border:2px solid #535353; border-radius:8px; max-width:100%;"></canvas>
  <div style="font-family:monospace; color:#535353; margin-top:4px; font-size:13px;">
    Press SPACE or tap the canvas to jump — pass the time while your analysis runs!
  </div>
</div>
<script>
(function() {
  const canvas = document.getElementById('dinoGame');
  const ctx = canvas.getContext('2d');
  const groundY = 140;
  let dino = { x: 40, y: groundY - 30, w: 30, h: 30, vy: 0, jumping: false };
  const gravity = 1.2;
  const jumpVelocity = -16;
  let obstacles = [];
  let speed = 6;
  let score = 0;
  let gameOver = false;
  let started = false;
  let frame = 0;

  function reset() {
    dino.y = groundY - dino.h;
    dino.vy = 0;
    dino.jumping = false;
    obstacles = [];
    speed = 6;
    score = 0;
    gameOver = false;
    frame = 0;
  }
  reset();

  function jump() {
    if (gameOver) { reset(); started = true; return; }
    started = true;
    if (!dino.jumping) {
      dino.vy = jumpVelocity;
      dino.jumping = true;
    }
  }

  document.addEventListener('keydown', function(e) {
    if (e.code === 'Space' || e.code === 'ArrowUp') { e.preventDefault(); jump(); }
  });
  canvas.addEventListener('mousedown', jump);
  canvas.addEventListener('touchstart', function(e) { e.preventDefault(); jump(); });

  function update() {
    if (!started || gameOver) return;
    frame++;
    dino.vy += gravity;
    dino.y += dino.vy;
    if (dino.y > groundY - dino.h) {
      dino.y = groundY - dino.h;
      dino.vy = 0;
      dino.jumping = false;
    }

    if (frame % Math.max(40, 90 - Math.floor(speed * 2)) === 0) {
      obstacles.push({ x: canvas.width, y: groundY - 25, w: 15 + Math.random() * 10, h: 25 });
    }
    obstacles.forEach(function(o) { o.x -= speed; });
    obstacles = obstacles.filter(function(o) { return o.x + o.w > 0; });

    obstacles.forEach(function(o) {
      if (dino.x < o.x + o.w && dino.x + dino.w > o.x && dino.y < o.y + o.h && dino.y + dino.h > o.y) {
        gameOver = true;
      }
    });

    score += 1;
    if (frame % 300 === 0) speed += 0.5;
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#535353';
    ctx.fillRect(0, groundY, canvas.width, 2);

    ctx.fillStyle = '#4c8bf5';
    ctx.fillRect(dino.x, dino.y, dino.w, dino.h);

    ctx.fillStyle = '#2e7d32';
    obstacles.forEach(function(o) { ctx.fillRect(o.x, o.y, o.w, o.h); });

    ctx.fillStyle = '#535353';
    ctx.font = '14px monospace';
    ctx.fillText('Score: ' + Math.floor(score / 10), canvas.width - 100, 20);

    if (!started) {
      ctx.fillText('Press SPACE or click to start', canvas.width / 2 - 90, canvas.height / 2);
    } else if (gameOver) {
      ctx.fillText('Game Over -- press SPACE to restart', canvas.width / 2 - 115, canvas.height / 2);
    }
  }

  function loop() {
    update();
    draw();
    requestAnimationFrame(loop);
  }
  loop();
})();
</script>
"""

PLAYER_MODEL_PATH = "models/best.pt"
PITCH_MODEL_PATH = "sports/examples/soccer/data/football-pitch-detection.pt"
SAMPLE_VIDEOS = {
    "0bfacc_0.mp4": "sports/examples/soccer/data/0bfacc_0.mp4",
    "2e57b9_0.mp4": "sports/examples/soccer/data/2e57b9_0.mp4",
    "08fd33_0.mp4": "sports/examples/soccer/data/08fd33_0.mp4",
    "573e61_0.mp4": "sports/examples/soccer/data/573e61_0.mp4",
    "121364_0.mp4": "sports/examples/soccer/data/121364_0.mp4",
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

st.set_page_config(page_title="AI Tactical Dashboard", layout="wide", page_icon="⚽")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="viz-root">
      <div class="hero-banner">
        <h1>⚽ AI Tactical Dashboard</h1>
        <p>Custom-trained YOLOv8 detection · ByteTrack · team classification · homography-based tactical radar</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
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


def circular_frame_html(frame_bgr, size=340):
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
    b64 = base64.b64encode(buf).decode("ascii")
    return f"""
    <style>
    @keyframes spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
    </style>
    <div style="display:flex; justify-content:center; margin:12px 0;">
      <div style="position:relative; width:{size}px; height:{size}px;">
        <div style="position:absolute; inset:0; border-radius:50%;
             background:conic-gradient(from 0deg, #cde2fb, #2a78d6, #184f95, #2a78d6, #cde2fb);
             animation: spin 3s linear infinite;"></div>
        <div style="position:absolute; inset:6px; border-radius:50%; overflow:hidden; background:#000;">
          <img src="data:image/jpeg;base64,{b64}" style="width:100%; height:100%; object-fit:cover;">
        </div>
      </div>
    </div>
    """


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
    """Yields the output path first, then (frame_index, annotated_frame_bgr) per
    frame, and finally ("summary", {0: frames_won, 1: frames_won}) tallying which
    team's players stayed closer to the ball on average each frame."""
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
    ball_proximity_wins = {0: 0, 1: 0}

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

            ball_idx = next(
                (i for i, cls in enumerate(detections.class_id) if cls == ball_class_id), None
            )
            if ball_idx is not None:
                ball_xy = pitch_xy[ball_idx]
                team_a_dists = [np.linalg.norm(pitch_xy[i] - ball_xy) for i, t in team_by_index.items() if t == 0]
                team_b_dists = [np.linalg.norm(pitch_xy[i] - ball_xy) for i, t in team_by_index.items() if t == 1]
                if team_a_dists and team_b_dists:
                    if np.mean(team_a_dists) < np.mean(team_b_dists):
                        ball_proximity_wins[0] += 1
                    else:
                        ball_proximity_wins[1] += 1

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
    yield "summary", ball_proximity_wins


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
    else:
        uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        video_path = None
        video_label = uploaded.name if uploaded is not None else None
        if uploaded is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(uploaded.read())
            video_path = tmp.name

    if PAYMENTS_ENABLED:
        premium = db.is_premium()
        tier_max = PREMIUM_TIER_MAX_FRAMES if premium else FREE_TIER_MAX_FRAMES
        st.caption(f"{'⭐ Premium' if premium else 'Free tier'} — up to {tier_max} frames per run")
    else:
        tier_max = SINGLE_TIER_MAX_FRAMES

    max_frames = st.slider(
        "Frames to process", min_value=30, max_value=tier_max, value=min(150, tier_max), step=30,
        help="Lower = faster preview. Sample clips are ~30fps.",
    )

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
                            "product_data": {"name": "AI Tactical Dashboard — Premium"},
                            "unit_amount": PREMIUM_PRICE_USD_CENTS,
                        },
                        "quantity": 1,
                    }],
                    success_url="http://localhost:8501/?session_id={CHECKOUT_SESSION_ID}",
                    cancel_url="http://localhost:8501/",
                )
                st.link_button("Continue to payment", checkout_session.url)

    if not PAYMENTS_ENABLED:
        st.markdown("---")
        st.subheader("⭐ Premium waitlist")
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

    st.markdown("---")
    st.subheader("🎮 Prediction game")
    total_points = db.get_total_points()
    total, correct = db.get_prediction_stats()
    st.metric("Points", total_points)
    if total > 0:
        st.caption(f"{correct}/{total} correct predictions")

    run_clicked = st.button("Run analysis", type="primary", disabled=video_path is None)

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

prediction = st.radio(
    "🎯 Before you run it: which team do you think will control the ball more in this clip?",
    ["Team A", "Team B"],
    horizontal=True,
)

if run_clicked and video_path is not None:
    progress_bar = st.progress(0.0, text="Processing video...")
    frame_placeholder = st.empty()

    st.caption("Bored waiting? Play while it processes:")
    components.html(DINO_GAME_HTML, height=210)  # rendered once, keeps running independently of the loop below

    gen = process_video(video_path, max_frames)
    output_path = next(gen)  # the generator yields the output file path first
    dominance = {0: 0, 1: 0}
    frames_done = 0
    for item in gen:
        if item[0] == "summary":
            dominance = item[1]
            break
        frame_idx, annotated = item
        frames_done = frame_idx
        frame_placeholder.markdown(circular_frame_html(annotated), unsafe_allow_html=True)
        progress_bar.progress(frame_idx / max_frames, text=f"Processing video... {int(frame_idx / max_frames * 100)}%")

    progress_bar.empty()

    actual_team = "Team A" if dominance[0] >= dominance[1] else "Team B"
    correct, points = db.add_prediction(video_label or "uploaded video", prediction, actual_team)
    db.add_history(video_label or "uploaded video", frames_done, output_path)

    # persist results in session_state: local variables here vanish on the next
    # Streamlit rerun (e.g. clicking an expander elsewhere), which broke the
    # download button and result banner right after they first appeared
    st.session_state["last_result"] = {
        "output_path": output_path,
        "actual_team": actual_team,
        "correct": correct,
        "points": points,
    }

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    if os.path.exists(result["output_path"]):
        st.success("Done! (playback above was live frame-by-frame; download below for the encoded video file)")
        with open(result["output_path"], "rb") as f:
            st.download_button(
                "Download result", f, file_name="tactical_analysis.mp4", mime="video/mp4",
                key="download_result_btn",
            )
        if result["correct"]:
            st.balloons()
            st.success(f"🎉 Correct! {result['actual_team']} controlled the ball more often (+{result['points']} points)")
        else:
            st.info(f"Not quite — {result['actual_team']} actually controlled the ball more often (+{result['points']} points)")
    else:
        st.warning("The processed video file is no longer available (temp file was cleaned up). Run the analysis again to redownload.")

with st.expander("📜 History"):
    history_rows = db.get_history()
    if not history_rows:
        st.caption("No analyses run yet.")
    else:
        for row in history_rows:
            st.text(f"{row['timestamp'][:19]} — {row['video_name']} ({row['frames_processed']} frames)")

with st.expander("📊 Analytics Dashboard"):
    history_all = db.get_all_history()
    predictions_all = db.get_all_predictions()
    waitlist_all = db.get_all_waitlist()

    if not history_all and not predictions_all and not waitlist_all:
        st.caption("No data yet — run an analysis or two to populate the dashboard.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total analyses", len(history_all))
        col2.metric("Total predictions", len(predictions_all))
        col3.metric("Waitlist signups", len(waitlist_all))

        hist_df = pd.DataFrame([dict(r) for r in history_all]) if history_all else None
        pred_df = pd.DataFrame([dict(r) for r in predictions_all]) if predictions_all else None
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

        if pred_df is not None:
            st.subheader("Prediction accuracy")
            counts = pred_df["correct"].value_counts()
            values = [int(counts.get(1, 0)), int(counts.get(0, 0))]
            fig = go.Figure(go.Bar(
                x=["Correct", "Incorrect"], y=values,
                marker_color=[STATUS_GOOD, STATUS_CRITICAL], text=values, textposition="outside",
            ))
            st.plotly_chart(_style_fig(fig), use_container_width=True)

            st.subheader("Which team dominates more often")
            team_counts = pred_df["actual_team"].value_counts()
            team_values = [int(team_counts.get("Team A", 0)), int(team_counts.get("Team B", 0))]
            fig2 = go.Figure(go.Bar(
                x=["Team A", "Team B"], y=team_values,
                marker_color=[TEAM_A_COLOR, TEAM_B_COLOR], text=team_values, textposition="outside",
            ))
            st.plotly_chart(_style_fig(fig2), use_container_width=True)

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
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        exp_col1.download_button(
            "history.csv", hist_df.to_csv(index=False) if hist_df is not None else "",
            file_name="history.csv", mime="text/csv", disabled=hist_df is None,
        )
        exp_col2.download_button(
            "predictions.csv", pred_df.to_csv(index=False) if pred_df is not None else "",
            file_name="predictions.csv", mime="text/csv", disabled=pred_df is None,
        )
        exp_col3.download_button(
            "waitlist.csv", wl_df.to_csv(index=False) if wl_df is not None else "",
            file_name="waitlist.csv", mime="text/csv", disabled=wl_df is None,
        )
