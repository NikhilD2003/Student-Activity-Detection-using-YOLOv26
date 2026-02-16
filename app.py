import streamlit as st
import tempfile
import os
import cv2
import subprocess
import shutil
import plotly.express as px
import av

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from inference_engine import run_inference_streaming
from analytics import compute_analytics


# ============================================================
# ✅ MUST BE FIRST STREAMLIT COMMAND
# ============================================================

st.set_page_config(layout="wide")


# ============================================================
# ✅ LOAD MODEL ONCE (THREAD-SAFE)
# ============================================================

@st.cache_resource
def load_model():
    return YOLO("runs/detect/weights/best.pt")

model = load_model()


# ============================================================
# WEBRTC CONFIG
# ============================================================

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class VideoProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.model = model
        self.frame_count = 0
        self.skip = 12

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % self.skip == 0:

            small = cv2.resize(img, (640, 360))

            results = self.model(
                small,
                imgsz=480,
                conf=0.3,
                verbose=False
            )

            if results and results[0].boxes is not None:
                for box in results[0].boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================================================
# FFMPEG
# ============================================================

def reencode_for_browser(src, dst):

    ffmpeg_bin = shutil.which("ffmpeg")

    if ffmpeg_bin is None:
        raise RuntimeError("FFmpeg not found")

    cmd = [
        ffmpeg_bin, "-y", "-i", src,
        "-vcodec", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-acodec", "aac",
        dst,
    ]

    subprocess.run(cmd, check=True)


# ============================================================
# UI
# ============================================================

st.title("🎓 Student Activity Detection Dashboard")


# ================= LIVE =================

st.subheader("📡 Live Classroom Monitoring")

webrtc_streamer(
    key="live",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=lambda: VideoProcessor(model),
    media_stream_constraints={"video": True, "audio": False},
)


# ================= UPLOAD MODE =================

uploaded_file = st.file_uploader(
    "Upload classroom video",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_file:

    with tempfile.TemporaryDirectory() as tmpdir:

        input_path = os.path.join(tmpdir, uploaded_file.name)

        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        output_video = os.path.join(tmpdir, "output_raw.mp4")
        csv_file = os.path.join(tmpdir, "detections.csv")

        if st.button("▶ Run Detection"):

            progress_bar = st.progress(0)
            frame_slot = st.empty()

            def progress_cb(p):
                progress_bar.progress(min(int(p * 100), 100))

            def frame_cb(frame):
                preview = cv2.resize(frame, (960, 540))
                rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                frame_slot.image(rgb, channels="RGB", use_container_width=True)

            out_vid, out_csv = run_inference_streaming(
                input_path,
                output_video,
                csv_file,
                model_path="runs/detect/weights/best.pt",
                progress_callback=progress_cb,
                frame_callback=frame_cb,
            )

            st.success("Inference complete!")

            browser_video = out_vid.replace(".mp4", "_browser.mp4")
            reencode_for_browser(out_vid, browser_video)

            video_bytes = open(browser_video, "rb").read()
            csv_bytes = open(out_csv, "rb").read()

            analytics = compute_analytics(out_csv)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.video(video_bytes)

            with col2:
                st.metric("Total Students Detected", analytics["total_students"])

            st.dataframe(analytics["student_activity_duration"])
