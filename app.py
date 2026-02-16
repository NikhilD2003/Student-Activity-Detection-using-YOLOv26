import streamlit as st
import tempfile
import os
import cv2
import subprocess
import shutil
import pandas as pd
import plotly.express as px
from ultralytics import YOLO

from inference_engine import run_inference_streaming
from analytics import compute_analytics


# ============================================================
# FFMPEG RE-ENCODER (Browser Safe)
# ============================================================

def reencode_for_browser(src, dst):

    ffmpeg_bin = shutil.which("ffmpeg")

    if ffmpeg_bin is None:
        raise RuntimeError(
            "FFmpeg not found in environment."
        )

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

st.set_page_config(layout="wide")
st.title("🎓 Student Activity Detection Dashboard")

mode = st.radio("Select Input Mode", ["Upload Video", "Live Camera"])


# ============================================================
# 🟢 LIVE CAMERA MODE
# ============================================================

if mode == "Live Camera":

    start_live = st.button("▶ Start Live Detection")

    if start_live:

        model = YOLO("runs/detect/weights/best.pt")

        cap = cv2.VideoCapture(0)

        frame_slot = st.empty()

        col1, col2 = st.columns([2, 1])

        live_df = pd.DataFrame(columns=[
            "timestamp", "frame", "student_id",
            "class_name", "confidence", "x1", "y1", "x2", "y2"
        ])

        frame_count = 0

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            results = model.track(frame, persist=True, verbose=False)

            annotated = results[0].plot()

            # show live video
            frame_slot.image(annotated, channels="BGR", use_container_width=True)

            # build live dataframe
            if results[0].boxes.id is not None:

                ids = results[0].boxes.id.cpu().numpy().astype(int)
                cls = results[0].boxes.cls.cpu().numpy().astype(int)
                conf = results[0].boxes.conf.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()

                for i in range(len(ids)):

                    live_df.loc[len(live_df)] = [
                        frame_count / 30,
                        frame_count,
                        ids[i],
                        model.names[cls[i]],
                        conf[i],
                        *boxes[i]
                    ]

            # update analytics every few frames
            if frame_count % 15 == 0 and len(live_df) > 0:

                analytics = compute_analytics_from_df(live_df)

                with col2:

                    st.subheader("📊 Live Summary")

                    st.metric("Total Students", analytics["total_students"])

                    fig = px.bar(
                        analytics["activity_distribution"],
                        x="class_name",
                        y="frames"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("⏱ Duration (seconds)")
                    st.dataframe(
                        analytics["student_activity_duration"],
                        use_container_width=True
                    )

        cap.release()


# ============================================================
# 🔵 VIDEO UPLOAD MODE (YOUR ORIGINAL — UNCHANGED)
# ============================================================

elif mode == "Upload Video":

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

                    frame_slot.image(
                        rgb,
                        channels="RGB",
                        use_container_width=True,
                    )

                with st.spinner("Running inference..."):

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

                with st.spinner("Preparing video for playback..."):
                    reencode_for_browser(out_vid, browser_video)

                with open(browser_video, "rb") as f:
                    video_bytes = f.read()

                with open(out_csv, "rb") as f:
                    csv_bytes = f.read()

                analytics = compute_analytics(out_csv)

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("🎥 Annotated Video")
                    st.video(video_bytes)

                    st.download_button(
                        "⬇ Download Video",
                        video_bytes,
                        file_name="output_inference_browser.mp4",
                        mime="video/mp4",
                    )

                with col2:
                    st.subheader("📊 Summary")

                    st.metric(
                        "Total Students Detected",
                        analytics["total_students"],
                    )

                    st.subheader("Activity Distribution")

                    fig = px.bar(
                        analytics["activity_distribution"],
                        x="class_name",
                        y="frames",
                        hover_data=["students"],
                        labels={
                            "class_name": "Activity",
                            "frames": "Total Frames",
                            "students": "Student IDs",
                        },
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.download_button(
                        "⬇ Download CSV",
                        csv_bytes,
                        file_name="detections.csv",
                        mime="text/csv",
                    )

                st.subheader("📈 Activity Timeline (Frame vs Activity)")

                timeline_df = analytics["timeline"].reset_index()

                timeline_long = timeline_df.melt(
                    id_vars="frame",
                    var_name="Activity",
                    value_name="Number of Students",
                )

                fig_timeline = px.line(
                    timeline_long,
                    x="frame",
                    y="Number of Students",
                    color="Activity",
                )

                st.plotly_chart(fig_timeline, use_container_width=True)

                st.subheader("🧾 Raw Detection Log (Preview)")
                st.dataframe(analytics["raw_df"].head(400))

                st.subheader("⏱ Per-Student Activity Duration (seconds)")
                st.dataframe(
                    analytics["student_activity_duration"],
                    use_container_width=True
                )
