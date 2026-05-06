import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os

# ---------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------

st.set_page_config(
    page_title="Helmet Detection System",
    layout="wide"
)

st.title("Helmet Detection System")

# ---------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------

model = YOLO("best.pt")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.5
)

input_type = st.sidebar.radio(
    "Select Input Type",
    ["Image", "Video"]
)

# ---------------------------------------------------
# IMAGE DETECTION
# ---------------------------------------------------

if input_type == "Image":

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        # YOLO Prediction
        results = model.predict(
            source=image,
            conf=confidence
        )

        annotated_image = results[0].plot()

        with col2:
            st.subheader("Detection Output")
            st.image(
                annotated_image,
                channels="BGR",
                use_container_width=True
            )

        # Detection Summary
        st.subheader("Detection Summary")

        boxes = results[0].boxes

        st.write(f"Total Objects Detected: {len(boxes)}")

        for i, box in enumerate(boxes):

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            conf_score = float(box.conf[0])

            st.write(
                f"{i+1}. {class_name} "
                f"(Confidence: {conf_score:.2f})"
            )

# ---------------------------------------------------
# VIDEO DETECTION
# ---------------------------------------------------

elif input_type == "Video":

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        st.subheader("Processing Video...")

        # Save uploaded video
        temp_input = tempfile.NamedTemporaryFile(delete=False)

        temp_input.write(uploaded_video.read())

        input_path = temp_input.name

        output_path = "output.mp4"

        # Video Capture
        cap = cv2.VideoCapture(input_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        progress_bar = st.progress(0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_count = 0

        # Process Video Frames
        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            results = model.predict(
                source=frame,
                conf=confidence
            )

            annotated_frame = results[0].plot()

            out.write(annotated_frame)

            frame_count += 1

            progress = int(
                (frame_count / total_frames) * 100
            )

            progress_bar.progress(
                min(progress, 100)
            )

        cap.release()
        out.release()

        st.success("Video Processing Completed")

        # Display Output Video
        video_file = open(output_path, "rb")

        video_bytes = video_file.read()

        st.video(video_bytes)

        # Download Button
        st.download_button(
            label="Download Processed Video",
            data=video_bytes,
            file_name="helmet_detection_output.mp4",
            mime="video/mp4"
        )

        os.remove(input_path)