import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import tempfile
import os

st.title("🎥 AI vs Real Video Detector")

@st.cache_resource
def load_full_model():
    model_path = "https://huggingface.co/midhat14/DBproject/resolve/main/ai_vs_real_cnn_conv2d.h5"  # Your uploaded model file
    model = load_model(model_path)
    return model

# Load model
with st.spinner('Loading AI vs Real Detection model...'):
    model = load_full_model()

# Upload video file
video_file = st.file_uploader("Upload a video file (mp4)", type=["mp4"])

if video_file is not None:
    # Save video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())

    st.video(tfile.name)

    with st.spinner('Analyzing video frames...'):
        cap = cv2.VideoCapture(tfile.name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        interval = max(1, fps // 2)  # Take ~2 frames per second

        real_count = 0
        ai_count = 0
        total_frames = 0
        sample_frames = []

        for i in range(0, frame_count, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img).resize((224, 224))

            x = image.img_to_array(pil_img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0

            pred = model.predict(x)[0][0]

            if pred > 0.5:
                real_count += 1
            else:
                ai_count += 1

            total_frames += 1

            # Save a few sample frames for display
            if len(sample_frames) < 3:
                sample_frames.append((pil_img, pred))

        cap.release()

    st.write(f"Total frames analyzed: {total_frames}")
    st.write(f"✅ Real frames detected: {real_count}")
    st.write(f"⚠️ AI-generated frames detected: {ai_count}")

    real_percent = (real_count / total_frames) * 100
    ai_percent = (ai_count / total_frames) * 100

    st.write(f"🎯 Real Confidence: {real_percent:.2f}%")
    st.write(f"⚡ AI-generated Confidence: {ai_percent:.2f}%")

    st.subheader("Sample Frames with Predictions:")
    for idx, (img, pred) in enumerate(sample_frames):
        st.image(img, caption=f"Frame {idx+1} — Predicted as {'Real' if pred > 0.5 else 'AI-generated'} (Conf: {pred:.2f})", use_container_width=True)
