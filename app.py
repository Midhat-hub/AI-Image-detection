# Paste your Streamlit code below
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.title("AI / Deepfake Image Detector")
st.write("Upload an image to check if it is AI-generated or real.")

# Load trained model
try:
    model = tf.keras.models.load_model('ai_vs_real_detector.h5')
    model_ready = True
except:
    st.warning("Model not found. Predictions will be available once training is complete.")
    model_ready = False

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])

def preprocess_image(image, size=(128,128)):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if model_ready:
        img_array = preprocess_image(image)
        pred = model.predict(img_array)[0][0]
        if pred > 0.5:
            st.error(f"Prediction: AI-generated ({pred*100:.2f}% confidence)")
        else:
            st.success(f"Prediction: Real ({(1-pred)*100:.2f}% confidence)")
    else:
        st.info("Model is still training. Predictions will be available once training is complete.")
