import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
from io import BytesIO

st.title("🧠 AI vs Real Image Detector")

@st.cache_resource  # Updated caching method
def load_full_model():
    model_path = "ai_vs_real_cnn_conv2d.h5"
    model = load_model(model_path)
    return model

# Load model once
with st.spinner('Loading model...'):
    model = load_full_model()

# Image uploader
uploaded_file = st.file_uploader("Upload an image (jpg, png)...", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(BytesIO(uploaded_file.read())).convert('RGB')
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess image
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    st.write(f"Input shape: {x.shape}")

    with st.spinner('Analyzing image...'):
        pred = model.predict(x)[0][0]

    if pred > 0.5:
        st.success(f"✅ Real Image (Confidence: {pred:.2f})")
    else:
        st.error(f"⚠️ AI-generated Image (Confidence: {1 - pred:.2f})")

