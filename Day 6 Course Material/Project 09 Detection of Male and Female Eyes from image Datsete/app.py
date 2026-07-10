"""
Project 09 – Male/Female Eye Detection – Streamlit App
=======================================================
Upload an eye image and the CNN will classify it as Male or Female.
"""
import streamlit as st
import numpy as np
import os
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras

st.set_page_config(page_title="Eye Gender Detection", page_icon="👁️", layout="centered")

st.title("👁️ Male / Female Eye Detection")
st.markdown("""
Upload an eye image and the **CNN model** will predict whether it belongs
to a **male** or **female** subject.
""")

# ── Load model ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "eye_cnn_model.h5")
IMG_SIZE = 64
CLASSES = ["Female", "Male"]

if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Model file `eye_cnn_model.h5` not found. Run `python train_cnn.py` first.")
    st.stop()

model = keras.models.load_model(MODEL_PATH)

# ── Upload ──────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload an eye image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        st.error("Could not read the image.")
        st.stop()

    # Display original
    st.image(img, caption="Uploaded Image", width=200, clamp=True)

    # Preprocess
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    inp = resized.astype(np.float32).reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

    # Predict
    pred = model.predict(inp, verbose=0)[0][0]
    label_idx = int(round(pred))
    label = CLASSES[label_idx]
    confidence = pred if label_idx == 1 else 1 - pred

    st.markdown("---")
    st.success(f"### Prediction: **{label}**")
    st.metric("Confidence", f"{confidence * 100:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.progress(1 - pred, text=f"Female: {(1 - pred) * 100:.1f}%")
    with col2:
        st.progress(pred, text=f"Male: {pred * 100:.1f}%")

# ── About ───────────────────────────────────────────────────────────────────
with st.expander("ℹ️ About"):
    st.markdown("""
    - **Model**: CNN (3 Conv2D + MaxPool layers, Dense 128, Dropout 0.3)
    - **Input**: 64×64 grayscale eye image
    - **Classes**: Female (0), Male (1)
    - **Training**: Synthetic eye dataset (geometric patterns)
    """)
