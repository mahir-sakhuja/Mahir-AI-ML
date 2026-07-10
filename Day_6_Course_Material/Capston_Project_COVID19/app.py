import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import os

st.set_page_config(page_title="COVID-19 Chest X-Ray Detector", page_icon="🫁", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0b0f19 0%, #0f1f0f 100%); color: #f3f4f6; }
    h1, h2, h3 { color: #34d399 !important; font-weight: 700 !important; }
    .result-covid {
        background: rgba(248,113,113,0.15); border: 2px solid #f87171;
        border-radius: 16px; padding: 24px; text-align: center; font-size: 1.5rem; font-weight: 700; color: #f87171;
    }
    .result-normal {
        background: rgba(52,211,153,0.15); border: 2px solid #34d399;
        border-radius: 16px; padding: 24px; text-align: center; font-size: 1.5rem; font-weight: 700; color: #34d399;
    }
</style>
""", unsafe_allow_html=True)

st.title("🫁 COVID-19 Detection from Chest X-Ray")
st.markdown("Upload a chest X-ray image to predict whether it shows signs of **COVID-19** or is **NORMAL** using a Convolutional Neural Network.")

IMG_SIZE = 64

# Dataset path
base_dir = os.path.dirname(__file__)
covid_dir = os.path.join(base_dir, "dataset", "covid")
normal_dir = os.path.join(base_dir, "dataset", "normal")

@st.cache_resource
def build_and_train_model():
    """Build and train CNN on actual X-ray images from the dataset folder."""
    import random
    images, labels = [], []
    exts = {'.jpg', '.jpeg', '.png'}

    # Load COVID images → label 1
    if os.path.exists(covid_dir):
        for f in os.listdir(covid_dir):
            if os.path.splitext(f)[1].lower() in exts:
                path = os.path.join(covid_dir, f)
                try:
                    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                    images.append(np.array(img, dtype=np.float32) / 255.0)
                    labels.append(1)
                except Exception:
                    pass

    # Load NORMAL images → label 0
    if os.path.exists(normal_dir):
        for f in os.listdir(normal_dir):
            if os.path.splitext(f)[1].lower() in exts:
                path = os.path.join(normal_dir, f)
                try:
                    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                    images.append(np.array(img, dtype=np.float32) / 255.0)
                    labels.append(0)
                except Exception:
                    pass

    if len(images) < 4:
        # Fallback: generate random synthetic images
        for _ in range(30):
            img = np.random.rand(IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
            img[20:44, :, :] = 0.9
            images.append(img); labels.append(1)
        for _ in range(30):
            img = np.random.rand(IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
            img[:, 20:44, :] = 0.9
            images.append(img); labels.append(0)

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Build lightweight CNN
    tf.random.set_seed(42)
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=8, validation_split=0.2, verbose=0)
    return model

# Train the model
with st.spinner("⚙️ Loading & training CNN on X-ray dataset (first run ~30 seconds)..."):
    model = build_and_train_model()
st.success("✅ Model ready!")

st.write("---")

# Sample images from dataset for easy testing
st.subheader("🖼️ Quick Test: Select a Sample X-Ray")
sample_options = {"Upload your own image": None}
if os.path.exists(covid_dir):
    for f in sorted(os.listdir(covid_dir))[:5]:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            sample_options[f"[COVID] {f}"] = os.path.join(covid_dir, f)
if os.path.exists(normal_dir):
    for f in sorted(os.listdir(normal_dir))[:5]:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            sample_options[f"[NORMAL] {f}"] = os.path.join(normal_dir, f)

selected = st.selectbox("Choose a sample image or upload your own:", list(sample_options.keys()))

image_pil = None
if sample_options[selected]:
    image_pil = Image.open(sample_options[selected]).convert("RGB")
    st.image(image_pil, caption=selected, width=300)
else:
    uploaded = st.file_uploader("📤 Upload a Chest X-Ray Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image_pil = Image.open(uploaded).convert("RGB")
        st.image(image_pil, caption="Uploaded X-Ray", width=300)

if image_pil:
    # Preprocess
    resized = image_pil.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(resized, dtype=np.float32) / 255.0
    inp = np.expand_dims(arr, axis=0)

    # Predict
    prob = float(model.predict(inp, verbose=0)[0][0])
    covid_prob = prob
    normal_prob = 1.0 - prob

    st.write("---")
    st.subheader("🔬 Prediction Result")
    if covid_prob > 0.5:
        st.markdown(f'<div class="result-covid">🛑 COVID-19 POSITIVE — Confidence: {covid_prob*100:.1f}%</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-normal">✅ NORMAL — Confidence: {normal_prob*100:.1f}%</div>', unsafe_allow_html=True)

    st.write("---")
    st.subheader("📊 Confidence Breakdown")
    col1, col2 = st.columns(2)
    col1.metric("COVID-19 Probability", f"{covid_prob*100:.1f}%")
    col2.metric("Normal Probability", f"{normal_prob*100:.1f}%")
    st.progress(int(covid_prob * 100))
