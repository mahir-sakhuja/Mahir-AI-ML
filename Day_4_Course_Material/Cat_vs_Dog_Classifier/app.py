import streamlit as st
import numpy as np
from PIL import Image
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐶", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0b0f19 0%, #1a0a00 100%); color: #f3f4f6; }
    h1, h2, h3 { color: #fb923c !important; font-weight: 700 !important; }
    .result-box {
        border-radius: 16px; padding: 24px; text-align: center;
        font-size: 1.6rem; font-weight: 700; margin-top: 16px;
    }
    .cat { background: rgba(99,102,241,0.15); border: 2px solid #6366f1; color: #a5b4fc; }
    .dog { background: rgba(251,146,60,0.15); border: 2px solid #fb923c; color: #fdba74; }
</style>
""", unsafe_allow_html=True)

st.title("🐱 Cat vs Dog Classifier")
st.markdown("Upload a photo — the model will predict **Cat** or **Dog** using Logistic Regression trained on-the-fly.")

IMG_SIZE = 64
BASE_DIR = os.path.dirname(__file__)

@st.cache_resource
def train_model():
    images, labels = [], []
    for label, folder in enumerate(["Cat", "Dog"]):
        folder_path = os.path.join(BASE_DIR, folder)
        if not os.path.exists(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            try:
                img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE))
                images.append(np.array(img).flatten())
                labels.append(label)
            except Exception:
                pass

    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test) * 100
    return model, acc, len(X)

with st.spinner("⚙️ Training model on Cat & Dog images..."):
    model, accuracy, total = train_model()

col1, col2 = st.columns(2)
col1.metric("Training Images", total)
col2.metric("Test Accuracy", f"{accuracy:.1f}%")

st.write("---")
uploaded_file = st.file_uploader("📤 Upload a Cat or Dog Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    resized = np.array(image.resize((IMG_SIZE, IMG_SIZE))).flatten().astype(np.float32) / 255.0
    prediction = model.predict([resized])[0]
    proba = model.predict_proba([resized])[0]

    if prediction == 0:
        st.markdown(f'<div class="result-box cat">🐱 Prediction: CAT<br><small>Confidence: {proba[0]*100:.1f}%</small></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-box dog">🐶 Prediction: DOG<br><small>Confidence: {proba[1]*100:.1f}%</small></div>', unsafe_allow_html=True)

    st.write("---")
    st.subheader("Confidence Breakdown")
    st.write(f"🐱 Cat: **{proba[0]*100:.2f}%**")
    st.progress(int(proba[0] * 100))
    st.write(f"🐶 Dog: **{proba[1]*100:.2f}%**")
    st.progress(int(proba[1] * 100))