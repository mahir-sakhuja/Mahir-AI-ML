import streamlit as st
import numpy as np
from PIL import Image
import os
import urllib.request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

try:
    import onnxruntime as ort
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "onnxruntime"])
    import onnxruntime as ort

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐶", layout="centered")

MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "mobilenetv2-7.onnx")

@st.cache_resource
def load_onnx_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading pre-trained MobileNetV2 model (13MB)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return ort.InferenceSession(MODEL_PATH)

try:
    session = load_onnx_model()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    use_onnx = True
except Exception as e:
    st.warning(f"Failed to load ONNX model. Falling back to local training. Error: {e}")
    use_onnx = False

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
st.markdown("Upload a photo — the model will predict **Cat** or **Dog** using a RandomForest classifier trained on-the-fly.")

IMG_SIZE = 32
BASE_DIR = os.path.dirname(__file__)

def extract_features(img):
    # Feature 1: Raw pixels resized to 32x32
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    pixels = np.array(img_resized).flatten() / 255.0
    
    # Feature 2: Color Histogram (8 bins per channel)
    arr = np.array(img)
    hist_r, _ = np.histogram(arr[:, :, 0], bins=8, range=(0, 256))
    hist_g, _ = np.histogram(arr[:, :, 1], bins=8, range=(0, 256))
    hist_b, _ = np.histogram(arr[:, :, 2], bins=8, range=(0, 256))
    hist = np.concatenate([hist_r, hist_g, hist_b]) / arr.size
    
    return np.concatenate([pixels, hist])

def predict_image(image):
    if use_onnx:
        try:
            # 1. Resize to 224x224
            img = image.resize((224, 224))
            # 2. Convert to float32 normalized
            arr = np.array(img, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std
            # 3. Transpose to CHW and add batch dimension
            arr = np.transpose(arr, (2, 0, 1))
            arr = np.expand_dims(arr, axis=0)
            
            # 4. Run session
            outputs = session.run([output_name], {input_name: arr})
            logits = outputs[0][0]
            
            # 5. Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            # Dog classes: 151 to 275 (inclusive)
            # Cat classes: 281 to 287 (inclusive)
            dog_prob = np.sum(probs[151:276])
            cat_prob = np.sum(probs[281:288])
            
            total = cat_prob + dog_prob
            if total > 0:
                cat_p = cat_prob / total
                dog_p = dog_prob / total
            else:
                cat_p, dog_p = 0.5, 0.5
                
            pred = 0 if cat_p > dog_p else 1
            return pred, [cat_p, dog_p]
        except Exception as e:
            pass
            
    # Fallback to local RandomForest model
    features = extract_features(image)
    pred = model.predict([features])[0]
    proba = model.predict_proba([features])[0]
    return pred, proba

@st.cache_resource
def train_model():
    features_list, labels = [], []
    for label, folder in enumerate(["Cat", "Dog"]):
        folder_path = os.path.join(BASE_DIR, folder)
        if not os.path.exists(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            try:
                img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
                features_list.append(extract_features(img))
                labels.append(label)
            except Exception:
                pass

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
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

    prediction, proba = predict_image(image)

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
