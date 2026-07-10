import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Gender Classifier (CNN)",
    page_icon="🧠",
    layout="centered"
)

IMG_SIZE = 64

# -------------------------
# Build & Train Model (cached so it only runs once)
# -------------------------
@st.cache_resource
def build_and_train_model():
    """Build a simple CNN and train it on synthetic dummy data."""
    np.random.seed(42)
    tf.random.set_seed(42)

    # Generate synthetic training data
    # Male images: random noise with a vertical bright stripe
    # Female images: random noise with a horizontal bright stripe
    X, y = [], []
    for _ in range(80):
        img = np.random.randint(0, 50, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8).astype(np.float32) / 255.0
        img[:, 28:36, :] = 0.9  # vertical stripe → Male
        X.append(img)
        y.append(0)
    for _ in range(80):
        img = np.random.randint(0, 50, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8).astype(np.float32) / 255.0
        img[28:36, :, :] = 0.9  # horizontal stripe → Female
        X.append(img)
        y.append(1)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Build CNN
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=8, batch_size=16, validation_split=0.2, verbose=0)
    return model

# -------------------------
# Load model
# -------------------------
st.title("🧠 Male & Female Image Classifier (CNN)")
st.write("Upload an image to predict whether it is Male or Female using a Convolutional Neural Network.")

with st.spinner("⚙️ Loading CNN model (first run may take ~30 seconds)..."):
    model = build_and_train_model()

st.success("✅ Model ready!")

# -------------------------
# Upload Image
# -------------------------
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess
    resized = image.resize((IMG_SIZE, IMG_SIZE))
    resized_arr = np.array(resized, dtype=np.float32) / 255.0
    input_data = np.expand_dims(resized_arr, axis=0)

    # Predict
    prediction_prob = float(model.predict(input_data, verbose=0)[0][0])
    female_prob = prediction_prob
    male_prob = 1.0 - female_prob

    # Display prediction
    if male_prob > female_prob:
        st.success("👨 Prediction: MALE")
    else:
        st.success("👩 Prediction: FEMALE")

    st.subheader("Prediction Confidence")
    st.write(f"👨 Male Probability:   **{male_prob * 100:.2f}%**")
    st.write(f"👩 Female Probability: **{female_prob * 100:.2f}%**")
