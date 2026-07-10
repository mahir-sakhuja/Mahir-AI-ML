import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Gender Classifier (CNN)",
    page_icon="🧠",
    layout="centered"
)

# -------------------------
# Load CNN Model
# -------------------------
model_path = os.path.join(os.path.dirname(__file__), "gender_cnn_model.keras")
model = tf.keras.models.load_model(model_path)

IMG_SIZE = 64

st.title("🧠 Male & Female Image Classifier (CNN)")
st.write("Upload an image to predict whether it is Male or Female using a Convolutional Neural Network.")

# -------------------------
# Upload Image
# -------------------------
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image using Pillow
    image = Image.open(uploaded_file)

    # Convert to RGB
    image = image.convert("RGB")

    # Display image
    st.image(image, caption="Uploaded Image", width=300)

    # Resize image
    resized = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to NumPy array and normalize
    resized_arr = np.array(resized) / 255.0

    # Expand dims to represent batch (1, 64, 64, 3)
    input_data = np.expand_dims(resized_arr, axis=0)

    # Prediction (probability of class 1 - Female)
    prediction_prob = model.predict(input_data)[0][0]

    # Calculate probabilities
    female_prob = float(prediction_prob)
    male_prob = 1.0 - female_prob

    # Display prediction
    if male_prob > female_prob:
        st.success(f"👨 Prediction: MALE")
    else:
        st.success(f"👩 Prediction: FEMALE")

    # Display probabilities
    st.subheader("Prediction Confidence")
    st.write(f"👨 Male Probability: **{male_prob * 100:.2f}%**")
    st.write(f"👩 Female Probability: **{female_prob * 100:.2f}%**")
