import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Gender Classifier (ML)",
    page_icon="👤",
    layout="centered"
)

# -------------------------
# Load Model
# -------------------------
model_path = os.path.join(os.path.dirname(__file__), "gender_ml_model.pkl")
model = joblib.load(model_path)

IMG_SIZE = 64

st.title("👤 Male & Female Image Classifier (ML)")
st.write("Upload an image to predict whether it is Male or Female.")

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

    # Convert to NumPy array
    resized = np.array(resized)

    # Flatten image for Logistic Regression
    resized = resized.flatten()

    # Prediction
    prediction = model.predict([resized])[0]
    probability = model.predict_proba([resized])[0]

    # Display prediction
    if prediction == 0:
        st.success("👨 Prediction: MALE")
    else:
        st.success("👩 Prediction: FEMALE")

    # Display probabilities
    st.subheader("Prediction Confidence")
    st.write(f"👨 Male Probability: **{probability[0] * 100:.2f}%**")
    st.write(f"👩 Female Probability: **{probability[1] * 100:.2f}%**")
