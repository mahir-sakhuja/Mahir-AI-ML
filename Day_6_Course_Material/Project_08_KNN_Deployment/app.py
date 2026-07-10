"""
Project 08 – KNN Deployment (Streamlit App)
============================================
Interactive Iris flower classifier powered by K-Nearest Neighbors.
"""
import streamlit as st
import numpy as np
import os
import joblib

st.set_page_config(page_title="KNN Iris Classifier", page_icon="🌸", layout="centered")

st.title("🌸 KNN Iris Flower Classifier")
st.markdown("""
Enter the measurements of an iris flower below, and the **K-Nearest Neighbors**
model will predict the species.
""")

# ── Load model ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

model_path  = os.path.join(BASE, "knn_iris_model.pkl")
scaler_path = os.path.join(BASE, "knn_iris_scaler.pkl")
feat_path   = os.path.join(BASE, "feature_names.pkl")
tgt_path    = os.path.join(BASE, "target_names.pkl")

for p, name in [(model_path, "Model"), (scaler_path, "Scaler")]:
    if not os.path.exists(p):
        st.error(f"⚠️ {name} not found. Run `python train_knn.py` first.")
        st.stop()

model   = joblib.load(model_path)
scaler  = joblib.load(scaler_path)
features = joblib.load(feat_path) if os.path.exists(feat_path) else \
           ["sepal length", "sepal width", "petal length", "petal width"]
targets  = joblib.load(tgt_path) if os.path.exists(tgt_path) else \
           ["setosa", "versicolor", "virginica"]

# ── Input sliders ───────────────────────────────────────────────────────────
st.subheader("📏 Flower Measurements (cm)")

cols = st.columns(2)
values = []
defaults = [5.1, 3.5, 1.4, 0.2]  # typical setosa

for i, (feat, default) in enumerate(zip(features, defaults)):
    with cols[i % 2]:
        val = st.slider(feat, 0.0, 10.0, default, 0.1)
        values.append(val)

# ── Predict ─────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Species", use_container_width=True):
    inp = np.array([values])
    inp_scaled = scaler.transform(inp)

    prediction = model.predict(inp_scaled)[0]
    probabilities = model.predict_proba(inp_scaled)[0]

    species = targets[prediction]

    st.markdown("---")
    st.success(f"### Predicted Species: **{species}**")

    st.subheader("Confidence per Class")
    for name, prob in zip(targets, probabilities):
        pct = prob * 100
        st.progress(prob, text=f"{name}: {pct:.1f}%")

# ── About section ───────────────────────────────────────────────────────────
with st.expander("ℹ️ About this app"):
    st.markdown("""
    - **Algorithm**: K-Nearest Neighbors (scikit-learn)
    - **Dataset**: Iris (150 samples, 3 classes)
    - **Features**: sepal length, sepal width, petal length, petal width
    - **Preprocessing**: StandardScaler
    - **Model selection**: best K chosen via accuracy on test set
    """)
