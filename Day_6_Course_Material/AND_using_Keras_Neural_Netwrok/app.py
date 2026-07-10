"""
AND Gate – Streamlit App
========================
Interactive demo: enter two binary inputs and see the AND prediction.
"""
import streamlit as st
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow import keras

st.set_page_config(page_title="AND Gate Neural Network", page_icon="🧠", layout="centered")

st.title("🧠 AND Gate – Keras Neural Network")
st.markdown("""
This app uses a **single-neuron Keras model** trained to learn the logical **AND** operation.
Enter two binary inputs below to see the model's prediction.
""")

# ── Load model ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "and_model.h5")

if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Model file `and_model.h5` not found. Run `python and_model.py` first.")
    st.stop()

model = keras.models.load_model(MODEL_PATH)

# ── User Input ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    a = st.selectbox("Input A", [0, 1], index=0)
with col2:
    b = st.selectbox("Input B", [0, 1], index=0)

if st.button("Predict", use_container_width=True):
    inp = np.array([[a, b]], dtype=np.float32)
    pred = model.predict(inp, verbose=0)[0][0]
    result = int(round(pred))

    st.markdown("---")
    st.metric(label=f"{a} AND {b}", value=result, delta=f"confidence {pred:.4f}")

    # Truth table
    st.subheader("Full Truth Table")
    all_inputs = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    all_preds = model.predict(all_inputs, verbose=0)
    for xi, pi in zip(all_inputs, all_preds):
        expected = int(xi[0]) & int(xi[1])
        icon = "✅" if int(round(pi[0])) == expected else "❌"
        st.write(f"{icon}  {int(xi[0])} AND {int(xi[1])} → **{int(round(pi[0]))}**  (raw: {pi[0]:.4f})")
