"""
XOR Gate – Streamlit App
========================
Interactive demo: enter two binary inputs and see the XOR prediction.
"""
import streamlit as st
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow import keras

st.set_page_config(page_title="XOR Gate Neural Network", page_icon="⚡", layout="centered")

st.title("⚡ XOR Gate – Keras Neural Network")
st.markdown("""
This app uses a **multi-layer Keras model** trained to learn the logical **XOR** operation.  
XOR is **not linearly separable**, so the network needs at least one hidden layer.
""")

# ── Load model ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xor_model.h5")

if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Model file `xor_model.h5` not found. Run `python xor_model.py` first.")
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
    st.metric(label=f"{a} XOR {b}", value=result, delta=f"confidence {pred:.4f}")

    # Truth table
    st.subheader("Full Truth Table")
    all_inputs = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    all_preds = model.predict(all_inputs, verbose=0)
    expected_xor = [0, 1, 1, 0]
    for xi, pi, ex in zip(all_inputs, all_preds, expected_xor):
        icon = "✅" if int(round(pi[0])) == ex else "❌"
        st.write(f"{icon}  {int(xi[0])} XOR {int(xi[1])} → **{int(round(pi[0]))}**  (raw: {pi[0]:.4f})")
