"""
Deep Learning with Basic Neural Network
========================================
Demonstrates a simple feed-forward neural network for classification
using synthetic data, built with TensorFlow / Keras.
"""
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === 1. Generate Synthetic Dataset (Two Moons) ===========================
print("[DATA] Generating synthetic 'two moons' dataset ...")
X, y = make_moons(n_samples=1000, noise=0.25, random_state=42)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training samples : {len(X_train)}")
print(f"   Test samples     : {len(X_test)}")

# === 2. Build Model ======================================================
model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(8,  activation="relu"),
    layers.Dense(1,  activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# === 3. Train ============================================================
print("\n[TRAIN] Training ...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=1
)

# === 4. Evaluate ==========================================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n[RESULT] Test Accuracy : {test_acc:.4f}")
print(f"[RESULT] Test Loss     : {test_loss:.4f}")

# === 5. Save model & scaler ==============================================
model.save("basic_nn_model.h5")

import joblib
joblib.dump(scaler, "scaler.pkl")

print("\n[OK] Model saved as basic_nn_model.h5")
print("[OK] Scaler saved as scaler.pkl")

# === 6. Quick Prediction Demo ============================================
print("\n-- Sample Predictions --")
sample_indices = np.random.choice(len(X_test), 5, replace=False)
for i in sample_indices:
    pred = model.predict(X_test[i:i+1], verbose=0)[0][0]
    label = int(round(pred))
    actual = y_test[i]
    icon = "OK" if label == actual else "WRONG"
    print(f"  [{icon}] Actual={actual}  Predicted={label}  (raw={pred:.4f})")
