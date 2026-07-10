"""
XOR Gate using Neural Network (Keras)
======================================
Requires a hidden layer because XOR is not linearly separable.
"""
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow import keras
from tensorflow.keras import layers

# -- Data -----------------------------------------------------------------
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y = np.array([[0],    [1],    [1],    [0]],    dtype=np.float32)

# -- Model ----------------------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("Training XOR gate model ...")
history = model.fit(X, Y, epochs=3000, verbose=0)

# -- Evaluate -------------------------------------------------------------
predictions = model.predict(X, verbose=0)
print("\n  Input   | Expected | Predicted")
print("  --------|----------|----------")
for xi, yi, pi in zip(X, Y, predictions):
    print(f"  {int(xi[0])} XOR {int(xi[1])} |    {int(yi[0])}     |   {pi[0]:.4f}")

final_acc = history.history["accuracy"][-1]
print(f"\nFinal training accuracy: {final_acc:.4f}")

# -- Save -----------------------------------------------------------------
model.save("xor_model.h5")
print("[OK] Model saved as xor_model.h5")
