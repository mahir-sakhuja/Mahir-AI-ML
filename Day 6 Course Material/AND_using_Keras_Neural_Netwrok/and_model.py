"""
AND Gate using Keras Neural Network
====================================
Trains a simple single-neuron model to learn the logical AND operation.
"""
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow import keras
from tensorflow.keras import layers

# -- Data -----------------------------------------------------------------
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y = np.array([[0],    [0],    [0],    [1]],    dtype=np.float32)

# -- Model ----------------------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("Training AND gate model ...")
history = model.fit(X, Y, epochs=3000, verbose=0)

# -- Evaluate -------------------------------------------------------------
predictions = model.predict(X, verbose=0)
print("\n  Input   | Expected | Predicted")
print("  --------|----------|----------")
for xi, yi, pi in zip(X, Y, predictions):
    print(f"  {int(xi[0])} AND {int(xi[1])} |    {int(yi[0])}     |   {pi[0]:.4f}")

# -- Save -----------------------------------------------------------------
model.save("and_model.h5")
print("\n[OK] Model saved as and_model.h5")
