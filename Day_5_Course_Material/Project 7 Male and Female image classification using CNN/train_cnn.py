import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

dataset_path = r"c:\Users\Mahir sakhuja\New folder (4)\Day 5 Course Material\Project 7 Male and Female image classification using CNN\dataset"
classes = ["Male", "Female"]
IMG_SIZE = 64

images = []
labels = []

for label, folder in enumerate(classes):
    folder_path = os.path.join(dataset_path, folder)
    if not os.path.exists(folder_path):
        continue
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # Normalize pixel values
        img = img / 255.0
        images.append(img)
        labels.append(label)

X = np.array(images)
y = np.array(labels)

print(f"Total training images: {len(X)}")

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Binary classification: 0 = Male, 1 = Female
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    X_train, y_train,
    epochs=5,
    validation_data=(X_test, y_test),
    batch_size=8
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"CNN Model test accuracy: {accuracy * 100:.2f}%")

model_save_path = r"c:\Users\Mahir sakhuja\New folder (4)\Day 5 Course Material\Project 7 Male and Female image classification using CNN\gender_cnn_model.keras"
model.save(model_save_path)
print(f"Model saved successfully to {model_save_path}!")
