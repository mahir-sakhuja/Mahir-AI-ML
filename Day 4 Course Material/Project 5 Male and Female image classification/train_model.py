import os
import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

dataset_path = r"c:\Users\Mahir sakhuja\New folder (4)\Day 4 Course Material\Project 5 Male and Female image classification\dataset"
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
        img = img.flatten()
        images.append(img)
        labels.append(label)

X = np.array(images)
y = np.array(labels)

print(f"Total training images: {len(X)}")

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Calculate accuracy
accuracy = model.score(X_test, y_test)
print(f"Model test accuracy: {accuracy * 100:.2f}%")

model_save_path = r"c:\Users\Mahir sakhuja\New folder (4)\Day 4 Course Material\Project 5 Male and Female image classification\gender_ml_model.pkl"
joblib.dump(model, model_save_path)
print(f"Model saved successfully to {model_save_path}!")
