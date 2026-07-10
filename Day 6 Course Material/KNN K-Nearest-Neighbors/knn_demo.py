"""
KNN (K-Nearest Neighbors) – Theory & Demo
==========================================
Demonstrates the KNN algorithm on synthetic clustered data,
with visualization of decision boundaries.
"""
import numpy as np
import os
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ═══════════════════════════════════════════════════════════════════════════
# 1. Generate Synthetic Dataset
# ═══════════════════════════════════════════════════════════════════════════
print("📊 Generating synthetic clustered dataset …")
X, y = make_blobs(
    n_samples=500,
    centers=3,
    n_features=2,
    cluster_std=1.5,
    random_state=42
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"   Training samples : {len(X_train)}")
print(f"   Test samples     : {len(X_test)}")
print(f"   Number of classes: {len(np.unique(y))}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Train KNN with Different K Values
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Trying different K values ──")
best_k, best_acc = 1, 0

for k in range(1, 16, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    marker = " ★ best" if acc > best_acc else ""
    if acc > best_acc:
        best_k, best_acc = k, acc
    print(f"   K={k:2d}  →  accuracy = {acc:.4f}{marker}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Final Model with Best K
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n🏆 Best K = {best_k} (accuracy = {best_acc:.4f})")

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n── Classification Report ──")
print(classification_report(y_test, y_pred))

# ═══════════════════════════════════════════════════════════════════════════
# 4. Save Model & Scaler
# ═══════════════════════════════════════════════════════════════════════════
joblib.dump(model, "knn_model.pkl")
joblib.dump(scaler, "knn_scaler.pkl")

print("✅ Model saved as knn_model.pkl")
print("✅ Scaler saved as knn_scaler.pkl")
