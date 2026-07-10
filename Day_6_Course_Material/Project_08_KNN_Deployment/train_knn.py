"""
Project 08 – KNN Deployment (Training Script)
==============================================
Trains a K-Nearest Neighbors classifier on the Iris dataset
and saves the model + scaler for the Streamlit app.
"""
import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# ═══════════════════════════════════════════════════════════════════════════
# 1. Load Dataset
# ═══════════════════════════════════════════════════════════════════════════
print("📊 Loading Iris dataset …")
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names  = iris.target_names

print(f"   Samples       : {X.shape[0]}")
print(f"   Features      : {X.shape[1]} → {feature_names}")
print(f"   Classes       : {list(target_names)}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Preprocess
# ═══════════════════════════════════════════════════════════════════════════
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ═══════════════════════════════════════════════════════════════════════════
# 3. Find Best K
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Finding optimal K ──")
best_k, best_acc = 1, 0
for k in range(1, 20, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    tag = " ← best" if acc > best_acc else ""
    if acc > best_acc:
        best_k, best_acc = k, acc
    print(f"   K={k:2d}  accuracy={acc:.4f}{tag}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. Train Final Model
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n🏆 Using K={best_k} (accuracy {best_acc:.4f})")
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=target_names))

# ═══════════════════════════════════════════════════════════════════════════
# 5. Save
# ═══════════════════════════════════════════════════════════════════════════
joblib.dump(model,        "knn_iris_model.pkl")
joblib.dump(scaler,       "knn_iris_scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")
joblib.dump(list(target_names), "target_names.pkl")

print("\n✅ Model  → knn_iris_model.pkl")
print("✅ Scaler → knn_iris_scaler.pkl")
