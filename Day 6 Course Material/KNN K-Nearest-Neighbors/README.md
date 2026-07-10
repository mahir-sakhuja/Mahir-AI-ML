# KNN (K-Nearest Neighbors)

Demonstrates the **K-Nearest Neighbors** classification algorithm on synthetic clustered data.

## Files
| File | Purpose |
|------|---------|
| `knn_demo.py` | Generates data, compares K values, trains the best model, saves it |

## Quick Start
```bash
pip install scikit-learn joblib
python knn_demo.py
```

## Key Concepts
- **Distance metrics**: Euclidean (default), Manhattan, Minkowski
- **Choosing K**: odd values to avoid ties; tested K=1…15
- **Scaling**: `StandardScaler` ensures equal feature contribution
- **Decision boundary**: KNN draws non-linear boundaries naturally
