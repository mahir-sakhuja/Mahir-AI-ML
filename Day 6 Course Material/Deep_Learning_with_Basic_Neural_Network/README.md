# Deep Learning with Basic Neural Network

Introductory deep-learning example using a feed-forward neural network
on a synthetic **two-moons** classification dataset.

## Files
| File | Purpose |
|------|---------|
| `basic_nn.py` | Generates data, trains, evaluates, and saves the model |

## Quick Start
```bash
pip install tensorflow scikit-learn joblib
python basic_nn.py
```

## Architecture
```
Input (2) → Dense(16, ReLU) → Dense(16, ReLU) → Dense(8, ReLU) → Dense(1, Sigmoid)
```

## Key Concepts Demonstrated
- Generating non-linearly separable data with `make_moons`
- Feature scaling with `StandardScaler`
- Train/test split
- Keras Sequential API
- Binary cross-entropy loss
- Model evaluation and saving
