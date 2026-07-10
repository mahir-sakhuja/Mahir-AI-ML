# XOR Gate using Neural Network (Keras)

A multi-layer Keras model that learns the logical XOR truth table.  
Unlike AND/OR, XOR is **not linearly separable** and requires hidden layers.

## Files
| File | Purpose |
|------|---------|
| `xor_model.py` | Trains the model and saves `xor_model.h5` |
| `app.py` | Streamlit UI for interactive predictions |
| `requirements.txt` | Python dependencies |

## Quick Start
```bash
pip install -r requirements.txt
python xor_model.py        # train & save model
streamlit run app.py       # launch web UI
```

## How It Works
- **Architecture**: 2 → 4 (ReLU) → 4 (ReLU) → 1 (Sigmoid)
- **Loss**: Binary cross-entropy
- **Optimizer**: Adam
- **Training**: 2 000 epochs on the 4-row XOR truth table

The hidden layers allow the network to learn the non-linear XOR decision boundary.
