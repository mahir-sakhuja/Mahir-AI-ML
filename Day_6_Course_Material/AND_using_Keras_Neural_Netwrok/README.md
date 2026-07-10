# AND Gate using Keras Neural Network

A single-neuron Keras model that learns the logical AND truth table.

## Files
| File | Purpose |
|------|---------|
| `and_model.py` | Trains the model and saves `and_model.h5` |
| `app.py` | Streamlit UI for interactive predictions |
| `requirements.txt` | Python dependencies |

## Quick Start
```bash
pip install -r requirements.txt
python and_model.py        # train & save model
streamlit run app.py       # launch web UI
```

## How It Works
- **Architecture**: 1 Dense neuron with sigmoid activation
- **Loss**: Binary cross-entropy
- **Optimizer**: Adam
- **Training**: 1 000 epochs on the 4-row AND truth table

The model learns to output ~0 for inputs (0,0), (0,1), (1,0) and ~1 for (1,1).
