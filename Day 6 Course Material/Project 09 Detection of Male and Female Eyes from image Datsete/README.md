# Project 09 – Detection of Male and Female Eyes from Image Dataset

CNN-based classifier that detects whether an eye image belongs to a male or female subject.

## Files
| File | Purpose |
|------|---------|
| `create_dummy_dataset.py` | Generates 200 synthetic eye images (100 male, 100 female) |
| `train_cnn.py` | Trains the CNN and saves `eye_cnn_model.h5` |
| `app.py` | Streamlit web UI for predictions |
| `requirements.txt` | Python dependencies |

## Quick Start
```bash
pip install -r requirements.txt
python create_dummy_dataset.py   # generate training images
python train_cnn.py              # train & save CNN
streamlit run app.py             # launch web UI
```

## CNN Architecture
```
Conv2D(32, 3×3) → MaxPool → Conv2D(64, 3×3) → MaxPool → Conv2D(64, 3×3) → MaxPool
→ Flatten → Dense(128) → Dropout(0.3) → Dense(1, sigmoid)
```

## Dataset
Synthetic 64×64 grayscale images:
- **Male eyes**: rounder shape, thicker brows
- **Female eyes**: almond shape, thinner brows, longer lashes
