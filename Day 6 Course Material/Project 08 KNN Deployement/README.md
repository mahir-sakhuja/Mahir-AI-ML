# Project 08 – KNN Deployment

Interactive **Iris flower classifier** powered by K-Nearest Neighbors, deployed via Streamlit.

## Files
| File | Purpose |
|------|---------|
| `train_knn.py` | Loads Iris data, finds best K, trains & saves model |
| `app.py` | Streamlit web UI for predictions |
| `requirements.txt` | Python dependencies |

## Quick Start
```bash
pip install -r requirements.txt
python train_knn.py        # train & save model
streamlit run app.py       # launch web UI
```

## Deployment on Streamlit Cloud
1. Push this folder to a public GitHub repo.
2. In Streamlit Cloud → New App → select `app.py`.
3. Done! The app will auto-install from `requirements.txt`.
