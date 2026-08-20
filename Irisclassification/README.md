# Iris Classifier — Streamlit App

Deploys the 5 models built in [svm_logistics.ipynb](svm_logistics.ipynb): SVC binary,
SVC multiclass, Logistic Regression binary, Logistic Regression One-vs-Rest, and
Logistic Regression multinomial (softmax).

Pick a model in the sidebar, set the 4 measurements, and get the predicted species.
Every prediction is written to MongoDB so the inputs can be used to retrain later.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `svm_binary.pkl`, `svm_multi.pkl`, `logistics_binary.pkl`, `logistics_ovr.pkl`, `logistics_multinomial.pkl` | The 5 trained models |
| `scaler_binary.pkl` | StandardScaler for the two binary models |
| `requirements.txt` | Dependencies |
| `.streamlit/secrets.toml` | MongoDB URI — **never commit this** |

## Scaled vs unscaled

The notebook trained the binary models on `scaler.fit_transform(x_train)` but the
multi-class models on the raw DataFrame. The app preserves that split in the
`MODELS` dict in [app.py](app.py) — `scaler_binary.pkl` is applied only to the two
binary models. Feeding a model the wrong one produces confident but wrong answers,
so if you ever retrain, keep that mapping in sync.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app runs without MongoDB — it just shows the record it would have saved.

## Connect MongoDB

1. Create a free cluster at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).
2. **Network Access** → add `0.0.0.0/0` so Streamlit Cloud can reach it.
3. **Connect → Drivers** → copy the connection string.
4. Paste it into `.streamlit/secrets.toml`:

```toml
MONGO_URI = "mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority"
```

Predictions land in database `iris`, collection `iris_predictions`.

## Deploy to Streamlit Cloud

```bash
git init && git add . && git commit -m "iris streamlit app"
git remote add origin <your-repo-url>
git push -u origin main
```

Then at [share.streamlit.io](https://share.streamlit.io): New app → pick the repo →
main file `app.py` → **Advanced settings → Secrets** → paste your `MONGO_URI` line.

`.gitignore` already excludes `.streamlit/secrets.toml`, so the URI never reaches
GitHub — add it through the Streamlit dashboard instead.

## Retraining on collected data

Each stored document has the 4 features, the model used, the prediction, the
confidence, and `actual_species` — the optional ground truth entered in the UI.
Only rows where `actual_species` is not null are usable for supervised retraining:

```python
rows = list(collection.find({"actual_species": {"$ne": None}}, {"_id": 0}))
```
