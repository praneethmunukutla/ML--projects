"""
Iris classifier - Streamlit app for the 5 models built in svm_logistics.ipynb.

Pick a model, enter the 4 flower measurements, get the predicted species back.
Every prediction is written to MongoDB so the inputs can be used to retrain later.

Run locally:  streamlit run app.py
"""

from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from pymongo.server_api import ServerApi

FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

# Which .pkl each model needs, and whether its inputs must be scaled.
#
# This mirrors how the notebook actually trained them: the binary models were fit
# on scaler.fit_transform(x_train), the multi-class models on the raw DataFrame.
# Feeding a model the wrong one gives confident, wrong answers - so it is recorded
# here rather than left to memory.
MODELS = {
    "SVC - binary (setosa vs versicolor)": {
        "file": "svm_binary.pkl",
        "scaler": "scaler_binary.pkl",
        "classes": CLASS_NAMES[:2],
        "note": "RBF kernel, C=60. Trained only on setosa and versicolor - it can never return virginica.",
    },
    "SVC - multiclass (all 3 species)": {
        "file": "svm_multi.pkl",
        "scaler": None,
        "classes": CLASS_NAMES,
        "note": "Linear kernel. Saved with probability=False, so no confidence scores are available.",
    },
    "Logistic Regression - binary (setosa vs versicolor)": {
        "file": "logistics_binary.pkl",
        "scaler": "scaler_binary.pkl",
        "classes": CLASS_NAMES[:2],
        "note": "Trained only on setosa and versicolor - it can never return virginica.",
    },
    "Logistic Regression - multiclass (One-vs-Rest)": {
        "file": "logistics_ovr.pkl",
        "scaler": None,
        "classes": CLASS_NAMES,
        "note": "Fits 3 separate binary classifiers, one per species, and takes the most confident.",
    },
    "Logistic Regression - multiclass (Multinomial / Softmax)": {
        "file": "logistics_multinomial.pkl",
        "scaler": None,
        "classes": CLASS_NAMES,
        "note": "One model over all 3 species at once - probabilities sum to 1 by construction.",
    },
}


# --------------------------------------------------------------------- loading
@st.cache_resource
def load_model(model_file, scaler_file):
    """Load a model and its scaler. Cached so the .pkl files are read once."""
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file) if scaler_file else None
    return model, scaler


@st.cache_resource
def get_collection():
    """Connect to MongoDB Atlas. Returns None if no URI is configured."""
    uri = st.secrets.get("MONGO_URI")
    # An unedited placeholder is still a truthy string - treat it as unconfigured
    # rather than burning the connection timeout on every page load.
    if not uri or "<" in uri:
        return None
    client = MongoClient(uri, server_api=ServerApi("1"),
                         serverSelectionTimeoutMS=5000)
    # fail fast on a bad URI instead of on first insert
    client.admin.command("ping")
    db = client["iris"]
    return db["iris_predictions"]


# ------------------------------------------------------------------ prediction
def predict(model, scaler, values):
    """Run one prediction. Returns (class_index, probabilities_or_None)."""
    df = pd.DataFrame([values], columns=FEATURES)

    # transform, never fit_transform - the mean/std must come from training data
    x = scaler.transform(df) if scaler is not None else df

    prediction = int(model.predict(x)[0])

    # svm_multi.pkl was saved with probability=False, so predict_proba is absent
    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(x)[0]
        except (AttributeError, NotImplementedError):
            probabilities = None

    return prediction, probabilities


def save_prediction(collection, record):
    """Write one prediction to MongoDB. Returns an error string, or None on success."""
    try:
        # copy: insert_one mutates with _id
        collection.insert_one(dict(record))
        return None
    except Exception as exc:
        return str(exc)


# ------------------------------------------------------------------------- app
def main():
    st.set_page_config(page_title="Iris Classifier",
                       page_icon="*", layout="centered")
    st.title("Iris Species Classifier")
    st.write(
        "Five models from the SVM and Logistic Regression class. "
        "Pick one, enter the measurements, and get the predicted species."
    )

    try:
        collection = get_collection()
    except Exception as exc:
        collection = None
        st.warning(
            f"MongoDB unavailable, predictions will not be saved: {exc}")

    # ---- model selection
    st.sidebar.header("Model")
    choice = st.sidebar.selectbox("Choose a model", list(MODELS.keys()))
    config = MODELS[choice]
    st.sidebar.info(config["note"])

    if collection is None:
        st.sidebar.warning(
            "Not connected to MongoDB - add MONGO_URI to your secrets.")
    else:
        st.sidebar.success("Connected to MongoDB")

    try:
        model, scaler = load_model(config["file"], config["scaler"])
    except FileNotFoundError:
        st.error(
            f"Could not find `{config['file']}`. "
            "Run the joblib.dump cells at the end of svm_logistics.ipynb first."
        )
        return

    # ---- feature inputs, ranged to the real iris data
    st.subheader("Measurements")
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
        petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 3.8, 0.1)
    with col2:
        sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1)
        petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.2, 0.1)

    values = [sepal_length, sepal_width, petal_length, petal_width]

    # Optional ground truth. Unlabelled inputs are useless for supervised
    # retraining, so this is what makes the stored rows worth collecting.
    actual = st.selectbox(
        "Actual species, if you know it (optional - used for retraining later)",
        ["Not sure"] + CLASS_NAMES,
    )

    # ---- predict
    if st.button("Predict species", type="primary"):
        prediction, probabilities = predict(model, scaler, values)
        species = CLASS_NAMES[prediction]

        st.success(f"Predicted species: **{species}**")

        if probabilities is not None:
            confidence = float(np.max(probabilities))
            st.metric("Confidence", f"{confidence:.1%}")
            st.bar_chart(
                pd.DataFrame(
                    {"probability": probabilities}, index=config["classes"]
                )
            )
        else:
            confidence = None
            st.caption(
                "This model was saved with probability=False, so no confidence "
                "score is available. Re-save it with SVC(..., probability=True) "
                "if you want one."
            )

        # ---- store the input for later retraining
        record = {
            **dict(zip(FEATURES, [float(v) for v in values])),
            "model": choice,
            "model_file": config["file"],
            "prediction": species,
            "confidence": confidence,
            "actual_species": None if actual == "Not sure" else actual,
            "timestamp": datetime.now(timezone.utc),
        }

        if collection is None:
            st.info("Prediction not saved - MongoDB is not configured.")
            with st.expander("Record that would have been saved"):
                st.json({k: str(v) for k, v in record.items()})
        else:
            error = save_prediction(collection, record)
            if error:
                st.warning(
                    f"Prediction succeeded but could not be saved: {error}")
            else:
                st.caption("Saved to MongoDB for future retraining.")

    # ---- what has been collected so far
    if collection is not None:
        with st.expander("Collected predictions"):
            try:
                rows = list(collection.find({}, {"_id": 0}).sort(
                    "timestamp", -1).limit(25))
                if rows:
                    st.caption(
                        f"{collection.count_documents({})} total, showing latest 25")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.write("No predictions stored yet.")
            except Exception as exc:
                st.warning(f"Could not read from MongoDB: {exc}")


if __name__ == "__main__":
    main()
