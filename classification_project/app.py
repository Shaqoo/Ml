"""
FastAPI application for Customer Churn Prediction.
Loads all 4 trained models and exposes prediction endpoints.
"""

import joblib
import pandas as pd
from pathlib import Path
from typing import List
from fastapi import FastAPI
from contextlib import asynccontextmanager
from schema.churn import ChurnInput, ChurnResponse, BatchChurnResponse

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# --- Model storage ---
ml_models = {}
metrics_cache = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all 4 models and metrics on startup."""
    # Load each trained model
    for name in ["logistic", "tree", "random_forest", "hist_gradient"]:
        ml_models[name] = joblib.load(MODEL_DIR / f"{name}_model.pkl")

    # Load precomputed metrics
    metrics_cache.update(joblib.load(MODEL_DIR / "metrics.pkl"))

    yield
    ml_models.clear()
    metrics_cache.clear()


# --- Create app ---
app = FastAPI(title="Customer Churn Prediction API", version="1.0", lifespan=lifespan)


@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "Customer Churn Prediction API"}


@app.post("/predict-logistic", response_model=ChurnResponse)
def predict_logistic(input_data: ChurnInput):
    """Predict churn using Logistic Regression."""
    df = pd.DataFrame([input_data.model_dump()])
    model = ml_models["logistic"]
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return ChurnResponse(churn=bool(pred), probability=round(float(prob), 4))


@app.post("/predict-tree", response_model=ChurnResponse)
def predict_tree(input_data: ChurnInput):
    """Predict churn using Decision Tree Classifier."""
    df = pd.DataFrame([input_data.model_dump()])
    model = ml_models["tree"]
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return ChurnResponse(churn=bool(pred), probability=round(float(prob), 4))


@app.post("/predict-random", response_model=ChurnResponse)
def predict_random(input_data: ChurnInput):
    """Predict churn using Random Forest Classifier."""
    df = pd.DataFrame([input_data.model_dump()])
    model = ml_models["random_forest"]
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return ChurnResponse(churn=bool(pred), probability=round(float(prob), 4))


@app.post("/predict-gradient", response_model=ChurnResponse)
def predict_gradient(input_data: ChurnInput):
    """Predict churn using HistGradientBoosting Classifier."""
    df = pd.DataFrame([input_data.model_dump()])
    model = ml_models["hist_gradient"]
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return ChurnResponse(churn=bool(pred), probability=round(float(prob), 4))


@app.post("/predict-batch", response_model=BatchChurnResponse)
def predict_batch(inputs: List[ChurnInput], model_name: str = "logistic"):
    """Batch predict using any of the 4 models."""
    # Get the requested model (default to logistic)
    model = ml_models.get(model_name, ml_models["logistic"])
    # Convert list of inputs to DataFrame
    df = pd.DataFrame([h.model_dump() for h in inputs])
    # Make predictions
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]
    results = [ChurnResponse(churn=bool(p), probability=round(float(pr), 4))
               for p, pr in zip(preds, probs)]
    return BatchChurnResponse(results=results)


@app.get("/metrics")
def get_metrics():
    """Return Accuracy, F1, ROC-AUC for all 4 models."""
    return metrics_cache
