"""
FastAPI application for House Price Prediction.
Loads all 4 trained models and exposes prediction endpoints.
"""

import joblib
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from fastapi import FastAPI
from contextlib import asynccontextmanager
from schema.house import HouseInput, PredictionResponse, BatchPredictionResponse

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
SHADOW_LOG = BASE_DIR / "shadow_log.txt"
DATA_PATH = BASE_DIR / "house_prices.csv"

# --- Model storage ---
ml_models = {}
metrics_cache = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all 4 models and compute metrics on startup."""
    # Load each trained model
    for name in ["linear", "tree", "random_forest", "gradient"]:
        ml_models[name] = joblib.load(MODEL_DIR / f"{name}_model.pkl")

    # Compute training metrics for each model
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["house_id", "price_usd"])
    y = df["price_usd"]

    for name, model in ml_models.items():
        y_pred = model.predict(X)
        metrics_cache[name] = {
            "r2": round(float(r2_score(y, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y, y_pred)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y, y_pred))), 2),
        }

    yield
    ml_models.clear()
    metrics_cache.clear()


# --- Import metrics functions ---
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- Create app ---
app = FastAPI(title="House Price Prediction API", version="1.0", lifespan=lifespan)


@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "House Price Prediction API"}


@app.post("/predict-linear", response_model=PredictionResponse)
def predict_linear(input_data: HouseInput, shadow_mode: bool = False):
    """Predict price using Linear Regression."""
    model = ml_models["linear"]
    df = pd.DataFrame([input_data.model_dump()])
    prediction = model.predict(df)[0]

    # Shadow mode: log predictions without returning to user
    if shadow_mode:
        with open(SHADOW_LOG, "a") as f:
            f.write(f"{datetime.datetime.now()}, linear, {prediction:.2f}\n")

    return PredictionResponse(predicted_price=round(float(prediction), 2))


@app.post("/predict-tree", response_model=PredictionResponse)
def predict_tree(input_data: HouseInput, shadow_mode: bool = False):
    """Predict price using Decision Tree Regressor."""
    model = ml_models["tree"]
    df = pd.DataFrame([input_data.model_dump()])
    prediction = model.predict(df)[0]

    if shadow_mode:
        with open(SHADOW_LOG, "a") as f:
            f.write(f"{datetime.datetime.now()}, tree, {prediction:.2f}\n")

    return PredictionResponse(predicted_price=round(float(prediction), 2))


@app.post("/predict-random-forest", response_model=PredictionResponse)
def predict_random_forest(input_data: HouseInput, shadow_mode: bool = False):
    """Predict price using Random Forest Regressor."""
    model = ml_models["random_forest"]
    df = pd.DataFrame([input_data.model_dump()])
    prediction = model.predict(df)[0]

    if shadow_mode:
        with open(SHADOW_LOG, "a") as f:
            f.write(f"{datetime.datetime.now()}, random_forest, {prediction:.2f}\n")

    return PredictionResponse(predicted_price=round(float(prediction), 2))


@app.post("/predict-gradient", response_model=PredictionResponse)
def predict_gradient(input_data: HouseInput, shadow_mode: bool = False):
    """Predict price using HistGradientBoosting Regressor."""
    model = ml_models["gradient"]
    df = pd.DataFrame([input_data.model_dump()])
    prediction = model.predict(df)[0]

    if shadow_mode:
        with open(SHADOW_LOG, "a") as f:
            f.write(f"{datetime.datetime.now()}, gradient, {prediction:.2f}\n")

    return PredictionResponse(predicted_price=round(float(prediction), 2))


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(houses: List[HouseInput], model_name: str = "linear"):
    """Batch predict using any of the 4 models."""
    # Get the requested model (default to linear)
    model = ml_models.get(model_name, ml_models["linear"])
    # Convert list of inputs to DataFrame
    input_data = pd.DataFrame([h.model_dump() for h in houses])
    # Make predictions
    predictions = model.predict(input_data)
    return BatchPredictionResponse(predicted_prices=[round(float(p), 2) for p in predictions])


@app.get("/metrics")
def get_metrics():
    """Return training metrics (R2, MAE, RMSE) for all 4 models."""
    return metrics_cache
