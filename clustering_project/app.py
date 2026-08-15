"""
FastAPI application for Customer Segmentation & Anomaly Detection.
Loads KMeans, DBSCAN, PCA, and IsolationForest models.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from contextlib import asynccontextmanager
from schema.segmentation import CustomerInput, ClusterResponse, AnomalyResponse, MetricsResponse

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# --- Model storage ---
bundle = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all clustering models on startup."""
    # Load KMeans model (main clustering)
    kmeans_data = joblib.load(MODEL_DIR / "kmeans_model.pkl")
    bundle["feature_pipeline"] = kmeans_data["feature_pipeline"]
    bundle["kmeans"] = kmeans_data["kmeans"]
    bundle["best_k"] = kmeans_data["best_k"]
    bundle["metrics"] = kmeans_data["metrics"]

    # Load Isolation Forest (anomaly detection)
    iso_data = joblib.load(MODEL_DIR / "isolation_model.pkl")
    bundle["iso_forest"] = iso_data["iso_forest"]

    yield
    bundle.clear()


# --- Create app ---
app = FastAPI(title="Customer Segmentation API", version="1.0", lifespan=lifespan)


@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "Customer Segmentation & Anomaly Detection API"}


@app.post("/predict-cluster", response_model=ClusterResponse)
def predict_cluster(input_data: CustomerInput):
    """Assign a customer to a cluster using KMeans."""
    # Convert input to DataFrame
    df = pd.DataFrame([input_data.model_dump()])
    # Apply feature engineering and preprocessing
    X = bundle["feature_pipeline"].transform(df)
    # Predict cluster
    cluster = int(bundle["kmeans"].predict(X)[0])
    return ClusterResponse(cluster=cluster, cluster_label=f"Cluster {cluster}")


@app.post("/detect-anomaly", response_model=AnomalyResponse)
def detect_anomaly(input_data: CustomerInput):
    """Detect if a customer is an outlier using Isolation Forest."""
    # Convert input to DataFrame
    df = pd.DataFrame([input_data.model_dump()])
    # Apply feature engineering and preprocessing
    X = bundle["feature_pipeline"].transform(df)
    # Predict: 1 = normal, -1 = anomaly
    pred = int(bundle["iso_forest"].predict(X)[0])
    # Anomaly score: lower = more anomalous
    score = float(bundle["iso_forest"].decision_function(X)[0])
    return AnomalyResponse(is_anomaly=(pred == -1), anomaly_score=round(score, 4))


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """Return clustering metrics (silhouette, inertia, cluster counts)."""
    return bundle["metrics"]
