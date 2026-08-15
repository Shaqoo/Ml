"""
Train DBSCAN clustering for customer segmentation.
Saves the trained model to models/dbscan_model.pkl
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from pathlib import Path
from transformers import SegmentFeatureEngineer

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "customer_segmentation.csv"
MODEL_PATH = BASE_DIR / "models" / "dbscan_model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Load data ---
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# --- Features (no target) ---
X = df.drop(columns=["customer_id"])

# --- Column types ---
categorical_cols = ["channel", "region", "loyalty_tier"]
numerical_cols = ["age", "annual_income", "total_spent", "number_of_orders",
                  "avg_order_value", "days_since_last_purchase", "discount_usage_pct",
                  "website_visits", "spend_per_order", "income_to_spend_ratio",
                  "engagement_score", "discount_sensitivity", "is_active"]

# --- Preprocessing ---
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ]
)

# --- Feature engineering + preprocessing ---
feature_pipeline = Pipeline([
    ("features", SegmentFeatureEngineer()),
    ("preprocessor", preprocessor)
])

X_processed = feature_pipeline.fit_transform(X)
print(f"Processed shape: {X_processed.shape}")

# --- Train DBSCAN ---
# eps = maximum distance between two samples to be neighbors
# min_samples = minimum points to form a dense region
dbscan = DBSCAN(eps=2.0, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_processed)

# --- Evaluate ---
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"\n--- DBSCAN Results ---")
print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise}")

# Silhouette score (only if more than 1 cluster found)
if n_clusters > 1:
    # Exclude noise points for silhouette calculation
    mask = dbscan_labels != -1
    if mask.sum() > 1:
        sil = silhouette_score(X_processed[mask], dbscan_labels[mask])
        print(f"Silhouette: {sil:.4f}")

# --- Save model ---
joblib.dump({
    "feature_pipeline": feature_pipeline,
    "dbscan": dbscan,
}, MODEL_PATH)

print(f"\nModel saved to {MODEL_PATH}")
