"""
Train PCA for dimensionality reduction (visualization).
Saves the trained model to models/pca_model.pkl
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from pathlib import Path
from transformers import SegmentFeatureEngineer

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "customer_segmentation.csv"
MODEL_PATH = BASE_DIR / "models" / "pca_model.pkl"
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

# --- Train PCA (reduce to 2 dimensions for visualization) ---
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_processed)

# --- Evaluate ---
print(f"\n--- PCA Results ---")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")

# --- Save model ---
joblib.dump({
    "feature_pipeline": feature_pipeline,
    "pca": pca,
}, MODEL_PATH)

print(f"\nModel saved to {MODEL_PATH}")
