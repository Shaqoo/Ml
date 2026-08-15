"""
Train KMeans clustering for customer segmentation.
Saves the trained model to models/kmeans_model.pkl
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.cluster import KMeans
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
MODEL_PATH = BASE_DIR / "models" / "kmeans_model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Load data ---
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# --- Features (no target in unsupervised learning) ---
X = df.drop(columns=["customer_id"])

# --- Column types (after feature engineering) ---
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

# --- Feature engineering + preprocessing pipeline ---
feature_pipeline = Pipeline([
    ("features", SegmentFeatureEngineer()),
    ("preprocessor", preprocessor)
])

# Transform all data
X_processed = feature_pipeline.fit_transform(X)
print(f"Processed shape: {X_processed.shape}")

# --- Find optimal K using silhouette score ---
print("\n--- Finding Optimal K ---")
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_processed)
    sil = silhouette_score(X_processed, labels)
    sil_scores.append(sil)
    print(f"K={k}: silhouette={sil:.4f}")

# Pick K with highest silhouette score
best_k = list(K_range)[np.argmax(sil_scores)]
print(f"\nBest K by silhouette: {best_k}")

# --- Train final KMeans ---
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_processed)

# --- Evaluate ---
sil_final = silhouette_score(X_processed, kmeans_labels)
print(f"\n--- Final KMeans (K={best_k}) ---")
print(f"Silhouette: {sil_final:.4f}")
print(f"Inertia: {kmeans.inertia_:.0f}")

# Print cluster distribution
unique, counts = np.unique(kmeans_labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Cluster {u}: {c} customers")

# --- Save model ---
joblib.dump({
    "feature_pipeline": feature_pipeline,
    "kmeans": kmeans,
    "best_k": best_k,
    "metrics": {
        "n_clusters": best_k,
        "silhouette_score": round(float(sil_final), 4),
        "inertia": round(float(kmeans.inertia_), 2),
        "cluster_counts": {int(u): int(c) for u, c in zip(unique, counts)},
    }
}, MODEL_PATH)

print(f"\nModel saved to {MODEL_PATH}")
