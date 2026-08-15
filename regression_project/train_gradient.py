"""
Train HistGradientBoosting Regressor for house price prediction.
Saves the trained model to models/gradient_model.pkl
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
from transformers import FeatureEngineer

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "house_prices.csv"
MODEL_PATH = BASE_DIR / "models" / "gradient_model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Load data ---
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# --- Features and target ---
X = df.drop(columns=["house_id", "price_usd"])
y = df["price_usd"]

# --- Column types ---
categorical_cols = ["neighborhood", "house_type", "condition"]
numerical_cols = ["bedrooms", "bathrooms", "sqft", "lot_size",
                  "year_built", "garage", "pool", "distance_to_city_km",
                  "house_age", "bed_bath_ratio", "sqft_per_room", "is_new"]

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

# --- Full pipeline ---
pipeline = Pipeline([
    ("features", FeatureEngineer()),
    ("preprocessor", preprocessor),
    ("model", HistGradientBoostingRegressor(random_state=42))
])

# --- Cross-validation ---
print("\n--- Cross-Validation (5-Fold) ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2 = cross_val_score(pipeline, X, y, cv=kf, scoring="r2")
cv_mae = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error")
cv_rmse = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_root_mean_squared_error")

print(f"R2:   {cv_r2.mean():.4f} (+/- {cv_r2.std():.4f})")
print(f"MAE:  {-cv_mae.mean():.2f} (+/- {cv_mae.std():.2f})")
print(f"RMSE: {-cv_rmse.mean():.2f} (+/- {cv_rmse.std():.2f})")

# --- Train on full data ---
pipeline.fit(X, y)
y_pred = pipeline.predict(X)

print("\n--- Training Metrics (full dataset) ---")
print(f"R2:   {r2_score(y, y_pred):.4f}")
print(f"MAE:  {mean_absolute_error(y, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.2f}")

# --- Save model ---
joblib.dump(pipeline, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
