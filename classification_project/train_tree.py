"""
Train Decision Tree Classifier for customer churn prediction.
Saves the trained model to models/tree_model.pkl
"""

import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from pathlib import Path
from transformers import ChurnFeatureEngineer

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "customer_churn.csv"
MODEL_PATH = BASE_DIR / "models" / "tree_model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Load data ---
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# --- Features and target ---
X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

# --- Column types ---
categorical_cols = ["gender", "contract_type", "payment_method", "internet_service"]
numerical_cols = ["age", "tenure_months", "monthly_charges", "total_spend",
                  "support_tickets", "has_partner", "has_dependents", "paperless_billing",
                  "tenure_years", "charge_per_month", "spend_per_ticket",
                  "is_high_charges", "is_new_customer"]

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
    ("features", ChurnFeatureEngineer()),
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(max_depth=8, random_state=42))
])

# --- Cross-validation ---
print("\n--- Cross-Validation (5-Fold) ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_acc = cross_val_score(pipeline, X, y, cv=kf, scoring="accuracy")
cv_f1 = cross_val_score(pipeline, X, y, cv=kf, scoring="f1")
cv_roc = cross_val_score(pipeline, X, y, cv=kf, scoring="roc_auc")

print(f"Accuracy: {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
print(f"F1:       {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
print(f"ROC-AUC:  {cv_roc.mean():.4f} (+/- {cv_roc.std():.4f})")

# --- Train on full data ---
pipeline.fit(X, y)

# --- Save model ---
joblib.dump(pipeline, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
