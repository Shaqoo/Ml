import pandas as pd
import joblib
from datetime import date

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from pathlib import Path
from transformers import CarFeatureTransformer



BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "used_cars.csv"
MODEL_PATH = BASE_DIR / "used_cars_model.pkl"



# =========================
# 1. Load data
# =========================

df = pd.read_csv(DATA_PATH)

print(df.head())


# =========================
# 2. Features and target
# =========================

X = df.drop(columns=["price_usd", "car_id"])
y = df["price_usd"]


# =========================
# 4. Column types
# =========================

categorical_cols = [
    "brand",
    "model",
    "fuel_type",
    "transmission",
    "color",
    "condition"
]

numerical_cols = [
    "year",
    "mileage_km",
    "horsepower",
    "owners_count",
    "car_age",
    "mileage_per_year"
]


# =========================
# 5. Preprocessor
# =========================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numerical_cols
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_cols
        )
    ]
)


# =========================
# 6. Model
# =========================

model = HistGradientBoostingRegressor(
    random_state=42
)


# =========================
# 7. Pipeline
# =========================

pipeline = Pipeline([
    (
        "feature_transformer",
        CarFeatureTransformer(date.today().year)
    ),
    ("preprocessor", preprocessor),
    ("model", model)
])


# =========================
# 8. Train/Test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# 9. Cross-validation
# =========================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_r2 = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="r2"
)

print("\nCross-validation R2 scores:")
print(cv_r2)

print("Mean CV R2:", cv_r2.mean())
print("CV R2 std:", cv_r2.std())


# =========================
# 10. Cross-validation MAE
# =========================

cv_mae = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="neg_mean_absolute_error"
)

print("\nCross-validation MAE scores:")
print(-cv_mae)

print("Mean CV MAE:", -cv_mae.mean())


# =========================
# 11. Train final model
# =========================

pipeline.fit(X_train, y_train)


# =========================
# 12. Save model
# =========================

joblib.dump(
    pipeline,
    MODEL_PATH
)

print(f"\nModel saved to: {MODEL_PATH}")
# =========================
# 11. Train final model
# =========================

pipeline.fit(X_train, y_train)


# =========================
# 12. Save model
# =========================

joblib.dump(
    pipeline,
    MODEL_PATH
)

print(f"\nModel saved to: {MODEL_PATH}")


# =========================
# 13. Make predictions
# =========================

y_pred = pipeline.predict(X_test)


# =========================
# 14. Final test evaluation
# =========================

r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(
    y_test,
    y_pred
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5


print("\nFinal Test Results")
print("------------------")

print("R2 Score:", r2)
print("MSE:", mse)
print("MAE:", mae)
print("RMSE:", rmse)
