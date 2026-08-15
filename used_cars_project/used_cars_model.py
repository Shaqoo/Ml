import pandas as pd
import joblib
import datetime

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent



# =========================
# 1. Load data
# =========================

df = pd.read_csv(BASE_DIR / "used_cars.csv")

print(df.head())


# =========================
# 2. Features and target
# =========================

X = df.drop(columns=["price_usd", "car_id"])
y = df["price_usd"]

class CarFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for creating additional features for used-car data.

    Creates:
        car_age:
            The age of the car in years.

        mileage_per_year:
            The average mileage driven per year.
            A minimum car age of 1 year is used to avoid division by zero.

    Parameters
    ----------
    current_year : int, default=2026
        The year used to calculate the car's age.
    """

    def __init__(self, current_year=2026):
        self.current_year = current_year

    def fit(self, X, y=None):
        """
        Fit the transformer.

        No parameters need to be learned from the data, so the
        transformer simply returns itself.
        """
        return self

    def transform(self, X):
        """
        Create car_age and mileage_per_year features.

        Parameters
        ----------
        X : pandas.DataFrame
            Input car dataset.

        Returns
        -------
        pandas.DataFrame
            Dataset with the new engineered features.
        """
        X = X.copy()

        # Calculate the age of the car
        X["car_age"] = self.current_year - X["year"]

        # Calculate average mileage per year
        X["mileage_per_year"] = (
            X["mileage_km"] /
            X["car_age"].clip(lower=1)
        )

        return X




# =========================
# 3. Column types
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
# 4. Preprocessor
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
        ),
    ],
    remainder='passthrough'                    
)


# =========================
# 5. Model
# =========================

model = HistGradientBoostingRegressor(
    random_state=42
)


# =========================
# 6. Pipeline
# =========================

pipeline = Pipeline([
    ("feature_transformer", CarFeatureTransformer(datetime.date.today().year)),
    ("preprocessor", preprocessor),
    ("model", model)
])


# =========================
# 7. Train/Test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# 8. Cross-validation
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
# 9. Cross-validation MAE
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
# 10. Train final model
# =========================

pipeline.fit(X_train, y_train)


# =========================
# 11. Save model
# =========================

joblib.dump(
    pipeline,
    "used_cars_model.pkl"
)


# =========================
# 12. Make predictions
# =========================

y_pred = pipeline.predict(X_test)


# =========================
# 13. Final test evaluation
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
