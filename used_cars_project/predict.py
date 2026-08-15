import joblib
import pandas as pd
from pathlib import Path
from transformers import CarFeatureTransformer



BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "used_cars_model.pkl"


# =========================
# 1. Load trained model
# =========================

model = joblib.load(MODEL_PATH)


# =========================
# 2. New car
# =========================

new_car = pd.DataFrame([{
    "brand": "Honda",
    "model": "Civic",
    "year": 2018,
    "mileage_km": 60000,
    "fuel_type": "Petrol",
    "transmission": "Automatic",
    "horsepower": 150,
    "color": "Black",
    "owners_count": 2,
    "condition": "Good"
}])


# =========================
# 3. Make prediction
# =========================

prediction = model.predict(new_car)


# =========================
# 4. Display prediction
# =========================

print("Predicted price: $", round(prediction[0], 2))
