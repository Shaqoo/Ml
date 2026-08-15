# Regression Project — House Price Prediction

## Goal

Build an ML model that predicts the price of a house based on its features.

## Dataset

**File:** `house_prices.csv` — 1000 rows

| Column | Type | Description |
| --- | --- | --- |
| house_id | ID | Unique identifier |
| neighborhood | Categorical | Downtown, Suburb, Rural, Midtown, Uptown |
| house_type | Categorical | House, Apartment, Condo, Townhouse |
| bedrooms | Numerical | 1–6 |
| bathrooms | Numerical | 1–4 |
| sqft | Numerical | Square footage (400–4500) |
| lot_size | Numerical | Lot size in sqft (500–20000) |
| year_built | Numerical | 1960–2024 |
| condition | Categorical | Excellent, Good, Fair, Poor |
| garage | Numerical | 0 or 1 |
| pool | Numerical | 0 or 1 |
| distance_to_city_km | Numerical | 0.5–40 km |
| price_usd | **Target** | House price |

## Algorithms

| Model | Train File | Saved Model |
| --- | --- | --- |
| LinearRegression | `train_linear.py` | `models/linear_model.pkl` |
| DecisionTreeRegressor | `train_tree.py` | `models/tree_model.pkl` |
| RandomForestRegressor | `train_random_forest.py` | `models/random_forest_model.pkl` |
| HistGradientBoostingRegressor | `train_gradient.py` | `models/gradient_model.pkl` |

## API Endpoints

```text
POST /predict-linear           → predict using LinearRegression
POST /predict-tree             → predict using DecisionTreeRegressor
POST /predict-random-forest    → predict using RandomForestRegressor
POST /predict-gradient         → predict using HistGradientBoostingRegressor
POST /predict-batch            → batch predict (model_name param: linear, tree, random_forest, gradient)
GET  /metrics                  → R2, MAE, RMSE for all 4 models
```

## Project Structure

```text
regression_project/
├── schema/
│   ├── __init__.py
│   └── house.py
├── models/
├── transformers.py          # FeatureEngineer custom transformer
├── train_linear.py
├── train_tree.py
├── train_random_forest.py
├── train_gradient.py
├── app.py
└── house_prices.csv
```

## Preprocessing Pipeline

- Custom transformer (`FeatureEngineer`) → house_age, bed_bath_ratio, sqft_per_room, is_new
- Numerical features → SimpleImputer(strategy="median") → StandardScaler
- Categorical features → SimpleImputer(strategy="most_frequent") → OneHotEncoder
- Cross-validation: 5-fold

## Run

```bash
# Train each model
python regression_project/train_linear.py
python regression_project/train_tree.py
python regression_project/train_random_forest.py
python regression_project/train_gradient.py

# Start API
uvicorn regression_project.app:app --reload
```
