# Classification Project — Customer Churn Prediction

## Goal

Build an ML model that predicts whether a customer will cancel their subscription (churn).

## Dataset

**File:** `customer_churn.csv` — 1000 rows

| Column | Type | Description |
| --- | --- | --- |
| customer_id | ID | Unique identifier |
| age | Numerical | 18–75 |
| gender | Categorical | Male, Female |
| tenure_months | Numerical | 1–72 months |
| monthly_charges | Numerical | $20–$120 |
| total_spend | Numerical | Total amount spent |
| contract_type | Categorical | Month-to-month, One year, Two year |
| payment_method | Categorical | Credit card, Bank transfer, Electronic check, Mailed check |
| internet_service | Categorical | DSL, Fiber optic, No |
| support_tickets | Numerical | 0–12 tickets |
| has_partner | Numerical | 0 or 1 |
| has_dependents | Numerical | 0 or 1 |
| paperless_billing | Numerical | 0 or 1 |
| churn | **Target** | 0 = stays, 1 = leaves |

## Algorithms

| Model | Train File | Saved Model |
| --- | --- | --- |
| LogisticRegression | `train_logistic.py` | `models/logistic_model.pkl` |
| DecisionTreeClassifier | `train_tree.py` | `models/tree_model.pkl` |
| RandomForestClassifier | `train_random_forest.py` | `models/random_forest_model.pkl` |
| HistGradientBoostingClassifier | `train_gradient.py` | `models/gradient_model.pkl` |

## API Endpoints

```text
POST /predict-logistic       → predict using LogisticRegression
POST /predict-tree           → predict using DecisionTreeClassifier
POST /predict-random         → predict using RandomForestClassifier
POST /predict-gradient       → predict using HistGradientBoostingClassifier
POST /predict-batch          → batch predict (model_name param: logistic, tree, random_forest, hist_gradient)
GET  /metrics                → Accuracy, F1, ROC-AUC for all 4 models
```

## Project Structure

```text
classification_project/
├── schema/
│   ├── __init__.py
│   └── churn.py
├── models/
├── transformers.py          # ChurnFeatureEngineer custom transformer
├── train_logistic.py
├── train_tree.py
├── train_random_forest.py
├── train_gradient.py
├── app.py
└── customer_churn.csv
```

## Preprocessing Pipeline

- Custom transformer (`ChurnFeatureEngineer`) → tenure_years, charge_per_month, spend_per_ticket, is_high_charges, is_new_customer
- Numerical features → SimpleImputer(strategy="median") → StandardScaler
- Categorical features → SimpleImputer(strategy="most_frequent") → OneHotEncoder
- Cross-validation: 5-fold

## Run

```bash
# Train each model
python classification_project/train_logistic.py
python classification_project/train_tree.py
python classification_project/train_random_forest.py
python classification_project/train_gradient.py

# Start API
uvicorn classification_project.app:app --reload
```
