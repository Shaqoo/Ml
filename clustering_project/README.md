# Clustering Project — Customer Segmentation & Anomaly Detection

## Goal

Discover natural customer groups and identify unusual/anomalous customers with no predefined labels.

## Dataset

**File:** `customer_segmentation.csv` — 1000 rows

| Column | Type | Description |
| --- | --- | --- |
| customer_id | ID | Unique identifier |
| age | Numerical | 18–70 |
| annual_income | Numerical | $20k–$150k |
| total_spent | Numerical | Total spending |
| number_of_orders | Numerical | 1–80 |
| avg_order_value | Numerical | $10–$300 |
| days_since_last_purchase | Numerical | 0–365 days |
| discount_usage_pct | Numerical | 0%–60% |
| website_visits | Numerical | 5–500 |
| channel | Categorical | Online, In-store, Mobile app |
| region | Categorical | North, South, East, West |
| loyalty_tier | Categorical | Bronze, Silver, Gold, Platinum |

**No target column — this is unsupervised learning.**

## Algorithms

| Model | Train File | Saved Model |
| --- | --- | --- |
| KMeans | `train_kmeans.py` | `models/kmeans_model.pkl` |
| DBSCAN | `train_dbscan.py` | `models/dbscan_model.pkl` |
| PCA | `train_pca.py` | `models/pca_model.pkl` |
| IsolationForest | `train_isolation.py` | `models/isolation_model.pkl` |

## API Endpoints

```text
POST /predict-cluster     → assign customer to a cluster (KMeans)
POST /detect-anomaly      → detect if customer is an outlier (IsolationForest)
GET  /metrics             → silhouette score, inertia, cluster counts
```

## Project Structure

```text
clustering_project/
├── schema/
│   ├── __init__.py
│   └── segmentation.py
├── models/
├── transformers.py          # SegmentFeatureEngineer custom transformer
├── train_kmeans.py
├── train_dbscan.py
├── train_pca.py
├── train_isolation.py
├── app.py
└── customer_segmentation.csv
```

## Preprocessing Pipeline

- Custom transformer (`SegmentFeatureEngineer`) → spend_per_order, income_to_spend_ratio, engagement_score, discount_sensitivity, is_active
- Numerical features → SimpleImputer(strategy="median") → StandardScaler
- Categorical features → SimpleImputer(strategy="most_frequent") → OneHotEncoder
- No target variable — the algorithm discovers structure on its own

## Run

```bash
# Train each model
python clustering_project/train_kmeans.py
python clustering_project/train_dbscan.py
python clustering_project/train_pca.py
python clustering_project/train_isolation.py

# Start API
uvicorn clustering_project.app:app --reload
```
