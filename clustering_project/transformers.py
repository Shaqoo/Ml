"""
Custom transformer for customer segmentation feature engineering.
Creates new features from existing columns.
"""

from sklearn.base import BaseEstimator, TransformerMixin


class SegmentFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create new features from existing columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Average spend per order
        X["spend_per_order"] = X["total_spent"] / (X["number_of_orders"] + 1)
        # Ratio of total spend to annual income
        X["income_to_spend_ratio"] = X["total_spent"] / (X["annual_income"] + 1)
        # Engagement score: visits per day since last purchase
        X["engagement_score"] = X["website_visits"] / (X["days_since_last_purchase"] + 1)
        # Discount sensitivity relative to spending
        X["discount_sensitivity"] = X["discount_usage_pct"] / (X["total_spent"] + 1) * 1000
        # Flag for active customers (purchased within 30 days)
        X["is_active"] = (X["days_since_last_purchase"] < 30).astype(int)
        return X
