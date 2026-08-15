"""
Custom transformer for customer churn feature engineering.
Creates new features from existing columns.
"""

from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create new features from existing columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Convert tenure to years
        X["tenure_years"] = X["tenure_months"] / 12
        # Monthly charge rate (spend per month of tenure)
        X["charge_per_month"] = X["monthly_charges"] / (X["tenure_months"] + 1)
        # Spend per support ticket
        X["spend_per_ticket"] = X["total_spend"] / (X["support_tickets"] + 1)
        # Flag for high monthly charges
        X["is_high_charges"] = (X["monthly_charges"] > 70).astype(int)
        # Flag for new customers (less than 6 months)
        X["is_new_customer"] = (X["tenure_months"] < 6).astype(int)
        return X
