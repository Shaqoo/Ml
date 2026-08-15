import datetime
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create new features from existing columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Calculate house age from current year
        X["house_age"] = datetime.date.today().year - X["year_built"]
        # Ratio of bedrooms to bathrooms
        X["bed_bath_ratio"] = X["bedrooms"] / (X["bathrooms"] + 1)
        # Square footage per room
        X["sqft_per_room"] = X["sqft"] / (X["bedrooms"] + X["bathrooms"] + 1)
        # Flag for recently built houses
        X["is_new"] = (X["year_built"] >= 2015).astype(int)
        return X
