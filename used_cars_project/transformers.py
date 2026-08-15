from sklearn.base import BaseEstimator, TransformerMixin


class CarFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for creating additional features for used-car data.

    Creates
    -------
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

        No parameters need to be learned from the data.
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

        X["car_age"] = self.current_year - X["year"]

        X["mileage_per_year"] = (
            X["mileage_km"] /
            X["car_age"].clip(lower=1)
        )

        return X
