"""!@file preprocessing.py
@brief Different components for preprocessing.

@details Different components for preprocessing. Most of them include a scaling component that
allows for correct distance computation. Includes removal of duplicates, outliers and near zero
variance features, as well as imputation of data and targets.

@author Created by I. Petrov on 15/12/2023
"""

from sklearn.base import BaseEstimator
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import scipy.stats as stats


class DuplicateRemover(BaseEstimator):
    """! A module that removes duplicate rows, i.e. rows with the same features."""

    def __init__(self):
        """! A module that removes duplicate rows, i.e. rows with the same features."""
        super(DuplicateRemover, self).__init__()

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        """Removes duplicate rows from sample X.

        @param X    The sample set.
        @param y    The associated labels/targets.
        @return     The sample with removed duplicates.
        """
        if isinstance(X, pd.DataFrame):
            data = X.values
        else:
            data = X.copy()
        rows_to_keep = np.ones(len(data), dtype=bool)
        rows_to_nullify = np.zeros(len(data), dtype=bool)
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                # Compare rows - if all values are equal
                # or the disagreement is because of a missing value
                if np.all((data[i] == data[j]) | np.isnan(data[i]) | np.isnan(data[j])):
                    # If the second row has an empty label, take the first
                    if y[j] is None:
                        rows_to_keep[j] = False
                    else:
                        # If labels agree, take one
                        if y[i] == y[j] or np.isnan(y[j]):
                            rows_to_keep[j] = False
                        elif np.isnan(y[i]):
                            rows_to_keep[i] = False
                        # Else, take one and set label to NaN
                        else:
                            rows_to_keep[j] = False
                            rows_to_nullify[i] = True
        print(f"Removed {np.count_nonzero(~rows_to_keep)} duplicate rows")
        return data[rows_to_keep], np.where(
            rows_to_nullify[rows_to_keep], np.nan, y[rows_to_keep]
        )

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class LabelImputer(BaseEstimator):
    """! Completes missing labels by inferring them from a Logistic Regression"""

    def __init__(self):
        """! Completes missing labels by inferring them from a Logistic Regression"""
        super(LabelImputer, self).__init__()
        self.scaler = StandardScaler()
        self.predictor = LogisticRegression(penalty="l1", solver="liblinear", C=2.0)

    def fit(self, X, y=None):
        """Trains the imputer model from a set (X, y)

        @param X    The sample set.
        @param y    The ground truth labels.
        """
        X_scaled = self.scaler.fit_transform(X)
        # Fit on the non-missing data
        self.predictor.fit(X_scaled[~np.isnan(y)], y[~np.isnan(y)])

    def transform(self, X, y):
        """Applies the imputer model on a set (X, y)

        @param X    The sample set.
        @param y    The ground truth labels.

        @returns    The imputed targets.
        """
        y_copy = y.copy()
        mask = np.isnan(y_copy)
        missing_pred = self.predictor.predict(self.scaler.transform(X[mask]))
        y_copy[mask] = missing_pred
        return y_copy.astype(np.int64)

    def fit_transform(self, X, y):
        """Trains and applies the imputer model on a set (X, y)

        @param X    The sample set.
        @param y    The ground truth labels.

        @returns    The imputed targets.
        """
        self.fit(X, y)
        return self.transform(X, y)


class DataImputer(BaseEstimator):
    """! Imputes the missing data using a KNNImputer.

    @param n_neighbors  The number of neighbouring data points to base the prediction on.
    """

    def __init__(self, n_neighbors=7):
        """! Imputes the missing data using a KNNImputer.

        @param n_neighbors  The number of neighbouring data points to base the prediction on.
        """
        super(DataImputer, self).__init__()
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, X, y=None):
        """Trains the imputer on the training set (X, y)

        @param X    The sample set.
        @param y    The ground truth labels."""
        X_scaled = self.scaler.fit_transform(X)
        self.imputer.fit(X_scaled)

    def transform(self, X, y=None):
        """Applies the imputer on the set (X, y)

        @param X    The sample set.
        @param y    The ground truth labels.
        @returns    The sample set with imputed values."""
        return self.scaler.inverse_transform(
            self.imputer.transform(self.scaler.transform(X))
        )

    def fit_transform(self, X, y=None):
        """Trains and applies the imputer on the set (X, y)

        @param X    The sample set.
        @param y    The ground truth labels.

        @returns    The sample set with imputed values."""
        self.fit(X, y)
        return self.transform(X, y)


class LOFRemove(BaseEstimator):
    """! Removes outlier values based on the distances between elements.

    @param threshold    The threshold above which we consider values outliers.
    @param n_neighbors  The number of nearest datapoints on which we base our detection.
    """

    def __init__(self, threshold=1.5, n_neighbors=7):
        """! Removes outlier values based on the distances between elements.

        @param threshold    The threshold above which we consider values outliers.
        @param n_neighbors  The number of nearest datapoints on which we base our detection.
        """
        super(LOFRemove, self).__init__()
        self.threshold = threshold
        self.scaler = StandardScaler()
        self.lof = LocalOutlierFactor(n_neighbors=n_neighbors)

    def fit_transform(self, X, y=None):
        """! Trains the detector and removes the outliers on the set (X, y)
        @param X    The sample set.
        @param y    The ground truth labels.

        @returns    The sample set with removed outliers.
        """
        X_rescaled = self.scaler.fit_transform(X)
        self.lof.fit_predict(X_rescaled)
        scores = -self.lof.negative_outlier_factor_
        is_inlier = scores < self.threshold

        if y is None:
            return X[is_inlier]
        else:
            return X[is_inlier], y[is_inlier]


class NZVarianceRemover(BaseEstimator):
    """! Removes features with Near-Zero Variance.

    @param frequency_threshold  If the most common value is met with
    a higher frequency than the threshold, the feature is considered with Near-Zero Variance.
    """

    def __init__(self, frequency_threshold=0.9):
        """! Removes features with Near-Zero Variance.

        @param frequency_threshold  If the most common value is met with
        a higher frequency than the threshold, the feature is considered with Near-Zero Variance.
        """
        super(NZVarianceRemover, self).__init__()
        self.frequency_threshold = frequency_threshold
        self.column_mask = None

    def fit(self, X, y=None):
        """! Discovers the estimated columns from the sample (X, y).

        @param X    The sample set.
        @param y    The ground truth labels.
        """
        if isinstance(X, pd.DataFrame):
            data = X.values
        else:
            data = X.copy()

        column_mask = np.zeros(data.shape[1], dtype=bool)
        for i in range(data.shape[1]):
            if stats.mode(data[:, i]).count / data.shape[0] < self.frequency_threshold:
                column_mask[i] = True
        self.column_mask = column_mask
        print(f"Found {np.count_nonzero(~column_mask)} columns with Near-Zero Variance")

    def transform(self, X, y=None):
        """! Removes the estimated columns from the sample (X, y).

        @param X    The sample set.
        @param y    The ground truth labels.

        @returns    The sample set with removed features."""
        if isinstance(X, pd.DataFrame):
            data = X.values
        else:
            data = X.copy()

        return data[:, self.column_mask].copy()

    def fit_transform(self, X, y=None):
        """! Discovers and removes the estimated columns from the sample (X, y).

        @param X    The sample set.
        @param y    The ground truth labels.

        @returns    The sample set with removed features."""
        self.fit(X, y)
        return self.transform(X, y)
