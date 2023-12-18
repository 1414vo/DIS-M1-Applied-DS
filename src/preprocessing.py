from sklearn.base import BaseEstimator
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import scipy.stats as stats


class DuplicateRemover(BaseEstimator):
    def __init__(self):
        super(DuplicateRemover, self).__init__()

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            data = X.values
        else:
            data = X.copy()
        rows_to_keep = np.ones(len(data), dtype=bool)
        rows_to_nullify = np.zeros(len(data), dtype=bool)
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if np.all((data[i] == data[j]) | np.isnan(data[i]) | np.isnan(data[j])):
                    if y is None:
                        rows_to_keep[j] = False
                    else:
                        if y[i] == y[j] or np.isnan(y[j]):
                            rows_to_keep[j] = False
                        elif np.isnan(y[i]):
                            rows_to_keep[i] = False
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
    def __init__(self):
        super(LabelImputer, self).__init__()
        self.scaler = StandardScaler()
        self.predictor = LogisticRegression(penalty="l1", solver="liblinear", C=2.0)

    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        self.predictor.fit(X_scaled[~np.isnan(y)], y[~np.isnan(y)])

    def transform(self, X, y):
        y_copy = y.copy()
        mask = np.isnan(y_copy)
        missing_pred = self.predictor.predict(self.scaler.transform(X[mask]))
        y_copy[mask] = missing_pred
        return y_copy.astype(np.int64)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)


class DataImputer(BaseEstimator):
    def __init__(self, n_neighbors=7):
        super(DataImputer, self).__init__()
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        self.imputer.fit(X_scaled)

    def transform(self, X, y=None):
        return self.scaler.inverse_transform(
            self.imputer.transform(self.scaler.transform(X))
        )

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class LOFRemove(BaseEstimator):
    def __init__(self, threshold=1.5, n_neighbors=7):
        super(LOFRemove, self).__init__()
        self.threshold = threshold
        self.scaler = StandardScaler()
        self.lof = LocalOutlierFactor(n_neighbors=n_neighbors)

    def fit_transform(self, X, y=None):
        X_rescaled = self.scaler.fit_transform(X)
        self.lof.fit_predict(X_rescaled)
        scores = -self.lof.negative_outlier_factor_
        is_inlier = scores < self.threshold
        return X[is_inlier]


class NZVarianceRemover(BaseEstimator):
    def __init__(self, frequency_threshold=0.9):
        super(NZVarianceRemover, self).__init__()
        self.frequency_threshold = frequency_threshold

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            data = X.values
        else:
            data = X.copy()

        column_mask = np.zeros(data.shape[1], dtype=bool)
        for i in range(data.shape[1]):
            if stats.mode(data[:, i]).count / data.shape[0] < self.frequency_threshold:
                column_mask[i] = True
        return data[:, column_mask].copy()
