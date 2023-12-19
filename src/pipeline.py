from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import LOFRemove, NZVarianceRemover


class PreprocessingPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, frequency_threshold, outlier_threshold):
        super(PreprocessingPipeline, self).__init__()
        self.scaler = StandardScaler()
        self.lof_remover = LOFRemove(threshold=outlier_threshold)
        self.nzremover = NZVarianceRemover(frequency_threshold)

    def fit(self, X, y=None) -> ndarray:
        self.scaler.fit(X)
        self.lof_remover.fit(X)

    def transform(self, X, y=None) -> ndarray:
        X_new = X.copy()
        if y is None:
            y_new = None
        else:
            y_new = y.copy()
        X_new = self.nzremover.transform(X_new)
        X_new = self.scaler.transform(X_new)
        if y is None:
            return X_new
        else:
            return X_new, y_new

    def fit_transform(self, X, y=None) -> ndarray:
        X_new = X.copy()
        if y is None:
            y_new = None
        else:
            y_new = y.copy()
        X_new = self.nzremover.fit_transform(X_new)
        X_new, y_new = self.lof_remover.fit_transform(X_new, y_new)
        X_new = self.scaler.fit_transform(X_new)
        return X_new, y_new

    def inverse_transform(self, X, y=None) -> ndarray:
        return self.scaler.inverse_transform(X)


class ClassifierPipeline(BaseEstimator, ClassifierMixin):
    def __init__(
        self, frequency_threshold, outlier_threshold, classification_model, **kwargs
    ):
        super(ClassifierPipeline, self).__init__()
        self.frequency_threshold = frequency_threshold
        self.outlier_threshold = outlier_threshold
        self.model_type = classification_model
        self.model_args = kwargs

        self._generate_estimators()

    def _generate_estimators(self):
        self.preprocessing = PreprocessingPipeline(
            self.frequency_threshold, self.outlier_threshold
        )
        self.model = self.model_type(**self.model_args)

    def fit(self, X, y=None) -> ndarray:
        if y is not None:
            self.classes_ = unique_labels(y)
        X_new, y_new = self.preprocessing.fit_transform(X, y)
        self.model.fit(X_new, y_new)

    def predict(self, X) -> ndarray:
        X_new = self.preprocessing.transform(X)
        return self.model.predict(X_new)

    def fit_predict(self, X, y=None) -> ndarray:
        if y is not None:
            self.classes_ = unique_labels(y)
        X_new, y_new = self.preprocessing.fit_transform(X, y)
        return self.model.fit_predict(X_new, y_new)

    def predict_proba(self, X, y=None) -> ndarray:
        X_new = self.preprocessing.transform(X)
        return self.model.predict_proba(X_new)

    def score(self, X, y) -> ndarray:
        X_new, y_new = self.preprocessing.transform(X, y)
        return self.model.score(X_new, y_new)

    def get_params(self, deep: bool = True) -> dict:
        params = {
            "frequency_threshold": self.frequency_threshold,
            "outlier_threshold": self.outlier_threshold,
        }
        params.update(self.model.get_params(deep))
        return params

    def set_params(self, **parameters):
        parameters = parameters.copy()
        if "frequency_threshold" in parameters:
            self.frequency_threshold = parameters["frequency_threshold"]
            parameters.pop("frequency_threshold")
        if "outlier_threshold" in parameters:
            self.outlier_treshold = parameters["outlier_threshold"]
            parameters.pop("outlier_threshold")
        self.model.set_params(**parameters)
        self._generate_estimators()
        return self


class SVCPipeline(ClassifierPipeline):
    def __init__(self, frequency_threshold, outlier_threshold, **kwargs):
        super(SVCPipeline, self).__init__(
            frequency_threshold, outlier_threshold, SVC, **kwargs
        )


class RFPipeline(ClassifierPipeline):
    def __init__(self, frequency_threshold, outlier_threshold, **kwargs):
        super(RFPipeline, self).__init__(
            frequency_threshold, outlier_threshold, RandomForestClassifier, **kwargs
        )
