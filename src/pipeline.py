"""!@file pipeline.py
@brief Modules for composing a classifier/preprocessing pipeline.

@details Modules for composing a classifier/preprocessing pipeline. The pipeline contains
components that remove Near-Zero Variance features, remove outlier datapoints using Local
Outlier Factor and rescaling through Z-transform. Wrappers for SVM or Random Forest were also
included.

@author Created by I. Petrov on 19/12/2023
"""

from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import LOFRemove, NZVarianceRemover


class PreprocessingPipeline(BaseEstimator, TransformerMixin):
    """! A pipeline used for removing outliers, non-zero variance features and scaling.

    @param frequency_threshold  The threshold for the Near-Zero Variance frequency,
    above which features will be removed. Refer to the NZRemover class for further details.
    @param outlier_threshold    The threshold for the Local Outlier Score above which points
    will be considered outliers. Refer to the LOFRemove class for further details.
    """

    def __init__(self, frequency_threshold, outlier_threshold):
        """! A pipeline used for removing outliers, non-zero variance features and scaling.

        @param frequency_threshold  The threshold for the Near-Zero Variance frequency,
        above which features will be removed. Refer to the NZRemover class for further details.
        @param outlier_threshold    The threshold for the Local Outlier Score above which points
        will be considered outliers. Refer to the LOFRemove class for further details.
        """
        super(PreprocessingPipeline, self).__init__()
        self.scaler = StandardScaler()
        self.lof_remover = LOFRemove(threshold=outlier_threshold)
        self.nzremover = NZVarianceRemover(frequency_threshold)

    def fit(self, X, y=None) -> ndarray:
        """! Build a preprocessing pipeline from the training set (X, y).

        @param X    The training input samples.
        @param y    The target values (class labels in classification)"""
        self.scaler.fit(X)
        self.lof_remover.fit(X)

    def transform(self, X, y=None) -> ndarray:
        """Apply preprocessing to the sample set X.

        @param X    The input samples.
        @param y    The target values (class labels in classification).
        @return     The preprocessed features and labels.
        """
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
        """Build a preprocessing pipeline from the training set (X, y) and
        applies preprocessing to the sample set X.

        @param X    The input samples.
        @param y    The target values (class labels in classification)
        @return     The preprocessed features and labels.
        """
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
        """! Undoes the scaling component of the . The remaining steps cannot
        be undone, as we lose data permanently."""
        return self.scaler.inverse_transform(X)


class ClassifierPipeline(BaseEstimator, ClassifierMixin):
    """! A pipeline used for classification, composed of a preprocessing stage and a classification model.
    The preprocessing stage is set to be a PreprocessingPipieline instance with configurable parameters.
    The classification model can be set to be anything, or a wrapper can be created.

    @param frequency_threshold  The threshold for the Near-Zero Variance frequency,
    above which features will be removed. Refer to the NZRemover class for further details.
    @param outlier_threshold    The threshold for the Local Outlier Score above which points
    will be considered outliers. Refer to the LOFRemove class for further details.
    @param classification_model The class of the classifier model. Ex: RandomForestClassifier
    """

    def __init__(
        self, frequency_threshold, outlier_threshold, classification_model, **kwargs
    ):
        """! A pipeline used for classification, composed of a preprocessing stage and a classification model.
        The preprocessing stage is set to be a PreprocessingPipieline instance with configurable parameters.
        The classification model can be set to be anything, or a wrapper can be created.

        @param frequency_threshold  The threshold for the Near-Zero Variance frequency,
        above which features will be removed. Refer to the NZRemover class for further details.
        @param outlier_threshold    The threshold for the Local Outlier Score above which points
        will be considered outliers. Refer to the LOFRemove class for further details.
        @param classification_model The class of the classifier model. Ex: RandomForestClassifier
        """
        super(ClassifierPipeline, self).__init__()
        self.frequency_threshold = frequency_threshold
        self.outlier_threshold = outlier_threshold
        self.model_type = classification_model
        self.model_args = kwargs

        self._generate_estimators()

    def _generate_estimators(self):
        """! Reinstantiates the pipeline with the latest parameters."""
        self.preprocessing = PreprocessingPipeline(
            self.frequency_threshold, self.outlier_threshold
        )
        self.model = self.model_type(**self.model_args)

    def fit(self, X, y=None) -> ndarray:
        """! Build a preprocessing pipeline and classifer from the training set (X, y).

        @param X    The training input samples.
        @param y    The target values (class labels in classification)"""
        if y is not None:
            self.classes_ = unique_labels(y)
        X_new, y_new = self.preprocessing.fit_transform(X, y)
        self.model.fit(X_new, y_new)

    def predict(self, X) -> ndarray:
        """Predict the class for sample set X

        @param X    The input samples.
        @return     The predicted classes.
        """
        X_new = self.preprocessing.transform(X)
        return self.model.predict(X_new)

    def fit_predict(self, X, y=None) -> ndarray:
        """Build a preprocessing pipeline and classifer from the training set (X, y) and
        predict the class for sample set X.

        @param X    The input samples.
        @param y    The target values (class labels in classification)
        @return     The predicted classes.
        """
        if y is not None:
            self.classes_ = unique_labels(y)
        X_new, y_new = self.preprocessing.fit_transform(X, y)
        return self.model.fit_predict(X_new, y_new)

    def predict_proba(self, X, y=None) -> ndarray:
        """Predict the probabilities per class class for sample set X.

        @param X    The input samples.
        @return     The predicted class probabilities.
        """
        X_new = self.preprocessing.transform(X)
        return self.model.predict_proba(X_new)

    def score(self, X, y) -> ndarray:
        """! Measures the score as defined by the classifier for a sample (X, y).

        @param X    The input samples.
        @param y    The target values (class labels in classification)
        @return     The score of the predictions."""
        X_new, y_new = self.preprocessing.transform(X, y)
        return self.model.score(X_new, y_new)

    def get_params(self, deep: bool = True) -> dict:
        """! Obtains the parameters for the pipeline.

        @param deep If True, will return the parameters for this estimator and contained subobjects that are estimators.

        @return     Parameter names mapped to their values."""
        params = {
            "frequency_threshold": self.frequency_threshold,
            "outlier_threshold": self.outlier_threshold,
        }
        params.update(self.model.get_params(deep))
        return params

    def set_params(self, **parameters):
        """! Changes the parameters for the pipeline. They should be given in the form of
        kwargs.

        @return     The reparametrized estimator."""

        parameters = parameters.copy()
        if "frequency_threshold" in parameters:
            self.frequency_threshold = parameters["frequency_threshold"]
            parameters.pop("frequency_threshold")
        if "outlier_threshold" in parameters:
            self.outlier_treshold = parameters["outlier_threshold"]
            parameters.pop("outlier_threshold")
        self.model_args = parameters
        self._generate_estimators()
        return self


class SVCPipeline(ClassifierPipeline):
    """! Wrapper for the Support Vector Machine Classifier Pipeline, preceded by a preprocessing stage.

    @param frequency_threshold  The threshold for the Near-Zero Variance frequency,
    above which features will be removed. Refer to the NZRemover class for further details.
    @param outlier_threshold    The threshold for the Local Outlier Score above which points
    will be considered outliers. Refer to the LOFRemove class for further details."""

    def __init__(self, frequency_threshold, outlier_threshold, **kwargs):
        """! Wrapper for the Support Vector Machine Classifier Pipeline, preceded by a preprocessing stage.

        @param frequency_threshold  The threshold for the Near-Zero Variance frequency,
        above which features will be removed. Refer to the NZRemover class for further details.
        @param outlier_threshold    The threshold for the Local Outlier Score above which points
        will be considered outliers. Refer to the LOFRemove class for further details.
        """
        super(SVCPipeline, self).__init__(
            frequency_threshold, outlier_threshold, SVC, **kwargs
        )


class RFPipeline(ClassifierPipeline):
    """! Wrapper for the Random Forest Classifier Pipeline, preceded by a preprocessing stage.

    @param frequency_threshold  The threshold for the Near-Zero Variance frequency,
    above which features will be removed. Refer to the NZRemover class for further details.
    @param outlier_threshold    The threshold for the Local Outlier Score above which points
    will be considered outliers. Refer to the LOFRemove class for further details."""

    def __init__(self, frequency_threshold, outlier_threshold, **kwargs):
        """! Wrapper for the Random Forest Classifier Pipeline, preceded by a preprocessing stage.

        @param frequency_threshold  The threshold for the Near-Zero Variance frequency,
        above which features will be removed. Refer to the NZRemover class for further details.
        @param outlier_threshold    The threshold for the Local Outlier Score above which points
        will be considered outliers. Refer to the LOFRemove class for further details.
        """
        super(RFPipeline, self).__init__(
            frequency_threshold, outlier_threshold, RandomForestClassifier, **kwargs
        )
