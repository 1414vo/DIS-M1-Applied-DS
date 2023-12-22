"""!@file clustering.py
@brief Utilities for clustering comparisons.

@details Utilities for clustering comparisons. Displays the contingency matrix taking
different parameters.

@author Created by I. Petrov on 22/12/2023
"""
from sklearn.base import ClusterMixin
import pandas as pd
from sklearn.metrics.cluster import contingency_matrix


def compare_clusters(X, model_1: ClusterMixin, model_2: ClusterMixin):
    """! Prints the contingency matrix for a set X evaluated using 2 models.

    @param X        The sample set
    @param model_1  The first clustering model.
    @param model_2  The second clustering model."""
    pred_1 = model_1.fit_predict(X)
    pred_2 = model_2.fit_predict(X)

    compare_predictions(pred_1, pred_2)


def compare_predictions(pred_1, pred_2):
    """! Prints the contingency matrix 2 sets of predictions.

    @param pred_1   A set of clustering predictions
    @param pred_2   Another set of clustering predictions.

    @returns        A formatted Pandas DataFrame showing the contingency matrix."""
    ctable = contingency_matrix(pred_1, pred_2)
    ctable = pd.DataFrame(ctable)
    ctable.index = [f"Cluster {i + 1} \n(partition 1)" for i in range(len(ctable))]
    ctable.columns = [f"Cluster {i + 1} \n(partition 2)" for i in range(len(ctable))]
    return (
        ctable.style.format(precision=3, thousands=".", decimal=",")
        .format_index(str.upper, axis=1)
        .format_index(str.upper, axis=0)
        .highlight_max(axis=1, props="color:white; font-weight:bold")
        .set_properties(**{"text-align": "center"})
    )
