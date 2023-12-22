from sklearn.base import ClusterMixin
import pandas as pd
from sklearn.metrics.cluster import contingency_matrix


def compare_clusters(X, model_1: ClusterMixin, model_2: ClusterMixin):
    pred_1 = model_1.fit_predict(X)
    pred_2 = model_2.fit_predict(X)

    compare_predictions(pred_1, pred_2)


def compare_predictions(pred_1, pred_2):
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
