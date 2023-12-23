"""!@file evaluation.py
@brief Evaluation plots for classification and clustering.

@details Evaluation plots for classification and clustering.

@author Created by I. Petrov on 19/12/2023
"""
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import ClassifierMixin, ClusterMixin
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    silhouette_score,
    calinski_harabasz_score,
)


def confusion_matrix_plot(model: ClassifierMixin, X, y):
    """! Displays the confusion matrix for a classifier model on set (X, y)

    @param model    The fit classifier model.
    @param X        The sample set.
    @param y        The ground truth labels."""
    plt.figure(dpi=300)
    y_pred = model.predict(X)
    conf_mat = confusion_matrix(y_true=y, y_pred=y_pred)
    disp = ConfusionMatrixDisplay(conf_mat)
    disp.plot()


def evaluate_clustering(n_clusters_options, X, model: ClusterMixin):
    """! Displays clustering scores

    @param n_clusters_options   A list of options for the number of clusters.
    @param X                    The sample set.
    @param model                The clustering model."""
    silhouette_scores = []
    explained_variance_scores = []
    for n_clusters in n_clusters_options:
        model_instance = model(n_clusters=n_clusters)
        if "random_state" in model_instance.get_params():
            model_instance.set_params(random_state=42)

        predictions = model_instance.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels=predictions))
        explained_variance_scores.append(calinski_harabasz_score(X, labels=predictions))

    sns.set()
    plt.figure(figsize=(12, 6), dpi=300)

    # Display silhouette scores
    plt.subplot(1, 2, 1)
    plt.plot(n_clusters_options, silhouette_scores, label="Silhouette Scores")
    plt.xlabel("Number of clusters")
    plt.ylabel("Score")
    plt.legend()

    # Display Calinski-Harabasz scores
    plt.subplot(1, 2, 2)
    plt.plot(
        n_clusters_options, explained_variance_scores, label="Variance Ratio Criterion"
    )
    plt.xlabel("Number of clusters")
    plt.legend()
