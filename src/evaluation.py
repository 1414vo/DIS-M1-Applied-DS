"""!@file evaluation.py
@brief Evaluation plots for classification and clustering.

@details Evaluation plots for classification and clustering.

@author Created by I. Petrov on 19/12/2023
"""
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import ClassifierMixin, ClusterMixin
from sklearn.ensemble import RandomForestClassifier
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


def optimize_random_forest(n_trees_options, X_train, y_train, X_test, y_test):
    """! Displays the OOB, train and test scores for a given dataset.

    @param n_trees_options   A list of options for the number of trees in the forest.
    @param X_train           The train sample set.
    @param y_train           The ground truth labels for the train set.
    @param X_test            The test sample set.
    @param y_test            The ground truth labels for the test set."""
    scores = []
    train_scores = []
    test_scores = []
    # Reoptimize for less features
    for n_trees in n_trees_options:
        classifier = RandomForestClassifier(
            random_state=42, oob_score=True, n_estimators=n_trees
        )
        classifier.fit(X_train, y_train)
        # Calculate training OOB score
        score = classifier.oob_score_
        scores.append(score)
        test_scores.append(classifier.score(X_test, y_test))
        train_scores.append(classifier.score(X_train, y_train))

    sns.set()
    plt.figure(dpi=300)
    plt.plot(n_trees_options, train_scores, label="Train accuracy")
    plt.plot(n_trees_options, scores, label="Train OOB accuracy")
    plt.plot(n_trees_options, test_scores, label="Test accuracy")
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy")
    plt.legend()
