"""!@file plotting.py
@brief Utilities for visualising clustering.

@details Utilities for visualising clustering. Points are overlaid and colored as specified,
with colors being automatically generated if needed. One method visualizes the clustering
not only through coloring the samples, but also through drawing a convex hull around the relevant
areas.

@author Created by I. Petrov on 15/12/2023
"""
from scipy import interpolate
from scipy.spatial import ConvexHull
from src.utils import color_list
import matplotlib.pyplot as plt
import numpy as np


def visualize_clusters(
    data, model, transform, n_clusters, hues=None, cmap=None, has_centroids=True
):
    """! Visualizes the clusters and their centroids if available. Also draws a convex hull around
    the clusters. Adapted from https://towardsdatascience.com/visualizing-clusters-with-pythons-matplolib-35ae03d87489

    @param data             The feature samples.
    @param model            The clustering algorithm. Is not necessary to be fitted.
    @param transform        The transformation that is applied for the data, i.e. PCA.
    @param n_clusters       The number of clusters.
    @param hues             A list of hues for how the points could be coloured.
    @param cmap             If hues, are specified a colormap is required.
    @param has_centroids    Whether the model produces centroids.
    """
    clusters = model.fit_predict(data)
    reduced_data = transform.transform(data)

    colors = color_list(n_clusters)
    point_colors = np.vectorize(lambda x: colors[x])(clusters)

    # Plot datapoints
    plt.figure(figsize=(8, 8))
    plt.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=point_colors if hues is None else hues,
        cmap=cmap,
        alpha=0.6,
        s=10,
    )
    # Plot centroids
    if has_centroids:
        centroids = model.cluster_centers_
        coords = transform.transform(centroids)
        cen_x = coords[:, 0]
        cen_y = coords[:, 1]
        plt.scatter(cen_x, cen_y, marker="^", c=colors, s=70, edgecolors="black")

    for i in np.unique(clusters):
        points = reduced_data[clusters == i]
        # Create the convex hull
        hull = ConvexHull(points)
        x_hull = np.append(points[hull.vertices, 0], points[hull.vertices, 0][0])
        y_hull = np.append(points[hull.vertices, 1], points[hull.vertices, 1][0])

        # Compute the distance between points on the hull
        dist = np.sqrt(
            (x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2
        )
        # Infer where the lines cross
        dist_along = np.concatenate(([0], dist.cumsum()))
        spline, _ = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0, per=1)

        interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
        interp_x, interp_y = interpolate.splev(interp_d, spline)

        # Color the area of the hull
        plt.fill(interp_x, interp_y, "--", c=colors[i], alpha=0.2)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")


def visualize_clusters_no_hull(data, model, transform, n_clusters):
    """! Visualizes the clusters and their centroids if available without drawing a convex hull.
    Adapted from https://towardsdatascience.com/visualizing-clusters-with-pythons-matplolib-35ae03d87489

    @param data             The feature samples.
    @param model            The clustering algorithm. Is not necessary to be fitted.
    @param transform        The transformation that is applied for the data, i.e. PCA.
    @param n_clusters       The number of clusters.
    """
    clusters = model.fit_predict(data)
    # Obtain model centers
    centroids = model.cluster_centers_
    # Apply transformation on data
    reduced_data = transform.transform(data)
    coords = transform.transform(centroids)
    cen_x = coords[:, 0]
    cen_y = coords[:, 1]

    colors = color_list(n_clusters)

    point_colors = np.vectorize(lambda x: colors[x])(clusters)
    plt.figure(figsize=(8, 8))
    # Plot datapoints
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=point_colors, alpha=0.6, s=10)
    # Plot centroids
    plt.scatter(cen_x, cen_y, marker="^", c=colors, s=70, edgecolors="black")

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
