from scipy import interpolate
from scipy.spatial import ConvexHull
from src.utils import color_list
import matplotlib.pyplot as plt
import numpy as np


def visualize_clusters(data, model, transform, n_clusters):
    clusters = model.predict(data)
    # Obtain model centers
    centroids = model.cluster_centers_
    reduced_data = transform.transform(data)
    coords = transform.transform(centroids)
    cen_x = coords[:, 0]
    cen_y = coords[:, 1]

    colors = color_list(n_clusters)
    point_colors = np.vectorize(lambda x: colors[x])(clusters)
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=point_colors, alpha=0.6, s=10)
    plt.scatter(cen_x, cen_y, marker="^", c=colors, s=70, edgecolors="black")

    for i in np.unique(clusters):
        points = reduced_data[clusters == i]
        hull = ConvexHull(points)
        x_hull = np.append(points[hull.vertices, 0], points[hull.vertices, 0][0])
        y_hull = np.append(points[hull.vertices, 1], points[hull.vertices, 1][0])

        dist = np.sqrt(
            (x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2
        )
        dist_along = np.concatenate(([0], dist.cumsum()))
        spline, _ = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0, per=1)
        interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
        interp_x, interp_y = interpolate.splev(interp_d, spline)
        plt.fill(interp_x, interp_y, "--", c=colors[i], alpha=0.2)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")


def visualize_clusters_no_hull(data, model, transform, n_clusters):
    clusters = model.predict(data)
    # Obtain model centers
    centroids = model.cluster_centers_
    reduced_data = transform.transform(data)
    coords = transform.transform(centroids)
    cen_x = coords[:, 0]
    cen_y = coords[:, 1]

    colors = color_list(n_clusters)
    print(colors)
    point_colors = np.vectorize(lambda x: colors[x])(clusters)
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=point_colors, alpha=0.6, s=10)
    plt.scatter(cen_x, cen_y, marker="^", c=colors, s=70, edgecolors="black")

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
