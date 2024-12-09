import numpy as np
from sklearn.cluster import KMeans

def perform_clustering(points, roi_area, n_clusters):
    """Perform KMeans clustering."""
    k_means = KMeans(n_clusters=n_clusters, max_iter=300)
    k_means.fit(points)
    return k_means.cluster_centers_, k_means.labels_
