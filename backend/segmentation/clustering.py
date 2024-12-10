import numpy as np
from sklearn.cluster import KMeans

def perform_clustering(points, roi_area, n_clusters):
    """Perform KMeans clustering."""
    k_means = KMeans(n_clusters=n_clusters, max_iter=300)
    k_means.fit(points)
    centroids = k_means.cluster_centers_
    labels= k_means.labels_

    return centroids, labels

def clustering_image(points, n_ROIs, labels):
    for i in range(n_ROIs):
        p=points[labels==i]
        clustering_image[p[:,0],p[:,1]]=255*(i+1)//(n_ROIs)