import numpy as np
import cv2

def scale_and_extract_rois(points, labels, image_shape, scale_h, scale_w, roi_area):
    """Scale ROIs and extract from original image."""
    H, W = image_shape
    scaled_rois = []
    for i in range(len(labels)):
        cluster_points = points[labels == i]
        if len(cluster_points) < roi_area * 0.7:
            continue
        y1, x1 = cluster_points.min(axis=0)
        y2, x2 = cluster_points.max(axis=0)
        roi = (int(y1 * scale_h), int(x1 * scale_w), int(y2 * scale_h), int(x2 * scale_w))
        scaled_rois.append(roi)
    return scaled_rois
