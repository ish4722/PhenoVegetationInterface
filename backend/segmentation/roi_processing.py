import numpy as np
import cv2
import random


def ROI_Points(masks, images):
    """
    Process ROI points for multiple images and return individual components.
    
    Parameters:
        masks: List of masks (one mask per image).
        images: List of images corresponding to the masks.

    Returns:
        n_ROIs_list: List of number of ROIs for each image.
        points_list: List of points arrays for each image.
        ROI_area_list: List of ROI areas for each image.
        scale_h_list: List of scale factors for height for each image.
        scale_w_list: List of scale factors for width for each image.
    """
    n_ROIs_list = []
    points_list = []
    ROI_area_list = []
    scale_h_list = []
    scale_w_list = []

    for mask, image in zip(masks, images):
        mask = mask[..., 1].squeeze()
        H, W = image.shape

        points = []
        for i in range(255):
            for j in range(255):
                if mask[i, j] > 0.9:
                    points.append([i, j])
        points = np.array(points)

        scale_h = H / 255
        scale_w = W / 255
        threshold_area = 20000 * (H * W / 1200000)
        ROI_area = int(threshold_area / (scale_h * scale_w))
        n_ROIs = points.shape[0] // ROI_area

        n_ROIs_list.append(n_ROIs)
        points_list.append(points)
        ROI_area_list.append(ROI_area)
        scale_h_list.append(scale_h)
        scale_w_list.append(scale_w)

    return n_ROIs_list, points_list, ROI_area_list, scale_h_list, scale_w_list


def ROI_processing(labels, points, ROI_area, scale_h, scale_w, images):
    """
    Process Regions of Interest (ROIs) for multiple images.
    
    Parameters:
        labels: Array of labels for segmentation.
        points: Array of points associated with each label.
        ROI_area: Minimum area for ROI to be considered.
        scale_h: Scale factor for height.
        scale_w: Scale factor for width.
        images: List of images to process.

    Returns:
        Lists of red, green, and blue indices for all images.
    """
    all_red = []
    all_green = []
    all_blue = []

    for image in images:
        ROIs = []
        for i in range(len(labels)):
            if len(points[labels == i]) < ROI_area * 0.7:
                continue
            x1 = min(points[labels == i][:, 1])
            y1 = min(points[labels == i][:, 0])
            x2 = max(points[labels == i][:, 1])
            y2 = max(points[labels == i][:, 0])

            ROI_IMG = np.zeros((y2 - y1 + 1, x2 - x1 + 1))
            ROI_IMG[points[labels == i][:, 0] - y1, points[labels == i][:, 1] - x1] = 1

            h, w = ROI_IMG.shape
            h, w = int(scale_h * h), int(scale_w * w)
            ROI_image = cv2.resize(ROI_IMG, (w, h))
            ROI = []
            for i in range(h):
                for j in range(w):
                    if ROI_image[i, j] > 0.5:
                        ROI.append([i, j])
            ROI = np.array(ROI)
            ROI[:, 0] += int(y1 * scale_h)
            ROI[:, 1] += int(x1 * scale_w)

            ROIs.append(ROI)

        rois = random.sample(ROIs, min(4, len(ROIs)))  # Sample up to 4 ROIs
        red_indices = []
        green_indices = []
        blue_indices = []

        for r in rois:
            green_indices.extend(image[(r[:, 0]), (r[:, 1]), 1])
            blue_indices.extend(image[(r[:, 0]), (r[:, 1]), 0])
            red_indices.extend(image[(r[:, 0]), (r[:, 1]), 2])

        # Append results for this image
        all_red.append(np.array(red_indices))
        all_green.append(np.array(green_indices))
        all_blue.append(np.array(blue_indices))

    return all_red, all_green, all_blue





    
