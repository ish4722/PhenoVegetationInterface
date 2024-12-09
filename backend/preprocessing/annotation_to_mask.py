import json
from skimage.draw import polygon
from skimage import io
import numpy as np
import os
import glob

def fill_mask(blobs, mask, label):
    for l in blobs:
        fill_row_coords, fill_col_coords = polygon(l[1], l[0], l[2])
        mask[fill_row_coords, fill_col_coords] = label

def generate_masks_from_annotations(dataset_path):
    """Convert JSON annotations to image masks."""
    label_value = {}
    pixel_value = [100]
    classes = {}

    for dirpath, _, _ in os.walk(dataset_path):
        json_files = glob.glob(dirpath + "/*.json")
        for json_file in json_files:
            mask_path = json_file[:-5] + "_mask.png"
            with open(json_file, 'r') as f:
                objects = json.load(f)
            annotations = objects['shapes']
            h, w = objects['imageHeight'], objects['imageWidth']
            mask = np.zeros((h, w), dtype='uint8')

            for annot in annotations:
                label = annot['label']
                points = annot['points']
                x_coord, y_coord = zip(*points)
                shape = (h, w)
                l = [np.array(x_coord), np.array(y_coord), shape]

                if label not in label_value:
                    label_value[label] = pixel_value[0]
                    pixel_value[0] += 100

                if label not in classes:
                    classes[label] = [l]
                else:
                    classes[label].append(l)

            for label in classes:
                fill_mask(classes[label], mask, label_value[label])
                classes[label] = []

            io.imsave(mask_path, mask)
