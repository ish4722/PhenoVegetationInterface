import json
from skimage.draw import polygon
from skimage import io
import os
import glob
from tqdm import tqdm
import numpy as np

def fill_mask(blobs,mask,label):
    for l in blobs:
        fill_row_coords, fill_col_coords = polygon(l[1], l[0],l[2])
        mask[fill_row_coords, fill_col_coords] = label


def convert_annotation_json_to_mask(path_to_annotation_json, path_to_masks_folder,classes,label_value,pixel_value):

    f = open(path_to_annotation_json)
    train = []
    objects = json.load(f)
    annotations = objects['shapes']
    h = objects['imageHeight']
    w = objects['imageWidth']
    mask = np.zeros((h, w)).astype('uint8')
    for annot in annotations:
        label= annot['label']
        points = annot['points']

        x_coord = []
        y_coord = []
        l = []
        for p in points:
            x_coord.append(int(p[0]))
            y_coord.append(int(p[1]))
        shape = (h, w)
        l.append(np.array(x_coord))
        l.append(np.array(y_coord))
        l.append(shape)
        
        if not label_value.get(label):
            label_value[label]=pixel_value[0]
            pixel_value[0]+=10

        if not classes.get(label):
            classes[label]=[l]
        else:
            classes[label].append(l)
            
    for label in classes:
        fill_mask(classes[label], mask,label_value[label])
        classes[label]=[]
        
    io.imsave(path_to_masks_folder, mask)

def get_classes(dataset_path):
    classes=dict()
    count=0
    pixel_value=[100]
    label_value=dict()
    for dirpath, dirname, filename in os.walk(dataset_path):
        path_to_annotation_json=glob.glob(dirpath+"/*.json")
        for json_file in path_to_annotation_json:
            path_to_mask_png=json_file[0:-5]+"_mask.png"
            convert_annotation_json_to_mask(json_file, path_to_mask_png,classes,label_value,pixel_value)
            count+=1

    assert count, "Dataset folder path does not contain any json mask file" 


    labels=[1]*len(label_value)

    for label, value in label_value.items():
        labels[(value-100)//10]=label

    class_list=''
    for clss in labels:
        class_list += clss+"$"
    return class_list[0:-1]