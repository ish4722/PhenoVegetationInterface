import os
import glob
from tqdm import tqdm
import cv2
import numpy as np
import keras
import tensorflow

class Dataset:  
    def __init__(
            self, 
            dataset_path,
            classes,
            augmentation=None, 
            preprocessing=None,
    ):
        
        self.images_fps = []
        self.mask_fps=[]
        self.n_classes=len(classes.split('$'))

        for dirpath, dirnames, filenames in os.walk(dataset_path):
            mask_src=glob.glob(dirpath + '/*.PNG' )
            mask_src.sort()
            img_src=glob.glob(dirpath + '/*.JPG' )
            img_src+=glob.glob(dirpath + '/*.JPEG' )
            img_src.sort()
            if len(img_src):
                assert len(mask_src)!=0,dirpath+ ", does not contain any annotation mask file of image/images "
                if len(mask_src)!=1:
                    assert len(mask_src)==len(img_src),dirpath+ ", does not contains total number of annotation mask files({ml}) equal to total number of images({il}) ".format(ml = len(mask_src), il = len(img_src))
                    self.mask_fps+=mask_src 
                    
                if len(mask_src)==1:
                    self.mask_fps+=[mask_src[0]]*len(img_src)
            
                self.images_fps+=img_src
            else:
                assert len(mask_src)==0,dirpath+ ", does not contain image files of annotation mask file/files "

        assert (len(self.images_fps)!=0), dataset_path+", does not contain images"
                
        self.actaul_size=len(self.mask_fps)
        self.augmentation = augmentation            
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        j=i
        if i>=self.actaul_size:
            j=i-self.actaul_size
        image = cv2.imread(self.images_fps[j])  
        h,w,_=image.shape
        h=min(h,256) 
        w=min(w,256)     
        image=cv2.resize(image, (w, h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask1 = cv2.imread(self.mask_fps[j],0)   

        mask=[(mask1==((i+1)*10)) for i in range(self.n_classes)] 

        mask = np.stack(mask, axis=-1).astype('float')
        
        # add background to mask
        if mask.shape[-1] != 1:
                background = 1 - mask.sum(axis=-1, keepdims=True)
                mask = np.concatenate((mask, background), axis=-1)
                
        mask=cv2.resize(mask, (w, h))
        mask=1.0*(mask>0.9)
        
        # apply augmentations
        if i>=self.actaul_size:
            sample = self.augmentation(image=image, msk=mask)
            image, mask = sample['image'], sample['msk']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(img=image, msk=mask)
            image, mask = sample['img'], sample['msk']
            
        return image, mask
        
    def __len__(self):
        if self.augmentation:
            return self.actaul_size*2
        return self.actaul_size
       

class Dataloder(keras.utils.Sequence):

    def __init__(self,indexes, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = indexes

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[self.indexes[j]])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)  
        