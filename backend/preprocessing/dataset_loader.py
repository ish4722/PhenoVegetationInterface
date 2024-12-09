import os
import cv2
import glob
import numpy as np
from keras.utils import Sequence

class Dataset(Sequence):
    def __init__(self, dataset_path, n_classes, augmentation=None):
        self.images_fps = []
        self.mask_fps = []
        self.n_classes = n_classes
        self.augmentation = augmentation

        for dirpath, _, _ in os.walk(dataset_path):
            mask_src = glob.glob(dirpath + '/*.png')
            mask_src.sort()
            img_src = glob.glob(dirpath + '/*.JPG') + glob.glob(dirpath + '/*.jpeg')
            img_src.sort()

            if img_src:
                if len(mask_src) != 1:
                    assert len(mask_src) == len(img_src), (
                        f"{dirpath} contains {len(mask_src)} masks but {len(img_src)} images."
                    )
                self.mask_fps += mask_src if len(mask_src) != 1 else [mask_src[0]] * len(img_src)
                self.images_fps += img_src

        self.actual_size = len(self.mask_fps)

    def __getitem__(self, i):
        j = i % self.actual_size
        image = cv2.imread(self.images_fps[j])
        h, w = 256, 256
        image = cv2.resize(image, (w, h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32') / 255.0

        mask = cv2.imread(self.mask_fps[j], 0)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = [(mask == (i + 1) * 100) for i in range(self.n_classes)]
        mask = np.stack(mask, axis=-1).astype('float32')

        background = (mask[..., 0] == 0).astype('float32')
        mask = np.concatenate([background[..., None], mask], axis=-1)

        return image, mask

    def __len__(self):
        return self.actual_size

class Dataloader(Sequence):
    def __init__(self, indexes, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = indexes
        self.on_epoch_end()

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = [self.dataset[idx] for idx in self.indexes[start:stop]]
        batch = tuple(np.stack(samples, axis=0) for samples in zip(*data))
        return batch

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
