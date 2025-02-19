import os

import pandas as pd
from pydicom import dcmread
import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import (
    Resize,
    Pad,
    CenterCrop,
)
import torchvision.transforms.functional as F
from utils.data_utils import preprocess_scan, get_breast_mask, shift_image
import random
import cv2


class RSNA_Dataset(Dataset):
    def __init__(self, csv_path='/home/milica.skipina.ivi/nj/mambo/src/rsna_train_with_shape.csv', images_path='/rsna/train_images', transform=None):
        
        self.csv_path = csv_path
        self.images_path = images_path
        df = pd.read_csv(self.csv_path)
        df = df[(df.machine_id == 49) & (df.height_processed == 3769) & (df.view.isin(['MLO', 'CC']))]
        df = df.reset_index(drop=True)
        
        self.data = df
        self.patches = pd.read_csv('/home/milica.skipina.ivi/nj/mambo/src/datasets/rsna_patches.csv')
        self.transform = transform

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        xmin = self.patches.loc[index, 'xmin']
        ymin = self.patches.loc[index, 'ymin']
        xmax = self.patches.loc[index, 'xmax']
        ymax = self.patches.loc[index, 'ymax']
        
        index = self.patches.loc[index, 'image_id']
        
        
        study_id = self.data.loc[index, "patient_id"]
        image_id = self.data.loc[index, "image_id"]
        
        image_path = os.path.join(
            self.images_path, str(study_id), str(image_id) + ".dcm"
        )

        dcm = dcmread(image_path)
        image = dcm.pixel_array
        image = image / 65536
        image = image.astype(np.float32)
        image, _ = preprocess_scan(image)
        
        x_center = (xmin + xmax) // 2
        y_center = (ymin + ymax) // 2
        
        shifted = shift_image(image, x_center, y_center)
        
        padded_image = np.pad(image, ((256, 256), (256, 256)), mode='constant', constant_values=np.min(image))

        lc = padded_image[xmin:xmax + 512, ymin:ymax + 512]
 
        image = cv2.resize(image, (256, 256))
        lc = cv2.resize(lc, (256, 256))
        shifted = cv2.resize(shifted, (256, 256))
         
        return torch.Tensor(((np.asarray([lc, shifted, image]) - 0.007) / 0.01).astype(np.float32))

