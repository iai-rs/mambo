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
from ..utils.data_utils import preprocess_scan, get_breast_mask
import random


def get_patch(
    image,
    patch_size=256,
    image_size=None,
    local_context_patches=3,
    transform = None
):

    mask = get_breast_mask(image)
    image = Tensor(np.array([image, mask]))
    if image_size is not None:
        image = Resize(image_size, antialias=True).forward(image)

    padded = Pad(patch_size).forward(image)
    local_context_size = patch_size * local_context_patches

    mask_coordinates = np.where(padded[1] == 1)  # find foreground indexes
    mask_coordinates = list(zip(mask_coordinates[0], mask_coordinates[1]))
    random_coordinate = random.choice(mask_coordinates)  # get random coordinate

    crop_x_center = random_coordinate[0]
    crop_y_center = random_coordinate[1]
    
    crop_params = (crop_x_center - local_context_size//2, crop_y_center - local_context_size // 2, local_context_size, local_context_size)
    local_context = F.crop(padded, *crop_params)
    patch = CenterCrop(patch_size).forward(local_context)

    patch = transform(patch[0].unsqueeze(0))
    local_context = transform(local_context[0].unsqueeze(0))
    image = transform(image[0].unsqueeze(0))

    x = torch.cat([patch, local_context, image], dim=0) 
    return x

class VINDR_Dataset(Dataset):
    def __init__(self, csv_path='data/train.csv', images_path='data/train_images', transform=None):
        self.csv_path = csv_path
        self.images_path = images_path
        df = pd.read_csv(self.csv_path)
        
#         df = df[df["finding_categories"] == r"['No Finding']"]
#         df = df[df['breast_birads'].isin(['BI-RADS 1', 'BI-RADS 2'])] #keep only the healthiest breasts
#         df = df[df["height"] == 3518]
        df = df[df['split'] == 'training']
        df = df.reset_index(drop=True)
        
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        study_id = self.data.loc[index, "study_id"]
        image_id = self.data.loc[index, "image_id"]
        image_path = os.path.join(
            self.images_path, str(study_id), str(image_id) + ".dicom"
        )

        dcm = dcmread(image_path)
        image = dcm.pixel_array
        image = image / 65536
        image = image.astype(np.float32)
        image, _ = preprocess_scan(image)

        x = get_patch(image, image_size=None, transform=self.transform)

        return x
        