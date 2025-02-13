import os

import pandas as pd
from pydicom import dcmread
import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import (
    Resize,
    CenterCrop,
)
import torchvision.transforms.functional as F
from ..utils.data_utils import preprocess_scan, get_breast_mask, shift_image
import random


def get_patch(
    image,
    local_context_size=256*3,
    image_size=None,
    transform = None
):
    
    H, W = image.shape
    mask = get_breast_mask(image)
    image = Tensor(np.array([image, mask]))
    if image_size is not None:
        image = Resize(image_size, antialias=True).forward(image)

    mask_coordinates = np.where(image[1] == 1)  # find foreground indexes
    mask_coordinates = list(zip(mask_coordinates[0], mask_coordinates[1]))
    random_coordinate = random.choice(mask_coordinates)  # get random coordinate
    x_center = random_coordinate[0]
    y_center = random_coordinate[1]
    
    shifted = shift_image(image[0], x_center, y_center)
    shifted_tensor = torch.from_numpy(shifted).float().unsqueeze(0)
    local_context = CenterCrop(local_context_size).forward(shifted_tensor)

    local_context = transform(local_context[0].unsqueeze(0))
    shifted = transform(shifted_tensor[0].unsqueeze(0))
    image = transform(image[0].unsqueeze(0))

    x = torch.cat([local_context, shifted, image], dim=0) 
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
        view_position = int(self.data.loc[index, "view_position"] == 'CC')
        dcm = dcmread(image_path)
        
        image = dcm.pixel_array
        image = image / 65536
        image = image.astype(np.float32)
        image, _ = preprocess_scan(image)

        x = get_patch(image, image_size=None, transform=self.transform)
        return x
                                              

