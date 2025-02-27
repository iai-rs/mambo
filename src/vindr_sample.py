import os
import itertools
import uuid
import cv2
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
from utils.data_utils import preprocess_scan, get_breast_mask
import random
from torchvision.utils import save_image

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

    local_context_size = patch_size * local_context_patches
    margin = patch_size // 2  # Ensure we don't sample too close to the edge

    mask_coordinates = np.where(image[1] == 1)  # find foreground indexes
    mask_coordinates = list(zip(mask_coordinates[0], mask_coordinates[1]))

    # Filter out coordinates that are too close to the image edge
    valid_mask_coordinates = [
        (x, y) for x, y in mask_coordinates
        if margin <= x < image.shape[1] - margin and margin <= y < image.shape[2] - margin
    ]

    if not valid_mask_coordinates:
        raise ValueError("No valid mask coordinates found that satisfy margin constraints.")

    random_coordinate = random.choice(valid_mask_coordinates)  # get random coordinate

    crop_x_center, crop_y_center = random_coordinate

    crop_params = (
        crop_x_center - local_context_size // 2,
        crop_y_center - local_context_size // 2,
        local_context_size,
        local_context_size,
    )

    local_context = F.crop(image, *crop_params)
    patch = CenterCrop(patch_size).forward(local_context)[0]

    return patch.detach(), mask

class VINDR_Dataset(Dataset):
    def __init__(self, csv_path='/home/a800_9010/lun2/data/vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/finding_annotations.csv', images_path='/home/a800_9010/lun2/data/vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images', transform=None):
        self.csv_path = csv_path
        self.images_path = images_path
        df = pd.read_csv(self.csv_path)

        df = df[df["finding_categories"] == r"['No Finding']"]
        df = df[df["height"] == 3518]
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
        image = (image - image.min()) / (image.max() - image.min())

        patch, mask = get_patch(image, image_size=None, transform=self.transform)

        return patch

dst_path = '/home/jovisic/projects/mambo/src/data/vindr'

os.makedirs(dst_path, exist_ok=True)

for i in itertools.count():
    for j, patch in enumerate(VINDR_Dataset()):
        save_image(patch, os.path.join(dst_path, f'{i}_{j}.png'))
    
