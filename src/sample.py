import os
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

    return patch.detach().numpy(), mask

src_path = '/home/skipina/mammo-diffusion/mambo/data/local_context_256/150'
dst_path = '/home/jovisic/projects/mambo/src/data/local_context_256'


for i, entity in enumerate(os.scandir(src_path)):
    image_path = os.path.join(entity.path, 'whole_large.png') 

    if not os.path.exists(image_path):
        continue

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image[20:, 20:]
    patch, mask = get_patch(image)
    
    #cv2.imwrite(os.path.join(dst_path, str(i) + '.png'), image) 
    #cv2.imwrite(os.path.join(dst_path, str(i) + '.png'), mask*255)
    cv2.imwrite(os.path.join(dst_path, str(i) + '.png'), patch)

