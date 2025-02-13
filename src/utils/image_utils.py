import os
from pathlib import Path
from torchvision.utils import save_image
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from src.utils.data_utils import normalize
import torch
import numpy as np

def plt_img_patches(img, num_patches):
    micro_patch_size = img.shape[1] // num_patches
    patches = []
    fig, axs = plt.subplots(nrows=num_patches, ncols=num_patches, figsize=(20, 20))    
    for i in range(num_patches):
        for j in range(num_patches):
            patch = img[i*micro_patch_size:(i+1)*micro_patch_size, j*micro_patch_size:(j+1)*micro_patch_size]
            patches.append(patch)
            axs[i][j].imshow(patch, 'gray', vmin=img.min(), vmax=img.max())

    plt.axis('off')
    plt.show()
    
def plt_img_channels(img_channels):
    for (patch, local_context, i) in img_channels:
        _, axs = plt.subplots(nrows=1, ncols=3, figsize=(5, 20))    
        axs[0].imshow(patch, 'gray', vmin=0, vmax=1)
        axs[1].imshow(local_context, 'gray', vmin=0, vmax=1)
        axs[2].imshow(i, 'gray', vmin=0, vmax=1)
        plt.show()


def save_image_to_dir(image, img_path):
    image = normalize(image)
    save_image(torch.tensor(image), img_path)


def save_patches_to_dir(patches, patches_dir):
    patches = np.array(patches)
    patches_normalized = (patches - patches.min()) / (patches.max() - patches.min())

    for i, patch in enumerate(patches_normalized):
        save_image(torch.from_numpy(patch[0]), str(Path(patches_dir) / f'local_context-{i}.png'))
