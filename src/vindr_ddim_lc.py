import os
from pathlib import Path

import itertools
import torch
from models.ddpm import *
from models.unet import Unet
from models.ddpm_classifier_free import Unet as Unet_class
from utils.image_utils import save_image_to_dir, save_patches_to_dir
from utils.model_utils import (load_model, load_classifier_free_model, generate_whole_image, create_lcl_ctx_channels, 
                               create_inputs, generate_patches, stitch_patches, create_patch_channels, generate_one_patch)
from config import IS_COND, OVERLAP, MID_IMAGE_SIZE, FINAL_IMAGE_SIZE, total_timesteps, WH_PATH, LC_PATH, PH_PATH, SAVE_DIR
import einops

from datasets.vindr_local_context import VINDR_Dataset
import matplotlib.pyplot as plt

model = load_model(LC_PATH, channels=3)
model = model.to('cuda')

dataset = VINDR_Dataset()
for i, x in enumerate(dataset):
    try:
        x = dataset[i]
        save_dir = f'/lustre/ddim-gen-eta/vindr/lc/{i}/'
        os.makedirs(save_dir, exist_ok=True)
        plt.imsave(save_dir + 'orig.png', x[0], cmap='gray')

        for steps in [50, 100, 150, 200]:
            y = generate_one_patch(model, einops.rearrange(x, 'c h w -> 1 c h w'), torch.ones((256, 256)), torch.ones((256, 256)), 'cuda', sampling_timesteps=steps).cpu()
            patch = y[0, 0]
            plt.imsave(save_dir + f'{steps}.png', patch, cmap='gray')
    except:
        print('error with num ', i)
