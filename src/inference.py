import os
from pathlib import Path

import itertools
import torch
from models.ddpm import *
from models.unet import Unet
# from models.ddpm_classifier_free import Unet as Unet_class
from utils.image_utils import save_image_to_dir, save_patches_to_dir
from utils.model_utils import (load_model, load_classifier_free_model, generate_whole_image, create_lcl_ctx_channels, 
                               create_inputs, generate_patches, stitch_patches, create_patch_channels)
from config import IS_COND, OVERLAP, MID_IMAGE_SIZE, FINAL_IMAGE_SIZE, total_timesteps, WH_PATH, LC_PATH, PH_PATH, SAVE_DIR

def load_models():
    # if IS_COND:
    whole_image_model = load_model(WH_PATH, channels=1)
    #     img_class = 0
    # else:
    #     whole_image_model = load_model(WH_PATH, channels=1)
    #     img_class = ''

    local_context_model = load_model(LC_PATH, channels=3)
    patch_model = load_model(PH_PATH, channels=3)

    return whole_image_model, local_context_model, patch_model

def pipeline(whole_image_model, local_context_model, patch_model, sampling_timesteps, save_dir, img_class=''):

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lc_dir = os.path.join(save_dir, 'local_contexts')
    patches_dir = os.path.join(save_dir, 'patches')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(lc_dir):
        os.makedirs(lc_dir)
    if not os.path.exists(patches_dir):
        os.makedirs(patches_dir)

    print(f'start: {save_dir}')

    # generate small image 
    img = generate_whole_image(whole_image_model, device, batch_size=1, img_class=img_class, sampling_timesteps=sampling_timesteps)
    save_image_to_dir(img, str(Path(save_dir) / f'whole_small.png'))

    # generate local contexts and middle size image
    img_channels, patch_coords = create_lcl_ctx_channels(img, overlap=OVERLAP)
    inputs, black_idx = create_inputs(img, img_channels, patch_coords, mask_shape=MID_IMAGE_SIZE)
    local_contexts = generate_patches(local_context_model, inputs, black_idx, total_timesteps=total_timesteps, sampling_timesteps=sampling_timesteps, overlap=OVERLAP, eta=1.0, device=device)
    mid_img = stitch_patches(local_contexts, overlap=OVERLAP, final_shape=MID_IMAGE_SIZE)
    save_image_to_dir(mid_img, str(Path(save_dir) / f'whole_middle.png'))
    save_patches_to_dir(local_contexts, lc_dir)

    # generate patches and final image
    img_channels, patch_coords = create_patch_channels(torch.from_numpy(mid_img).unsqueeze(0), img, overlap=OVERLAP)
    inputs, black_idx = create_inputs(img, img_channels, patch_coords, mask_shape=FINAL_IMAGE_SIZE)
    patches = generate_patches(patch_model, inputs, black_idx, total_timesteps=total_timesteps, sampling_timesteps=sampling_timesteps, overlap=OVERLAP, eta=1.0, device=device)
    final_img = stitch_patches(patches, overlap=OVERLAP, final_shape=FINAL_IMAGE_SIZE)
    save_image_to_dir(final_img, str(Path(save_dir) / f'whole_large.png'))
    save_patches_to_dir(patches, patches_dir)

    print(f'done: {save_dir}')

if __name__ == "__main__":
    models = load_models()

    for i in itertools.count():
        for j in [150]:
            pipeline(*models, j, os.path.join(SAVE_DIR, str(j), f'{i+4000}'))

