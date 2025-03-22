import os
from pathlib import Path

import torch
from models.ddpm import *
from models.unet import Unet
from models.unet_classifier_free import Unet as Unet_class
from utils.image_utils import save_image_to_dir, save_patches_to_dir
from utils.model_utils import (load_model, load_classifier_free_model, generate_whole_image, create_lcl_ctx_channels, 
                               create_inputs, generate_patches, stitch_patches, create_patch_channels)
from config import OVERLAP, MID_IMAGE_SIZE, FINAL_IMAGE_SIZE, total_timesteps


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sampling_timesteps=300

    # load model for generating whole images in resolution 256x256
    whole_img_model_path = '/lustre/mambo/models/artifacts/rsna_birads_whole:v462/model_204000.pt'
    whole_image_model = load_classifier_free_model(whole_img_model_path, channels=1, num_classes=3)

    # load model for generating local contexts (mid-resolution images)
    local_context_model_path = '/home/milica.skipina.ivi/projects/test_git/mambo/src/artifacts/rsna_3840_local_ctx:v62/model_94499.pt'
#     artifact = run.use_artifact('ivi-cvrs/mammo-diffusion/rsna_3840_local_ctx:v62', type='model')    
#     artifact_dir = artifact.download()
    local_context_model = load_model(local_context_model_path, channels=3)

    # load model for generating patches in native resolution
    patch_model_path = '/lustre/mambo/models/artifacts/rsna_3c_256:v70/model_131999.pt'
    patch_model = load_model(patch_model_path, channels=3)

    # MAMBO pipeline - generate new images in full resolution
    index=0
    num_images = 1000
    img_class=0
    base_dir = f'/lustre/mambo/data/rsna_birads/{img_class}/{sampling_timesteps}/'
    while index < num_images:
        save_dir = os.path.join(base_dir, str(index))
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
        img = generate_whole_image(whole_image_model, device, batch_size=1, sampling_timesteps=1000, img_class=img_class, cond_scale=6.0)
        save_image_to_dir(img, str(Path(save_dir) / f'whole_small.png'))

        # generate local contexts and middle size image
        img_channels, patch_coords = create_lcl_ctx_channels(img, overlap=OVERLAP)
        inputs, black_idx = create_inputs(img, img_channels, patch_coords, mask_shape=MID_IMAGE_SIZE)
        local_contexts = generate_patches(local_context_model, inputs, black_idx, total_timesteps=total_timesteps, sampling_timesteps=sampling_timesteps, eta=1.0, overlap=OVERLAP, device=device)
        mid_img = stitch_patches(local_contexts, overlap=OVERLAP, final_shape=MID_IMAGE_SIZE)
        save_image_to_dir(mid_img, str(Path(save_dir) / f'whole_middle.png'))
        save_patches_to_dir(local_contexts, lc_dir)

        # generate patches and final image
        img_channels, patch_coords = create_patch_channels(torch.from_numpy(mid_img).unsqueeze(0), img, overlap=OVERLAP)
        inputs, black_idx = create_inputs(img, img_channels, patch_coords, mask_shape=FINAL_IMAGE_SIZE)
        patches = generate_patches(patch_model, inputs, black_idx, total_timesteps=total_timesteps, sampling_timesteps=sampling_timesteps, eta=1.0, overlap=OVERLAP, device=device)
        final_img = stitch_patches(patches, overlap=OVERLAP, final_shape=FINAL_IMAGE_SIZE)
        save_image_to_dir(final_img, str(Path(save_dir) / f'whole_large.png'))
        save_patches_to_dir(patches, patches_dir)

        print(f'done: {save_dir}')
        index += 1