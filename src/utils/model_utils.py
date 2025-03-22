import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from tqdm import tqdm

from config import ORIG_IMG_SIZE, PATCH_REAL_SIZE, IS_COND, LOCAL_CONTEXT_SIZE, MID_IMAGE_SIZE, LOCAL_CONTEXT_SCALE_FACTOR, image_size, PATCH_SCALE_FACTOR
from models.unet import Unet
from models.unet_classifier_free import Unet as Unet_class
from models.ddpm_classifier_free import sample as sample_cond
from utils.data_utils import normalize, shift_image, keep_only_breast
import torchvision.transforms.functional as Fn
from torchvision.transforms import CenterCrop
from models.ddpm import *

def load_model(model_path, channels=3, dim=128, dim_mults=(1, 2, 2, 4, 4)):
    model_checkpoint = torch.load(model_path, map_location='cpu')
    model = Unet(
        dim=dim, 
        dim_mults=dim_mults, 
        channels=channels)
    model.load_state_dict((dict([(n.replace('module.', ''), p) for n, p in model_checkpoint['model_state'].items()])))
    return model


def load_classifier_free_model(model_path, channels=1, dim=128, dim_mults=(1, 2, 2, 4, 4), num_classes=3, cond_drop_prob=0.1):
    model_checkpoint = torch.load(model_path, map_location='cpu')
    model = Unet_class(
        dim=dim, 
        dim_mults=dim_mults, 
        channels=channels, 
        num_classes=num_classes, 
        cond_drop_prob=cond_drop_prob)
    model.load_state_dict((dict([(n.replace('module.', ''), p) for n, p in model_checkpoint['model_state'].items()])))
    return model


def generate_whole_image(model, device, batch_size=1, img_class=None, sampling_timesteps=150, cond_scale=5.0):
    model = model.to(device)
    model.eval()       
    if IS_COND:
        ctx = torch.tensor([img_class]).int().to(device)
        samples = sample_cond(model, image_size=(batch_size, 1, image_size, image_size), sampling_timesteps=sampling_timesteps, classes=ctx, cond_scale=cond_scale)
    else:
        samples = sample(model, image_size=(batch_size, 1, image_size, image_size), sampling_timesteps=sampling_timesteps)    
    model = model.to('cpu')
    return samples[0].cpu()


def create_lcl_ctx_channels(img, overlap=0.2):
    image_size = img.shape[1]
    
    num_patches = ORIG_IMG_SIZE // LOCAL_CONTEXT_SIZE
    micro_patch_size = image_size // num_patches  # size of one patch (without ovelap)
    stride = int(micro_patch_size * (1 - overlap))  # Pomeraj izmedju patcheva
    inputs = []
    patch_coords = []
    
    for i in range(0, image_size, stride):
        for j in range(0, image_size, stride):
            x_center = i + micro_patch_size // 2
            y_center = j + micro_patch_size // 2
            
            shifted = shift_image(img[0].numpy(), x_center, y_center)

            patch_coords.append((i*LOCAL_CONTEXT_SCALE_FACTOR, j*LOCAL_CONTEXT_SCALE_FACTOR, PATCH_REAL_SIZE, PATCH_REAL_SIZE))
            inputs.append((None, shifted, img))
    
    return inputs, patch_coords


def create_inputs(img, img_channels, patch_coords, mask_shape):
    upscaled_img = cv2.resize(img[0].numpy(), (mask_shape, mask_shape))
    _, breast_mask = keep_only_breast(upscaled_img)
    breast_mask = torch.from_numpy(breast_mask).float()
    inputs = []
    black_idx = []
    for idx, (patch, local_context, img) in enumerate(img_channels):
        x, y, h, w = patch_coords[idx]
        mask_crop = Fn.crop(breast_mask, x, y, h, w)
        if mask_crop.sum() == 0:
            black_idx.append(idx)
        patch = torch.randn_like(img)
        x = torch.cat([patch, torch.from_numpy(local_context).unsqueeze(0), img], dim=0)
        inputs.append(x)
    return inputs, black_idx


def get_overlapping_patch(input_tensor, overlap, local_contexts, patch_index, num_rows):
    img_shape = input_tensor.shape[2]
    mask = np.ones((1, img_shape, img_shape))
    mask_width = int(img_shape * overlap)
    foreground = np.zeros((1, img_shape, img_shape))
    i, j = patch_index // num_rows, patch_index % num_rows
    
    overlapping_patch = np.zeros((1, img_shape, img_shape))
    if i > 0:
        previous_patch = local_contexts[(i-1)*num_rows+j]
        if not (previous_patch.all() == 0):
            overlapping_patch[:, :mask_width, :] = previous_patch[0, -mask_width:, :] 
            mask[:, :mask_width, :] = 0
    if j > 0:
        previous_patch = local_contexts[i*num_rows+j-1]
        if not (previous_patch.all() == 0):
            overlapping_patch[:, :, :mask_width] = previous_patch[0, :, -mask_width:] 
            mask[:, :, :mask_width] = 0

    foreground = (1 - mask) * overlapping_patch + mask * input_tensor[0].numpy()
    return torch.from_numpy(mask), torch.from_numpy(foreground)


def prepare_input(in_img, mask, x_q):
    in_img[:, 1:, :, :] = x_q[:, 1:, :, :]
    in_img[:, 0, :, :] = x_q[:, 0, :, :].cpu()*(1 - mask) + in_img[:, 0, :, :].cpu()*mask  # keep the overlapping part
    in_img = in_img.to(torch.float32)
    return in_img


def generate_patches(model, inputs, black_idx, overlap, total_timesteps, sampling_timesteps, eta, device):
    model = model.to(device)
    model.eval()
    patches = []
    num_rows = int(np.sqrt(len(inputs)))
    for idx, input_tensor in enumerate(inputs):
        if idx in black_idx:
            patch = np.zeros((3, PATCH_REAL_SIZE, PATCH_REAL_SIZE)) + input_tensor[2].min().item()
            patches.append(patch)
            continue
        
        overlap_mask, overlapping_patch = get_overlapping_patch(input_tensor, overlap, patches, idx, num_rows)
        x = input_tensor.unsqueeze(0).to(torch.float32)
        patch = generate_one_patch(model, x, overlap_mask, overlapping_patch, device, total_timesteps, sampling_timesteps, eta)
    
        patches.append(patch[0].cpu().numpy())
    model = model.to('cpu')
    return patches


def generate_one_patch_ddpm(model, x, overlap_mask, overlapping_patch, device, timesteps=1000):
    b = x.shape[0]
    x = x.to(device)
    overlapping_patch = overlapping_patch.to(device)
    t = torch.full((b,), timesteps-1, device=device, dtype=torch.long)

    noise = torch.randn_like(x)  # use the same noise for every patch (because of the overlapping)
    noise[:, 1:, :, :] = x[:, 1:, :, :]

    x_start = x.clone()
    x_start[0, 0, :, :] = overlapping_patch
    x_q = q_sample(x_start, t, noise)
    in_img = x_q.clone()

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        x_q = q_sample(x_start, t, noise)
        in_img = prepare_input(in_img, overlap_mask, x_q)
        in_img = p_sample(model, in_img, t)

    return in_img

@torch.inference_mode()
def generate_one_patch_ddim(model, x, overlap_mask, overlapping_patch, device, total_timesteps=1000, sampling_timesteps=150, eta=0.7):
    b = x.shape[0]
    x = x.to(device)
    overlapping_patch = overlapping_patch.to(device)
    
    times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))  
    t = torch.full((b,), total_timesteps-1, device=device, dtype=torch.long)

    noise = torch.randn_like(x)
    noise[:, 1:, :, :] = x[:, 1:, :, :]

    x_t = x.clone()
    x_t[0, 0, :, :] = overlapping_patch
    x_q = q_sample(x_t, t, noise)
    img = prepare_input(noise, overlap_mask, x_q)

    for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
        time_cond = torch.full((b,), time, device=device, dtype=torch.long)
        
        x_q = q_sample(x_t, time_cond, noise)
        img = prepare_input(img, overlap_mask, x_q)

        pred_noise, x_start, *_ = model_predictions(model, img, time_cond, None, clip_x_start=False, rederive_pred_noise=True)

        if time_next < 0:
            img = x_start
            continue

        alpha = alphas_cumprod[time]
        alpha_next = alphas_cumprod[time_next]
            
        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()
        new_noise = torch.randn_like(img)

        img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * new_noise
        
    return img


def generate_one_patch(model, x, overlap_mask, overlapping_patch, device, total_timesteps=1000, sampling_timesteps=150, eta=0.7):
    if sampling_timesteps < total_timesteps:
        return generate_one_patch_ddim(model, x, overlap_mask, overlapping_patch, device, total_timesteps=timesteps, sampling_timesteps=sampling_timesteps, eta=eta)
    else:
        return generate_one_patch_ddpm(model, x, overlap_mask, overlapping_patch, device, timesteps=sampling_timesteps)

def stitch_patches(patches, overlap, final_shape):
    num_rows = int(np.sqrt(len(patches)))
    patches_arr = np.array(patches)
    min_value = patches_arr.min()

    final_img = np.zeros((num_rows*PATCH_REAL_SIZE, num_rows*PATCH_REAL_SIZE)) + min_value
    overlap_value = int(overlap * PATCH_REAL_SIZE)
    for idx, sample in enumerate(patches_arr):
        patch = sample[0]

        i = idx // num_rows
        j = idx % num_rows
        x = i * PATCH_REAL_SIZE - i * overlap_value
        y = j * PATCH_REAL_SIZE - j * overlap_value
        final_img[x:x+PATCH_REAL_SIZE, y:y+PATCH_REAL_SIZE] = patch

    final_img = final_img[:final_shape, :final_shape]
    return final_img


def create_patch_channels(lcl_ctx_img, img, overlap=0.125):
    image_size = lcl_ctx_img.shape[1]
    
    num_patches = ORIG_IMG_SIZE // PATCH_REAL_SIZE
    micro_patch_size = image_size // num_patches  # size of one patch (without ovelap)
    stride = int(micro_patch_size * (1 - overlap)) 
    inputs = []
    patch_coords = []

    for i in range(0, image_size, stride):
        for j in range(0, image_size, stride):
            x_center = i + micro_patch_size // 2
            y_center = j + micro_patch_size // 2
            
            shifted = shift_image(lcl_ctx_img[0].numpy(), x_center, y_center)
            shifted_tensor = torch.from_numpy(shifted)
            patch = CenterCrop(PATCH_REAL_SIZE).forward(shifted_tensor)

            patch_coords.append((i*PATCH_SCALE_FACTOR, j*PATCH_SCALE_FACTOR, PATCH_REAL_SIZE, PATCH_REAL_SIZE))
            inputs.append((None, patch.numpy(), img))
    
    return inputs, patch_coords
