import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F


# network helpers
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float32)


# variance schedules
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.00001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


timesteps = 1000

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)
# betas = sigmoid_beta_schedule(timesteps=timesteps)
# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# forward diffusion
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    # v2 - deleted the squared_root(alpha)
    # return x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise



# loss
def p_losses(model, x_start, t, noise=None, cond=None, loss_type="l2"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = model(x_noisy, t, cond)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

# sample
@torch.no_grad()
def p_sample(model, x, t, ctx=None, cond_scale = 6., rescaled_phi = 0.7):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, ctx) / sqrt_one_minus_alphas_cumprod_t
    )

    if t[0] == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    
    
# @torch.no_grad()
# def p_sample(self, x, t: int, classes, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = True):
#     model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, classes = classes, cond_scale = cond_scale, rescaled_phi = rescaled_phi, clip_denoised = clip_denoised)

#     noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
#     pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
#     return pred_img, x_start
    
    
# Algorithm 2 but save all images:
@torch.no_grad()
def p_sample_loop(model, shape, ctx=None, cond_scale = 6., rescaled_phi = 0.7):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, ctx, cond_scale, rescaled_phi)
        imgs.append(img.cpu().numpy())
    return img

@torch.no_grad()
def sample(model, image_size, ctx=None, cond_scale = 6., rescaled_phi = 0.7):
    return p_sample_loop(model, shape=image_size, ctx=ctx, cond_scale = cond_scale, rescaled_phi = rescaled_phi)


# test sample
@torch.no_grad()
def test_p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    epsilon_t = model(x, t)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    # v2 - deleted the squared_root(alpha)
    # model_mean = (
    #     x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    # )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        # deleted sigma_t*z - we probably don't need this non-deterministic behavior
        # return model_mean
        # alphas_cumprod_t = extract(
        #     alphas_cumprod, t, x.shape
        # )
        # x_t = 1/(alphas_cumprod_t) * (model_mean + torch.sqrt(posterior_variance_t)  - sqrt_one_minus_alphas_cumprod_t*model(x, t))
        # return x_t

        # remove noise
        # return model_mean + torch.sqrt(posterior_variance_t) * noise
        return model_mean

# Algorithm 2 but save all images:
@torch.no_grad()
def test_p_sample_loop(model, x_t, timesteps=1000):
    device = next(model.parameters()).device

    b = x_t.shape[0]
    # start from image embedding not from pure noise
    # img = z_v
    # return [x_t]
    x_ts = []
    x_0s = []
    z_v = x_t.clone()
    for i in reversed(range(0, timesteps)):
        x_t = test_p_sample(model, x_t, torch.full((b,), i, device=device, dtype=torch.long), i)
        if i == 0:
            x_0 = x_t
        else:
            t = torch.full((b,), i, device=device, dtype=torch.long)
            epsilon_t = model(z_v, t)
            sqrt_alphas_cumprod_t = extract(
                sqrt_alphas_cumprod, t, x_t.shape
            )
            sqrt_one_minus_alphas_cumprod_t = extract(
                sqrt_one_minus_alphas_cumprod, t, x_t.shape
            )
            x_0 = 1.0/sqrt_alphas_cumprod_t * (x_t - sqrt_one_minus_alphas_cumprod_t*epsilon_t)
        x_ts.append(x_t)
        x_0s.append(x_0)
    return x_ts

@torch.no_grad()
def test_sample(model, z_v, timesteps=1000):
    # z_v -> (batch_size, embedding size)
    # return [z_v]
    return test_p_sample_loop(model, z_v, timesteps=timesteps)
