import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F
from config import total_timesteps as timesteps
from collections import namedtuple


ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# network helpers
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def identity(t, *args, **kwargs):
    return t

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


objective = 'pred_noise'
use_cfg_plus_plus = False
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


log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)


# calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min =1e-20))
posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)


def predict_start_from_noise(x_t, t, noise):
    return (
        extract(sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        extract(sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )

def predict_noise_from_start(x_t, t, x0):
    return (
        (extract(sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
        extract(sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    )

def predict_v(x_start, t, noise):
    return (
        extract(sqrt_alphas_cumprod, t, x_start.shape) * noise -
        extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
    )

def predict_start_from_v(x_t, t, v):
    return (
        extract(sqrt_alphas_cumprod, t, x_t.shape) * x_t -
        extract(sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
    )


def q_posterior(x_start, x_t, t):
    posterior_mean = (
        extract(posterior_mean_coef1, t, x_t.shape) * x_start +
        extract(posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_variance_t = extract(posterior_variance, t, x_t.shape)
    posterior_log_variance_clipped_t = extract(posterior_log_variance_clipped, t, x_t.shape)
    return posterior_mean, posterior_variance_t, posterior_log_variance_clipped_t


def model_predictions(model, x, t, classes, cond_scale = 6., rescaled_phi = 0.7, clip_x_start = False):
    model_output, model_output_null = model.forward_with_cond_scale(x, t, classes, cond_scale = cond_scale, rescaled_phi = rescaled_phi)
    maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

    if objective == 'pred_noise':
        pred_noise = model_output if not use_cfg_plus_plus else model_output_null

        x_start = predict_start_from_noise(x, t, model_output)
        x_start = maybe_clip(x_start)

    elif objective == 'pred_x0':
        x_start = model_output
        x_start = maybe_clip(x_start)
        x_start_for_pred_noise = x_start if not use_cfg_plus_plus else maybe_clip(model_output_null)

        pred_noise = predict_noise_from_start(x, t, x_start_for_pred_noise)

    elif objective == 'pred_v':
        v = model_output
        x_start = predict_start_from_v(x, t, v)
        x_start = maybe_clip(x_start)

        x_start_for_pred_noise = x_start
        if use_cfg_plus_plus:
            x_start_for_pred_noise = predict_start_from_v(x, t, model_output_null)
            x_start_for_pred_noise = maybe_clip(x_start_for_pred_noise)

        pred_noise = predict_noise_from_start(x, t, x_start_for_pred_noise)

    return ModelPrediction(pred_noise, x_start)


def p_mean_variance(model, x, t, classes, cond_scale, rescaled_phi, clip_denoised = False):
    preds = model_predictions(model, x, t, classes, cond_scale, rescaled_phi)
    x_start = preds.pred_x_start

    if clip_denoised:
        x_start.clamp_(-1., 1.)

    model_mean, posterior_variance, posterior_log_variance = q_posterior(x_start = x_start, x_t = x, t = t)
    return model_mean, posterior_variance, posterior_log_variance, x_start


@torch.no_grad()
def p_sample(model, x, t, classes, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = False):
    b, *_, device = *x.shape, x.device
    model_mean, _, model_log_variance, x_start = p_mean_variance(model=model, x = x, t = t, classes = classes, cond_scale = cond_scale, rescaled_phi = rescaled_phi, clip_denoised = clip_denoised)
    noise = torch.randn_like(x) if t[0] > 0 else 0. # no noise if t == 0
    pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
    return pred_img, x_start


@torch.no_grad()
def p_sample_loop(model, classes, shape, cond_scale = 6., rescaled_phi = 0.7, sampling_timesteps=1000):
    batch, device = shape[0], next(model.parameters()).device

    img = torch.randn(shape, device=device)

    x_start = None

    for i in tqdm(reversed(range(0, sampling_timesteps)), desc = 'sampling loop time step', total = sampling_timesteps):
        t = torch.full((batch,), i, device=device, dtype=torch.long)
        img, x_start = p_sample(model, img, t, classes, cond_scale, rescaled_phi)

    return img


# @torch.no_grad()
# def interpolate(model, x1, x2, classes, t = None, lam = 0.5):
#     b, *_, device = *x1.shape, x1.device
#     t = default(t, self.num_timesteps - 1)

#     assert x1.shape == x2.shape

#     t_batched = torch.stack([torch.tensor(t, device = device)] * b)
#     xt1, xt2 = map(lambda x: q_sample(x, t = t_batched), (x1, x2))

#     img = (1 - lam) * xt1 + lam * xt2

#     for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
#         img, _ = p_sample(model, img, i, classes)

#     return img


@torch.no_grad()
def ddim_sample(model, classes, shape, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = False, sampling_timesteps=150, eta=1.0):
    batch, device, total_timesteps = shape[0], next(model.parameters()).device, timesteps
    
    times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    img = torch.randn(shape, device = device)

    x_start = None

    for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
        pred_noise, x_start, *_ = model_predictions(model, img, time_cond, classes, cond_scale = cond_scale, rescaled_phi = rescaled_phi, clip_x_start = clip_denoised)

        if time_next < 0:
            img = x_start
            continue

        alpha = alphas_cumprod[time]
        alpha_next = alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(img)

        img = x_start * alpha_next.sqrt() + \
              c * pred_noise + \
              sigma * noise

    return img


@torch.no_grad()
def sample(model, image_size, classes, sampling_timesteps=1000, cond_scale = 6., rescaled_phi = 0.7):
    is_ddim_sampling = sampling_timesteps < timesteps
    sample_fn = p_sample_loop if not is_ddim_sampling else ddim_sample
    return sample_fn(model, shape=image_size, classes=classes, cond_scale=cond_scale, rescaled_phi=rescaled_phi, sampling_timesteps=sampling_timesteps)



def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
        noise[:, 1:, :, :] = x_start[:, 1:, :, :]
        
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
  
    temp = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    temp[:, 1:, :, :] = x_start[:, 1:, :, :]
    return temp


def p_losses(model, x_start, t, classes=None, noise = None, loss_type="l2"):    
    if noise is None:
        noise = torch.randn_like(x_start)
        noise[:, 1:, :, :] = x_start[:, 1:, :, :]

    # noise sample

    x = q_sample(x_start = x_start, t = t, noise = noise)

    # predict and take gradient step

    model_out = self.model(x, t, classes)

    if self.objective == 'pred_noise':
        target = noise
    elif self.objective == 'pred_x0':
        target = x_start
    elif self.objective == 'pred_v':
        v = self.predict_v(x_start, t, noise)
        target = v
    else:
        raise ValueError(f'unknown objective {self.objective}')
        
        
    if loss_type == 'l1':
        loss = F.l1_loss(model_out[:, 0, :, :], target[:, 0, :, :])
    elif loss_type == 'l2':
        loss = F.mse_loss(model_out[:, 0, :, :], target[:, 0, :, :],reduction = 'none')
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(model_out[:, 0, :, :], target[:, 0, :, :])
    else:
        raise NotImplementedError()
        
    return loss
