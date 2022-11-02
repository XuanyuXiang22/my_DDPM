import torch
from tqdm import tqdm


def extract(a, t, x_shape):
    """allow us to extract the appropriate t index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape([batch_size] + [1, ] * (len(x_shape) - 1)).to(t.device)


def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    """forward diffusion (using the nice property)"""
    if noise is None:
        noise = torch.randn_like(x_start)  # [B, C, H, W]

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


@torch.no_grad()
def p_sample(model, x, t, t_index, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, postetior_variance):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # use model to predict the \mu_theta
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        postetior_variance_t = extract(postetior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # algorithm 2 line 4
        return model_mean + torch.sqrt(postetior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model, shape, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, postetior_variance, timesteps):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc="sampling loop time step", total=timesteps):
        img = p_sample(model, img, torch.full([b, ], i, device=device, dtype=torch.long), i,
                       betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, postetior_variance)
        if i % 10 == 0:
            imgs.append(img.cpu())

    return imgs


@torch.no_grad()
def sample(model, image_size, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance, timesteps, batch_size=16, channels=3):
    return p_sample_loop(model, [batch_size, channels, image_size, image_size],
                         betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance, timesteps)