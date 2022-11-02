"""
可视化正扩散过程
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import q_sample
from varSchedule import linear_beta_schedule
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Lambda, ToPILImage


def get_noisy_image(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    # add noise
    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

    # turn back into PIL image
    noisy_image = reverse_transform(x_noisy.squeeze())

    return noisy_image


def plot(imgs, with_orig=False, image=None, row_title=None, **kwargs):
    # 检查 with_orig 和 image 的一致性
    if with_orig:
        assert image, "with_orig 和 image 不一致！"
    else:
        assert image is None, "with_orig 和 image 不一致！"

    # make a 2D grid even if there's just 1 row
    if not isinstance(imgs[0], list):
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig

    fig, axs = plt.subplots(num_rows, num_cols, squeeze=False, figsize=(50, 10))

    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **kwargs)

    if with_orig:
        axs[0, 0].set(title="Original image")
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


if __name__ == "__main__":
    # 固定随机数种子
    torch.manual_seed(0)

    ########################### 超参数 ###########################
    timesteps = 200

    # define betas
    betas = linear_beta_schedule(timesteps)

    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion p(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    ########################### 超参数 ###########################

    # 图像变换
    image_size = 128
    transform = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),  # turn into CHW with [0., 1.]
        Lambda(lambda t: 2 * t - 1),
    ])

    reverse_transform = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    # 主逻辑
    img_name = "cats.jpg"
    img = Image.open(img_name)
    x_start = transform(img).unsqueeze(0)
    print(x_start.shape)

    plot([get_noisy_image(x_start, torch.tensor([t]), sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
          for t in [0, 50, 100, 150, 199]])
    plt.savefig("forward_diffusion.jpg")