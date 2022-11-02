import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import utils
from torch.utils.data import DataLoader
from varSchedule import linear_beta_schedule
from torchvision import datasets
from torchvision.utils import save_image
from pathlib import Path
from network import Unet
from loss import p_losses


if __name__ == "__main__":
    timesteps = 200
    image_size = 28
    channels = 1
    batch_size = 128
    epochs = 5
    save_and_sample_every = 200  # 每更新x次保存一次中间训练图像结果

    results_folder = Path("results")  # 存放训练过程中保存的结果
    results_folder.mkdir(exist_ok=True)

    # 固定随机数种子
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    # 训练数据
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    training_data = datasets.FashionMNIST(
        root="../data/data_xuanyu",
        train=True,
        download=True,
        transform=transform,
    )
    dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    # 训练设备
    device = "cuda:2" if torch.cuda.is_available() else "cpu"

    # 模型相关
    model = Unet(dim=32, channels=channels, dim_mults=(1, 2, 4)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 开始训练
    model.train()
    for epoch in range(epochs):

        print("epoch: ", epoch)
        for step, (batch, _) in enumerate(dataloader):
            optimizer.zero_grad()

            batch = batch.to(device)
            t = torch.randint(0, timesteps, [batch.shape[0], ], device=device).long().to(device)

            loss = p_losses(model, batch, t,
                            sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                            loss_type="huber")
            loss.backward()
            optimizer.step()

            # 打印训练信息
            if step % 100 == 0:
                print("Step %d loss: %f" % (step, loss.item()))

            # 保存中间模型生成的图像
            if step != 0 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                all_images_list = utils.sample(
                    model, image_size,
                    betas, sqrt_one_minus_alphas_cumprod,
                    sqrt_recip_alphas, posterior_variance, timesteps,
                    batch_size=1, channels=channels
                )
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow=10)

    # 保存模型
    torch.save(model.state_dict(), "model_weight.pth")