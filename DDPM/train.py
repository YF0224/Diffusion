import os
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model import SimpleUNet
from schedule import DDPMScheduler


# ==================================================
# 训练主函数
# ==================================================
def train():
    # ---- 超参数 ----
    T          = 1000
    EPOCHS     = 100
    BATCH      = 128
    LR         = 2e-4
    IMG_SIZE   = 32
    SAMPLE_INT = 10      # 每隔多少 epoch 保存样本
    SAVE_DIR   = "outputs"
    USE_EMA    = True
    EMA_DECAY  = 0.999
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- 数据集：CIFAR-100 ----
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),   # [-1, 1]
    ])
    dataset = datasets.CIFAR100(root="./data", train=True,
                                download=True, transform=transform)
    loader  = DataLoader(dataset, batch_size=BATCH,
                         shuffle=True, num_workers=4, pin_memory=True)

    # ---- 模型 & 优化器 ----
    model     = SimpleUNet(in_ch=3, base_ch=64, time_dim=128).to(device)
    ema_model = deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    # ---- 扩散调度器 ----
    diffusion = DDPMScheduler(T=T, device=device)

    # ---- 训练循环 ----
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for x0, _ in loader:
            x0 = x0.to(device)
            t  = torch.randint(0, T, (x0.size(0),), device=device)

            # 前向扩散：q_sample
            xt, noise = diffusion.q_sample(x0, t)

            # 预测噪声
            pred_noise = model(xt, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if USE_EMA:
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.data.mul_(EMA_DECAY).add_(p.data, alpha=1.0 - EMA_DECAY)

            total_loss += loss.item()

        scheduler_lr.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch:3d}/{EPOCHS}]  Loss: {avg_loss:.4f}")

        # ---- 定期生成样本：p_sample_loop ----
        if epoch % SAMPLE_INT == 0:
            sampler_model = ema_model if USE_EMA else model
            sampler_model.eval()
            samples = diffusion.p_sample_loop(sampler_model, shape=(16, 3, IMG_SIZE, IMG_SIZE))
            samples = (samples.clamp(-1, 1) + 1) / 2   # [0, 1]
            save_image(samples, f"{SAVE_DIR}/sample_epoch{epoch:03d}.png", nrow=4)
            print(f"  → Saved samples to {SAVE_DIR}/sample_epoch{epoch:03d}.png")

        # ---- 保存模型权重 ----
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f"{SAVE_DIR}/model_epoch{epoch:03d}.pt")
            if USE_EMA:
                torch.save(ema_model.state_dict(), f"{SAVE_DIR}/model_ema_epoch{epoch:03d}.pt")

    print("训练完成！")
