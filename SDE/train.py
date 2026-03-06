import os
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model import ScoreUNet
from sde import VPSDE, sample


# ==================================================
# VP-SDE 训练：Denoising Score Matching
# 目标 score = -ε / √(1-ᾱ_t)，损失 MSE(s_θ(x_t,t), score)
# ==================================================
def train():
    T = 1000
    EPOCHS = 100
    BATCH = 128
    LR = 2e-4
    IMG_SIZE = 32
    SAMPLE_INT = 10
    SAVE_DIR = "outputs_sde"
    USE_EMA = True
    EMA_DECAY = 0.999
    MIN_SNR_GAMMA = 5.0
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("VP-SDE training (score matching)")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(
        dataset, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True
    )

    model = ScoreUNet(in_ch=3, base_ch=64, time_dim=128).to(device)
    ema_model = deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    sde = VPSDE(T=T, device=device)

    # Min-SNR-γ 权重（shape: [T]）
    with torch.no_grad():
        snr = sde.alpha_bar / (1.0 - sde.alpha_bar)
        gamma = torch.tensor(MIN_SNR_GAMMA, device=device)
        min_snr_weight = torch.minimum(snr, gamma) / snr

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for x0, _ in loader:
            x0 = x0.to(device)
            t = torch.randint(0, T, (x0.size(0),), device=device)

            xt, noise = sde.sde_sample(x0, t)

            pred_score = model(xt, t)

            # 训练更稳定：把 score 预测映射回噪声预测 ε_θ，再做噪声空间 MSE
            # score = -ε / √(1-ᾱ)  =>  ε = -score * √(1-ᾱ)
            sa = sde.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
            pred_noise = -pred_score * sa

            per_example = F.mse_loss(pred_noise, noise, reduction="none").mean(dim=[1, 2, 3])
            w = min_snr_weight[t]
            loss = (per_example * w).mean()

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

        if epoch % SAMPLE_INT == 0:
            sampler_model = ema_model if USE_EMA else model
            samples = sample(sde, sampler_model, (16, 3, IMG_SIZE, IMG_SIZE))
            samples = (samples.clamp(-1, 1) + 1) / 2
            save_image(samples, f"{SAVE_DIR}/sample_epoch{epoch:03d}.png", nrow=4)
            print(f"  → Saved samples to {SAVE_DIR}/sample_epoch{epoch:03d}.png")

        if epoch % 20 == 0:
            torch.save(model.state_dict(), f"{SAVE_DIR}/model_epoch{epoch:03d}.pt")
            if USE_EMA:
                torch.save(ema_model.state_dict(), f"{SAVE_DIR}/model_ema_epoch{epoch:03d}.pt")

    print("训练完成！")
