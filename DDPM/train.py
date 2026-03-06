import os
import csv
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import SimpleUNet
from schedule import DDPMScheduler


# ==================================================
# Loss 曲线记录器
# ==================================================
class LossLogger:
    def __init__(self, save_dir):
        self.save_dir  = save_dir
        self.csv_path  = os.path.join(save_dir, "loss_log.csv")
        self.epoch_losses = []

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "avg_loss"])

    def log(self, epoch, avg_loss):
        self.epoch_losses.append((epoch, avg_loss))
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_loss:.6f}"])

    def plot(self):
        if not self.epoch_losses:
            return
        epochs = [r[0] for r in self.epoch_losses]
        losses = [r[1] for r in self.epoch_losses]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, losses, linewidth=1.5, color="#2563EB", label="Train Loss")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss (Min-SNR weighted MSE)", fontsize=12)
        ax.set_title("DDPM Training Loss — CIFAR-100", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(self.save_dir, "loss_curve.png"), dpi=150)
        plt.close(fig)


# ==================================================
# Warmup + Cosine LR 调度
# ==================================================
def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==================================================
# 训练主函数
# ==================================================
def train():
    # ---- 超参数 ----
    T             = 1000
    EPOCHS        = 500
    BATCH         = 128
    LR            = 1e-4
    IMG_SIZE      = 32
    SAMPLE_INT    = 20
    SAVE_INT      = 50
    SAVE_DIR      = "outputs"
    USE_EMA       = True
    EMA_DECAY     = 0.9999
    MIN_SNR_GAMMA = 5.0
    USE_AMP       = True
    WARMUP_EPOCHS = 5
    LOG_EVERY     = 20         # 每隔多少 step 打印一次进度

    os.makedirs(SAVE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    # ---- 数据集 ----
    print("Loading CIFAR-100 dataset...", flush=True)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    loader = DataLoader(
        dataset, batch_size=BATCH, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    steps_per_epoch = len(loader)
    total_steps     = EPOCHS * steps_per_epoch
    warmup_steps    = WARMUP_EPOCHS * steps_per_epoch
    print(f"Dataset ready. Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}", flush=True)

    # ---- 模型 ----
    print("Building model...", flush=True)
    model = SimpleUNet(in_ch=3, base_ch=128, time_dim=256, res_blocks=2).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model ready. Parameters: {n_params:.1f}M", flush=True)

    # ---- EMA ----
    print("Creating EMA model...", flush=True)
    # 直接复制 state_dict 而不是 deepcopy，避免卡住
    ema_model = SimpleUNet(in_ch=3, base_ch=128, time_dim=256, res_blocks=2).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    print("EMA model ready.", flush=True)

    # ---- 优化器 & 调度 ----
    print("Setting up optimizer...", flush=True)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    lr_scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
    scaler       = torch.amp.GradScaler("cuda", enabled=USE_AMP and device == "cuda")

    # ---- 扩散调度器 ----
    print("Building diffusion scheduler...", flush=True)
    diffusion = DDPMScheduler(T=T, schedule="cosine", device=device)

    # ---- Min-SNR-γ 权重 ----
    with torch.no_grad():
        snr = diffusion.alpha_bar / (1.0 - diffusion.alpha_bar)
        min_snr_weight = torch.minimum(snr, torch.tensor(MIN_SNR_GAMMA, device=device)) / snr

    # ---- Loss 记录器 ----
    logger = LossLogger(SAVE_DIR)

    print("=" * 60, flush=True)
    print("Start training!", flush=True)
    print("=" * 60, flush=True)

    global_step = 0

    # ---- 训练循环 ----
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss  = 0.0
        epoch_steps = 0

        for x0, _ in loader:
            x0 = x0.to(device)
            t  = torch.randint(0, T, (x0.size(0),), device=device)

            xt, noise = diffusion.q_sample(x0, t)

            with torch.amp.autocast("cuda", enabled=USE_AMP and device == "cuda"):
                pred_noise = model(xt, t)
                per_sample = F.mse_loss(pred_noise, noise, reduction="none").mean(dim=[1, 2, 3])
                w          = min_snr_weight[t]
                loss       = (per_sample * w).mean()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            # EMA 更新
            if USE_EMA:
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.data.mul_(EMA_DECAY).add_(p.data, alpha=1.0 - EMA_DECAY)

            total_loss  += loss.item()
            global_step += 1
            epoch_steps += 1

            # step 级进度
            if global_step % LOG_EVERY == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch:4d} [{epoch_steps:3d}/{steps_per_epoch}]"
                    f"  step {global_step:6d}"
                    f"  loss: {loss.item():.5f}"
                    f"  lr: {cur_lr:.2e}",
                    flush=True
                )

        avg_loss   = total_loss / steps_per_epoch
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:4d}/{EPOCHS}]  AvgLoss: {avg_loss:.5f}  LR: {current_lr:.2e}", flush=True)

        logger.log(epoch, avg_loss)
        logger.plot()

        # 定期生成样本
        if epoch % SAMPLE_INT == 0:
            sampler = ema_model if USE_EMA else model
            sampler.eval()
            with torch.no_grad():
                samples = diffusion.p_sample_loop(
                    sampler, shape=(16, 3, IMG_SIZE, IMG_SIZE)
                )
            samples = (samples.clamp(-1, 1) + 1) / 2
            save_image(
                samples,
                os.path.join(SAVE_DIR, f"sample_epoch{epoch:04d}.png"),
                nrow=4
            )
            print(f"  → Saved samples: sample_epoch{epoch:04d}.png", flush=True)

        # 定期保存权重
        if epoch % SAVE_INT == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "ema":   ema_model.state_dict() if USE_EMA else None,
                    "opt":   optimizer.state_dict(),
                },
                os.path.join(SAVE_DIR, f"ckpt_epoch{epoch:04d}.pt")
            )
            print(f"  → Saved checkpoint: ckpt_epoch{epoch:04d}.pt", flush=True)

    print("训练完成！", flush=True)
    print(f"Loss 曲线: {os.path.join(SAVE_DIR, 'loss_curve.png')}", flush=True)
    print(f"Loss CSV:  {os.path.join(SAVE_DIR, 'loss_log.csv')}", flush=True)
