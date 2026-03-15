"""
DDPM 训练入口：读 config，用 utils 数据/日志，diffusion.ddpm + models 训练。
在项目根目录执行：python -m scripts.train.train_ddpm [--config configs/ddpm.yaml]
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from configs import load_config
from utils import get_dataloader, get_save_dir, LossLogger
from models import SimpleUNet
from diffusion.ddpm import DDPMProcess


def _save_image_grid(tensor, path, nrow=4):
    """(B,C,H,W) in [0,1] 保存为网格图，不依赖 torchvision。"""
    from PIL import Image
    b, c, h, w = tensor.shape
    x = tensor.cpu().permute(0, 2, 3, 1).numpy()
    x = (x * 255).clip(0, 255).astype("uint8")
    ncol = min(nrow, b)
    nrows = (b + ncol - 1) // ncol
    out = Image.new("RGB", (ncol * w, nrows * h))
    for i in range(nrows):
        for j in range(ncol):
            idx = i * ncol + j
            if idx < b:
                out.paste(Image.fromarray(x[idx]), (j * w, i * h))
    out.save(path)


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(cfg_path: str = None):
    cfg = load_config(cfg_path or "ddpm.yaml")
    data_cfg = cfg["data"]
    diff_cfg = cfg["diffusion"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    out_cfg = cfg["output"]

    device = diff_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    diff_cfg = {**diff_cfg, "device": device}

    save_dir = get_save_dir(out_cfg["save_dir"])
    ckpt_dir = get_save_dir(out_cfg["save_dir"], out_cfg.get("subdir_checkpoints", "checkpoints"))
    sample_dir = get_save_dir(out_cfg["save_dir"], out_cfg.get("subdir_samples", "samples"))

    img_size = data_cfg["img_size"]
    T = diff_cfg["T"]
    batch_size = data_cfg["batch_size"]
    epochs = train_cfg["epochs"]
    num_workers = data_cfg.get("num_workers", 4)

    print("Loading data...", flush=True)
    loader = get_dataloader(
        dataset_name=data_cfg["dataset_name"],
        batch_size=batch_size,
        train=True,
        num_workers=num_workers,
        img_size=img_size,
    )
    steps_per_epoch = len(loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = train_cfg.get("warmup_epochs", 5) * steps_per_epoch

    print("Building model...", flush=True)
    model = SimpleUNet(
        in_ch=model_cfg.get("in_ch", 3),
        base_ch=model_cfg.get("base_ch", 128),
        time_dim=model_cfg.get("time_dim", 256),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f}M", flush=True)

    ema_model = None
    if train_cfg.get("use_ema", True):
        ema_model = SimpleUNet(
            in_ch=model_cfg.get("in_ch", 3),
            base_ch=model_cfg.get("base_ch", 128),
            time_dim=model_cfg.get("time_dim", 256),
            num_res_blocks=model_cfg.get("num_res_blocks", 2),
        ).to(device)
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    print("Building diffusion process...", flush=True)
    diffusion = DDPMProcess(
        schedule_type=diff_cfg.get("schedule_type", "cosine"),
        T=T,
        beta_start=diff_cfg.get("beta_start", 1e-4),
        beta_end=diff_cfg.get("beta_end", 0.02),
        device=device,
    )

    with torch.no_grad():
        snr = diffusion.alpha_bar / (1.0 - diffusion.alpha_bar)
        min_snr_weight = torch.minimum(
            snr, torch.tensor(train_cfg.get("min_snr_gamma", 5.0), device=device)
        ) / snr

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    lr_scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=train_cfg.get("use_amp", True) and device == "cuda"
    )
    logger = LossLogger(save_dir)

    sample_every = train_cfg.get("sample_every", 10)
    save_every = train_cfg.get("save_every", 10)
    log_every = train_cfg.get("log_every", 20)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    ema_decay = train_cfg.get("ema_decay", 0.9999)

    print("Start training.", flush=True)
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        epoch_steps = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False, dynamic_ncols=True)
        for x0, _ in pbar:
            x0 = x0.to(device)
            t = torch.randint(0, T, (x0.size(0),), device=device)
            res = diffusion.forward_step(x0, t)
            xt, noise = res.xt, res.noise

            with torch.amp.autocast("cuda", enabled=train_cfg.get("use_amp", True) and device == "cuda"):
                pred_noise = model(xt, t)
                per_sample = F.mse_loss(pred_noise, noise, reduction="none").mean(dim=[1, 2, 3])
                w = min_snr_weight[t]
                loss = (per_sample * w).mean()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            if ema_model is not None:
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

            total_loss += loss.item()
            global_step += 1
            epoch_steps += 1
            if epoch_steps % log_every == 0 or epoch_steps == steps_per_epoch:
                pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss/epoch_steps:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        avg_loss = total_loss / steps_per_epoch
        print(f"Epoch [{epoch:4d}/{epochs}]  AvgLoss: {avg_loss:.5f}", flush=True)
        logger.log(epoch, avg_loss)
        logger.plot(title="DDPM Training Loss", ylabel="Loss")

        if epoch % sample_every == 0:
            sampler = ema_model if ema_model is not None else model
            sampler.eval()
            with torch.no_grad():
                samples = diffusion.sample_loop(
                    sampler, shape=(16, 3, img_size, img_size)
                )
            samples = (samples.clamp(-1, 1) + 1) / 2
            _save_image_grid(
                samples,
                os.path.join(sample_dir, f"sample_epoch{epoch:04d}.png"),
                nrow=4,
            )
            print(f"  → Saved {sample_dir}/sample_epoch{epoch:04d}.png", flush=True)

        if epoch % save_every == 0:
            ckpt = {
                "epoch": epoch,
                "config": cfg,
                "model": model.state_dict(),
                "ema": ema_model.state_dict() if ema_model is not None else None,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(ckpt, os.path.join(ckpt_dir, f"ckpt_epoch{epoch:04d}.pt"))
            print(f"  → Saved {ckpt_dir}/ckpt_epoch{epoch:04d}.pt", flush=True)

    print("Training done.", flush=True)
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM")
    parser.add_argument("--config", type=str, default="ddpm.yaml", help="configs/ 下文件名或路径")
    args = parser.parse_args()
    train(args.config)
