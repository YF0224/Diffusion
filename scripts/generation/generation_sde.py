"""
SDE 生成：加载 checkpoint，用 SDEProcess + SimpleUNet 采样并保存图片。
可与 configs/sde.yaml 配合，或直接传参。
"""
import os
import argparse

import torch
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in __import__("sys").path:
    __import__("sys").path.insert(0, _PROJECT_ROOT)

from configs import load_config
from utils.logging_utils import get_save_dir
from models import SimpleUNet
from diffusion.sde import SDEProcess


def load_sde_checkpoint(
    checkpoint_path: str,
    device: str = None,
    use_ema: bool = True,
):
    """加载 SDE checkpoint，返回 (model, config_dict)。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "config" in ckpt:
        cfg = ckpt["config"]
    else:
        cfg = load_config("sde.yaml")
    model_cfg = cfg.get("model", {})
    model = SimpleUNet(
        in_ch=model_cfg.get("in_ch", 3),
        base_ch=model_cfg.get("base_ch", 128),
        time_dim=model_cfg.get("time_dim", 256),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
    ).to(device)
    state = ckpt.get("ema" if use_ema else "model", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def sample_sde(
    checkpoint_path: str,
    output_dir: str = None,
    num_samples: int = 16,
    img_size: int = None,
    batch_size: int = None,
    dt_coef: float = None,
    device: str = None,
    use_ema: bool = True,
    seed: int = None,
):
    """用 VP-SDE 从 checkpoint 生成图片并保存到 output_dir。"""
    if seed is not None:
        torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, cfg = load_sde_checkpoint(checkpoint_path, device=device, use_ema=use_ema)
    data_cfg = cfg.get("data", {})
    diff_cfg = cfg.get("diffusion", {})
    inference_cfg = cfg.get("inference", {})
    if img_size is None:
        img_size = data_cfg.get("img_size", 32)
    if batch_size is None:
        batch_size = min(num_samples, 16)
    if dt_coef is None:
        dt_coef = inference_cfg.get("dt_coef", 1.0)

    diffusion = SDEProcess(
        schedule_type=diff_cfg.get("schedule_type", "cosine"),
        T=diff_cfg.get("T", 1000),
        beta_start=diff_cfg.get("beta_start", 1e-4),
        beta_end=diff_cfg.get("beta_end", 0.02),
        device=device,
    )

    if output_dir is None:
        output_dir = get_save_dir("sde", "samples/generated")
    os.makedirs(output_dir, exist_ok=True)

    num_batches = (num_samples + batch_size - 1) // batch_size
    start = 0
    with torch.no_grad():
        for i in range(num_batches):
            n = min(batch_size, num_samples - start)
            shape = (n, 3, img_size, img_size)
            samples = diffusion.sample_loop(
                model, shape=shape, dt_coef=dt_coef
            )
            samples = (samples.clamp(-1, 1) + 1) / 2
            for j in range(n):
                path = os.path.join(output_dir, f"sample_{start + j:05d}.png")
                x = samples[j].cpu().permute(1, 2, 0).numpy()
                x = (x * 255).clip(0, 255).astype("uint8")
                Image.fromarray(x).save(path)
            start += n
    print(f"Saved {num_samples} samples to {output_dir}", flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="SDE sampling")
    parser.add_argument("checkpoint", type=str, help="path to ckpt_epochXXXX.pt")
    parser.add_argument("--output_dir", "-o", type=str, default=None)
    parser.add_argument("--num_samples", "-n", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--dt_coef", type=float, default=None, help="逆向欧拉步长系数")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_ema", action="store_true", help="use model instead of ema")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    sample_sde(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        img_size=args.img_size,
        batch_size=args.batch_size,
        dt_coef=args.dt_coef,
        device=args.device,
        use_ema=not args.no_ema,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
