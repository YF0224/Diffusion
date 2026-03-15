"""
配置加载：优先从 YAML 读，无则用默认 dict。
需要 PyYAML：pip install pyyaml
"""
import os

try:
    import yaml
except ImportError:
    yaml = None

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(_PROJECT_ROOT, "configs")


def load_config(path: str = None):
    """
    加载配置。path 为 configs/ 下的文件名（如 ddpm.yaml）或绝对路径。
    若 path 为空则返回 DDPM 默认配置。
    """
    default = _default_ddpm_config()
    if not path:
        return default
    if not os.path.isabs(path):
        path = os.path.join(CONFIGS_DIR, path)
    if not os.path.isfile(path):
        return default
    if yaml is None:
        return default
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _deep_merge(default, cfg or {})


def _deep_merge(base: dict, override: dict) -> dict:
    out = base.copy()
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _default_ddpm_config():
    return {
        "method": "ddpm",
        "data": {
            "dataset_name": "cifar100",
            "img_size": 32,
            "batch_size": 128,
            "num_workers": 4,
        },
        "diffusion": {
            "T": 1000,
            "schedule_type": "cosine",
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "device": "cuda",
        },
        "model": {
            "in_ch": 3,
            "base_ch": 128,
            "time_dim": 256,
            "num_res_blocks": 2,
        },
        "train": {
            "epochs": 500,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "use_amp": True,
            "warmup_epochs": 5,
            "use_ema": True,
            "ema_decay": 0.9999,
            "min_snr_gamma": 5.0,
            "grad_clip": 1.0,
            "sample_every": 20,
            "save_every": 50,
            "log_every": 20,
        },
        "output": {
            "save_dir": "ddpm",
            "subdir_checkpoints": "checkpoints",
            "subdir_samples": "samples",
        },
    }
