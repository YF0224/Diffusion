"""
只定义噪声强度：线性、余弦、log 三种 β/α_bar 调度。
加噪过程（forward_step）在各子模块里写（如 ddpm/ddpm.py），这里只提供系数。
"""
import math
import torch
import torch.nn.functional as F

SCHEDULE_TYPES = ("linear", "cosine", "log")

class NoiseSchedule:
    """
    只定义噪声强度：β、α_bar 及由它们推出的系数（sqrt_alpha_bar、posterior_variance 等）。
    不包含加噪过程；加噪在 diffusion/ddpm、sde/ 等里用这些系数自己写。
    """

    def __init__(
        self,
        T: int = 1000,
        schedule_type: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        cosine_s: float = 0.008,
        device: str = "cpu",
        **kwargs,
    ):
        self.T = T
        self.schedule_type = schedule_type
        self.device = device

        self.beta = self.get_beta(T, schedule_type, beta_start, beta_end, cosine_s, device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alpha)

        alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            self.beta * (1.0 - alpha_bar_prev) / (1.0 - self.alpha_bar)
        ).clamp(min=1e-20)

    def linear_beta(self, T: int, beta_start: float = 1e-4, beta_end: float = 0.02, device="cpu") -> torch.Tensor:
        """线性 β 调度。"""
        return torch.linspace(beta_start, beta_end, T, device=device)

    def cosine_beta(self, T: int, s: float = 0.008, device="cpu") -> torch.Tensor:
        """余弦 α_bar 调度（Nichol & Dhariwal），反推 β。"""
        steps = torch.arange(T + 1, dtype=torch.float64, device=device)
        f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        beta = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return beta.clamp(0.0, 0.999).float()

    def log_beta(self, T: int, beta_start: float = 1e-4, beta_end: float = 0.02, device="cpu") -> torch.Tensor:
        """Log 调度：log 空间线性插值。"""
        log_start = math.log(beta_start)
        log_end = math.log(beta_end)
        t = torch.arange(1, T + 1, dtype=torch.float64, device=device)
        log_beta_t = log_start + (log_end - log_start) * (t - 1) / max(T - 1, 1)
        return torch.exp(log_beta_t).float().clamp(1e-6, 0.999)

    def get_beta(self, T: int, schedule_type: str = "cosine", beta_start: float = 1e-4,
                 beta_end: float = 0.02, cosine_s: float = 0.008, device="cpu") -> torch.Tensor:
        """按 schedule_type 返回长度为 T 的 β。"""
        if schedule_type == "linear":
            return self.linear_beta(T, beta_start, beta_end, device)
        if schedule_type == "cosine":
            return self.cosine_beta(T, cosine_s, device)
        if schedule_type == "log":
            return self.log_beta(T, beta_start, beta_end, device)
        raise ValueError(f"schedule_type 应为 {SCHEDULE_TYPES} 之一，得到 {schedule_type}")
