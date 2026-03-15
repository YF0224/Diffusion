"""
DDPM 扩散过程：加噪过程（forward_step）和逆向（reverse_step）都在这里实现。
噪声强度用 diffusion.schedule.NoiseSchedule（传参 linear/cosine/log），只提供系数。
"""
import torch

from diffusion.base import DiffusionProcess, ForwardResult
from diffusion.schedule import NoiseSchedule


class DDPMProcess(DiffusionProcess):
    """DDPM：schedule 只给噪声强度；加噪与逆向都在本类实现。"""

    def __init__(self, schedule: NoiseSchedule = None, *, schedule_type: str = "cosine", **schedule_kw):
        if schedule is not None:
            self._schedule = schedule
        else:
            self._schedule = NoiseSchedule(schedule_type=schedule_type, **schedule_kw)

    @property
    def T(self):
        return self._schedule.T

    @property
    def device(self):
        return self._schedule.device

    @property
    def alpha_bar(self):
        """训练时 Min-SNR 等会用，直接暴露 schedule 的 alpha_bar。"""
        return self._schedule.alpha_bar

    def forward_step(self, x0: torch.Tensor, t, noise=None) -> ForwardResult:
        """加噪过程：x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε，在 ddpm 里写。"""
        s = self._schedule  # shape: [T]
        if noise is None:
            noise = torch.randn_like(x0)
        a = s.sqrt_alpha_bar[t].view(-1, 1, 1, 1)  # shape: [B] -> [B, 1, 1, 1]
        sa = s.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)  # shape: [B] -> [B, 1, 1, 1]
        xt = a * x0 + sa * noise  # shape: [B, 1, 1, 1] * [B, C, H, W] + [B, 1, 1, 1] * [B, C, H, W] -> [B, C, H, W] + [B, C, H, W] -> [B, C, H, W]
        return ForwardResult(xt=xt, noise=noise)

    @torch.no_grad()
    def reverse_step(self, model, xt: torch.Tensor, t, **kwargs) -> torch.Tensor:
        """单步反向去噪；model(xt, t) 预测噪声 ε。"""
        s = self._schedule
        t_scalar = t if isinstance(t, int) else t.item()
        t_tensor = torch.full(
            (xt.size(0),), t_scalar, dtype=torch.long, device=s.device
        )
        pred_noise = model(xt, t_tensor)

        coef = s.beta[t_scalar] / s.sqrt_one_minus_alpha_bar[t_scalar]
        mean = s.sqrt_recip_alpha[t_scalar] * (xt - coef * pred_noise)

        if t_scalar == 0:
            return mean
        z = torch.randn_like(xt)
        return mean + torch.sqrt(s.posterior_variance[t_scalar]) * z

    @torch.no_grad()
    def sample_loop(self, model, shape, **kwargs) -> torch.Tensor:
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.T)):
            x = self.reverse_step(model, x, t)
        return x
