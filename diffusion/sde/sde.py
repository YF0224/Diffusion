"""
VP-SDE 扩散过程：schedule 只给噪声强度；加噪过程与 get_score_target 在此实现，逆向为欧拉。
"""
import torch

from diffusion.base import DiffusionProcess, ForwardResult
from diffusion.schedule import NoiseSchedule


class SDEProcess(DiffusionProcess):
    """VP-SDE：schedule 只给系数；加噪、score target、逆向都在本类写。"""

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

    def forward_step(self, x0, t, noise=None) -> ForwardResult:
        """加噪过程在此写：x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε。"""
        s = self._schedule
        if noise is None:
            noise = torch.randn_like(x0)
        a = s.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sa = s.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        xt = a * x0 + sa * noise
        return ForwardResult(xt=xt, noise=noise)

    def get_score_target(self, noise, t):
        """∇log p_t(x_t|x_0) = -ε / √(1-ᾱ_t)。"""
        s = self._schedule
        sa = s.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        return -noise / sa

    @torch.no_grad()
    def reverse_step(self, model, xt, t, dt_coef=1.0, **kwargs) -> torch.Tensor:
        """欧拉单步；model 预测 ε，换算为 score 后代入反向 SDE。"""
        s = self._schedule
        t_scalar = t if isinstance(t, int) else t.item()
        t_tensor = torch.full(
            (xt.size(0),), t_scalar, dtype=torch.long, device=s.device
        )
        pred_noise = model(xt, t_tensor)
        sa = s.sqrt_one_minus_alpha_bar[t_scalar].view(-1, 1, 1, 1)
        pred_score = -pred_noise / sa

        beta_t = s.beta[t_scalar].view(-1, 1, 1, 1)
        drift = -0.5 * beta_t * xt + beta_t * pred_score
        diffusion = torch.sqrt(beta_t)
        dt = -dt_coef / self.T
        z = torch.randn_like(xt)
        x_next = xt + drift * dt + diffusion * torch.sqrt(torch.tensor(abs(dt), device=s.device)) * z
        return x_next

    @torch.no_grad()
    def sample_loop(self, model, shape, dt_coef=1.0, **kwargs) -> torch.Tensor:
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.T)):
            x = self.reverse_step(model, x, t, dt_coef=dt_coef)
        return x
