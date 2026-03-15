"""
DDIM 扩散过程：前向与 DDPM 相同；逆向为确定性，可用子序列加速采样。
噪声强度用 diffusion.schedule.NoiseSchedule，与 DDPM 共用系数。
"""
import torch

from diffusion.base import DiffusionProcess, ForwardResult
from diffusion.schedule import NoiseSchedule


class DDIMProcess(DiffusionProcess):
    """DDIM：与 DDPM 共用 schedule；逆向无随机性，支持 num_steps 子序列采样。"""

    def __init__(
        self,
        schedule: NoiseSchedule = None,
        *,
        schedule_type: str = "cosine",
        **schedule_kw,
    ):
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
        return self._schedule.alpha_bar

    def forward_step(self, x0: torch.Tensor, t, noise=None) -> ForwardResult:
        """加噪与 DDPM 相同：x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε。"""
        s = self._schedule
        if noise is None:
            noise = torch.randn_like(x0)
        a = s.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sa = s.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        xt = a * x0 + sa * noise
        return ForwardResult(xt=xt, noise=noise)

    @torch.no_grad()
    def reverse_step(
        self,
        model,
        xt: torch.Tensor,
        t: int,
        t_prev: int = None,
        eta: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        确定性逆向一步：x_t -> x_{t_prev}。
        eta=0 为纯 DDIM；eta=1 等价于 DDPM 单步。
        t_prev 为上一时间步索引，若为 None 则用 t-1。
        """
        s = self._schedule
        t_tensor = torch.full(
            (xt.size(0),), t, dtype=torch.long, device=s.device
        )
        pred_noise = model(xt, t_tensor)

        sqrt_alpha_bar_t = s.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = s.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        pred_x0 = (xt - sqrt_one_minus_alpha_bar_t * pred_noise) / (sqrt_alpha_bar_t.clamp(min=1e-8))

        if t_prev is None:
            t_prev = t - 1
        if t_prev < 0:
            return pred_x0

        sqrt_alpha_bar_prev = s.sqrt_alpha_bar[t_prev].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_prev = s.sqrt_one_minus_alpha_bar[t_prev].view(-1, 1, 1, 1)
        sigma_t = eta * torch.sqrt(
            (s.posterior_variance[t] * (1.0 - s.alpha_bar[t_prev]) / (1.0 - s.alpha_bar[t])).clamp(min=0)
        ).view(-1, 1, 1, 1)
        dir_xt = torch.sqrt(sqrt_one_minus_alpha_bar_prev ** 2 - sigma_t ** 2).clamp(min=0) * pred_noise
        x_prev = sqrt_alpha_bar_prev * pred_x0 + dir_xt + sigma_t * torch.randn_like(xt)
        return x_prev

    @torch.no_grad()
    def sample_loop(
        self,
        model,
        shape,
        num_steps: int = None,
        eta: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        采样：若 num_steps 给定则用子序列（如 50 步），否则用满 T 步。
        """
        if num_steps is None:
            num_steps = self.T
        # 从 T-1 到 0 均匀取 num_steps 个时间步
        step_indices = torch.linspace(
            self.T - 1, 0, num_steps, device=self.device
        ).long().tolist()
        step_indices = list(dict.fromkeys(step_indices))  # 去重保持顺序
        if step_indices[-1] != 0:
            step_indices.append(0)
        x = torch.randn(shape, device=self.device)
        for i in range(len(step_indices) - 1):
            t, t_prev = step_indices[i], step_indices[i + 1]
            x = self.reverse_step(model, x, t, t_prev=t_prev, eta=eta)
        return x
