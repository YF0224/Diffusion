import math
import torch
import torch.nn.functional as F


# ==================================================
# DDPM 扩散调度器（Cosine Schedule）
# ==================================================
class DDPMScheduler:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02,
                 schedule="cosine", device="cpu"):
        self.T = T
        self.device = device

        if schedule == "cosine":
            self.beta = self._cosine_schedule(T).to(device)
        else:
            self.beta = torch.linspace(beta_start, beta_end, T).to(device)

        self.alpha     = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # 预计算常用量
        self.sqrt_alpha_bar           = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha         = torch.sqrt(1.0 / self.alpha)

        alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        # 后验方差 β̃_t（clamp 防止 t=0 时为 0）
        self.posterior_variance = (
            self.beta * (1.0 - alpha_bar_prev) / (1.0 - self.alpha_bar)
        ).clamp(min=1e-20)

    # --------------------------------------------------
    # Cosine schedule（Nichol & Dhariwal 2021）
    # --------------------------------------------------
    @staticmethod
    def _cosine_schedule(T, s=0.008):
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        beta = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return beta.clamp(0.0, 0.999).float()

    # --------------------------------------------------
    # q_sample：前向扩散  x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε
    # 返回 (x_t, noise)
    # --------------------------------------------------
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a  = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sa = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        return a * x0 + sa * noise, noise

    # --------------------------------------------------
    # p_sample：单步反向去噪（DDPM 采样）
    # x_{t-1} = (1/√α_t)(x_t − β_t/√(1−ᾱ_t)·ε_θ) + σ_t·z
    # --------------------------------------------------
    @torch.no_grad()
    def p_sample(self, model, xt, t_scalar):
        t_tensor = torch.full(
            (xt.size(0),), t_scalar, dtype=torch.long, device=self.device
        )
        pred_noise = model(xt, t_tensor)

        coef = self.beta[t_scalar] / self.sqrt_one_minus_alpha_bar[t_scalar]
        mean = self.sqrt_recip_alpha[t_scalar] * (xt - coef * pred_noise)

        if t_scalar == 0:
            return mean
        noise = torch.randn_like(xt)
        return mean + torch.sqrt(self.posterior_variance[t_scalar]) * noise

    # --------------------------------------------------
    # p_sample_loop：完整生成流程 x_T → x_0
    # --------------------------------------------------
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
        return x
    