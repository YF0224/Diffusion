import torch
import torch.nn.functional as F
import math


# ==================================================
# VP-SDE（Variance Preserving SDE）
# 前向: dx = -0.5 β(t) x dt + √β(t) dW
# 与 DDPM 使用相同的离散调度，便于对比
# ==================================================
class VPSDE:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device

        # β cosine 调度（相比线性更平滑，常见于 improved DDPM）
        def cosine_schedule(T, s=0.008):
            steps = torch.arange(T + 1, device=device).float() / T
            f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
            alpha_bar = f / f[0]
            beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
            return beta.clamp(0, 0.999)

        self.beta = cosine_schedule(T).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # 预计算常用量
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alpha)
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        # 后验方差（反向采样用）
        self.posterior_variance = (
            self.beta * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )

    # --------------------------------------------------
    # 前向扩散采样: x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε
    # 与 DDPM 的 q_sample 相同
    # --------------------------------------------------
    def sde_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sa = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        xt = a * x0 + sa * noise
        return xt, noise

    # --------------------------------------------------
    # 真实 score: ∇_x log p_t(x_t|x_0) = -ε / √(1-ᾱ_t)
    # 训练时 target = get_score_target(noise, t)
    # --------------------------------------------------
    def get_score_target(self, noise, t):
        sa = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        return -noise / sa

    # --------------------------------------------------
    # 漂移 f(x,t) = -0.5 β(t) x（连续 VP-SDE）
    # --------------------------------------------------
    def drift(self, x, t_scalar):
        b = self.beta[t_scalar].view(-1, 1, 1, 1)
        return -0.5 * b * x

    # --------------------------------------------------
    # 扩散系数 g(t) = √β(t)
    # --------------------------------------------------
    def diffusion(self, t_scalar):
        return torch.sqrt(self.beta[t_scalar])

    # --------------------------------------------------
    # 单步反向采样（Langevin-style，与 DDPM 形式等价）
    # 反向 SDE: x_{t-1} = (1/√α_t)(x_t + β_t·s_θ) + σ_t·z
    # --------------------------------------------------
    @torch.no_grad()
    def reverse_step(self, model, xt, t_scalar):
        t_tensor = torch.full(
            (xt.size(0),), t_scalar, dtype=torch.long, device=self.device
        )
        pred_score = model(xt, t_tensor)

        coef = self.beta[t_scalar].view(-1, 1, 1, 1)
        mean = self.sqrt_recip_alpha[t_scalar].view(-1, 1, 1, 1) * (
            xt + coef * pred_score
        )

        if t_scalar == 0:
            return mean
        variance = self.posterior_variance[t_scalar]
        noise = torch.randn_like(xt)
        return mean + torch.sqrt(variance) * noise

    # --------------------------------------------------
    # 完整反向采样: x_T ~ N(0,I) → x_0
    # --------------------------------------------------
    @torch.no_grad()
    def sample_loop(self, model, shape):
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.T)):
            x = self.reverse_step(model, x, t)
        return x


# ==================================================
# 采样便捷函数（推理用）
# ==================================================
@torch.no_grad()
def sample(sde, model, shape, device=None):
    """从 x_T ~ N(0,I) 反向采样到 x_0。"""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    return sde.sample_loop(model, shape)


def load_and_sample(checkpoint_path, shape=(16, 3, 32, 32), T=1000, device=None):
    """加载 checkpoint 并生成样本。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sde = VPSDE(T=T, device=device)
    from model import ScoreUNet  # 延迟导入避免循环依赖
    model = ScoreUNet(in_ch=3, base_ch=64, time_dim=128).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    # 兼容：既支持直接 state_dict，也支持 {"ema": state_dict, ...}
    if isinstance(ckpt, dict) and "ema" in ckpt:
        ckpt = ckpt["ema"]
    model.load_state_dict(ckpt)
    return sample(sde, model, shape, device)
