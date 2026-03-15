"""
时间步嵌入，供 UNet 等网络对条件 t 编码。
"""
import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码，用于时间 t。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half).float() / max(half - 1, 1)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        # t: (B,) long or float
        args = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
