import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================================================
# 时间步正弦嵌入
# ==================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / max(half, 1)
        )
        args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)


# ==================================================
# UNet（支持 3 通道 CIFAR-100，32×32）
# ==================================================
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1  = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_linear = nn.Linear(time_dim, out_ch)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm1  = nn.GroupNorm(8, out_ch)
        self.norm2  = nn.GroupNorm(8, out_ch)

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_linear(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Encoder
        self.enc1 = ResBlock(in_ch,       base_ch,     time_dim)
        self.enc2 = ResBlock(base_ch,     base_ch * 2, time_dim)
        self.enc3 = ResBlock(base_ch * 2, base_ch * 4, time_dim)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, time_dim)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, time_dim)

        # Decoder
        self.up2  = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ResBlock(base_ch * 4, base_ch * 2, time_dim)   # cat skip

        self.up1  = nn.ConvTranspose2d(base_ch * 2, base_ch,     2, stride=2)
        self.dec1 = ResBlock(base_ch * 2, base_ch,     time_dim)   # cat skip

        self.out  = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)                  # (B, time_dim)

        # Encoder  [B,3,32,32] → [B,64,32,32] → [B,128,16,16] → [B,256,8,8]
        e1 = self.enc1(x,           t_emb)        # 32×32
        e2 = self.enc2(self.pool(e1), t_emb)      # 16×16
        e3 = self.enc3(self.pool(e2), t_emb)      #  8×8

        # Bottleneck
        m  = self.mid1(e3, t_emb)
        m  = self.mid2(m,  t_emb)

        # Decoder
        d2 = self.up2(m)                           # 16×16
        d2 = self.dec2(torch.cat([d2, e2], dim=1), t_emb)

        d1 = self.up1(d2)                          # 32×32
        d1 = self.dec1(torch.cat([d1, e1], dim=1), t_emb)

        return self.out(d1)
