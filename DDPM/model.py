import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            -math.log(10000)
            * torch.arange(half, device=t.device).float()
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


# ==================================================
# ResBlock（Pre-Norm 风格：Norm → Conv → 时间注入 → Norm → Conv → Skip）
# ==================================================
def get_groupnorm(num_channels, preferred=8):
    """选出能整除 num_channels 的最大 num_groups（不超过 preferred）"""
    for g in range(preferred, 0, -1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = get_groupnorm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)   # scale + shift
        self.norm2 = get_groupnorm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        # AdaGN: scale & shift from time embedding
        t_out = self.time_proj(F.silu(t_emb))          # (B, out_ch*2)
        scale, shift = t_out.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale.unsqueeze(-1).unsqueeze(-1)) \
                          + shift.unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


# ==================================================
# ResStack：n 个 ResBlock 串联
# ==================================================
class ResStack(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, n=2):
        super().__init__()
        blocks = [ResBlock(in_ch, out_ch, time_dim)]
        for _ in range(n - 1):
            blocks.append(ResBlock(out_ch, out_ch, time_dim))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, t_emb):
        for blk in self.blocks:
            x = blk(x, t_emb)
        return x


# ==================================================
# Self-Attention（带 Pre-Norm）
# ==================================================
class SelfAttention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        assert ch % heads == 0, f"ch={ch} must be divisible by heads={heads}"
        self.norm = get_groupnorm(ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x).view(b, c, h * w).permute(0, 2, 1)   # (B, HW, C)
        y, _ = self.attn(y, y, y, need_weights=False)
        y = y.permute(0, 2, 1).view(b, c, h, w)
        return x + y


# ==================================================
# UNet（base_ch=128，双层 Attention，ResStack n=2）
# 输入/输出：(B, 3, 32, 32)
# ==================================================
class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=128, time_dim=256, res_blocks=2):
        super().__init__()
        ch  = base_ch
        ch2 = base_ch * 2   # 256
        ch4 = base_ch * 4   # 512

        # 时间嵌入 MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # ---- Encoder ----
        self.enc1 = ResStack(in_ch, ch,  time_dim, n=res_blocks)   # 32×32
        self.enc2 = ResStack(ch,  ch2,   time_dim, n=res_blocks)   # 16×16
        self.enc2_attn = SelfAttention(ch2, heads=4)
        self.enc3 = ResStack(ch2, ch4,   time_dim, n=res_blocks)   #  8×8
        self.enc3_attn = SelfAttention(ch4, heads=8)
        self.pool = nn.MaxPool2d(2)

        # ---- Bottleneck ----
        self.mid1      = ResStack(ch4, ch4, time_dim, n=res_blocks)
        self.mid_attn1 = SelfAttention(ch4, heads=8)
        self.mid2      = ResStack(ch4, ch4, time_dim, n=res_blocks)
        self.mid_attn2 = SelfAttention(ch4, heads=8)

        # ---- Decoder ----
        self.up2       = nn.ConvTranspose2d(ch4, ch2, 2, stride=2)  # 8→16
        self.dec2      = ResStack(ch4, ch2, time_dim, n=res_blocks)  # cat skip → ch4
        self.dec2_attn = SelfAttention(ch2, heads=4)

        self.up1       = nn.ConvTranspose2d(ch2, ch,  2, stride=2)  # 16→32
        self.dec1      = ResStack(ch2, ch,  time_dim, n=res_blocks)  # cat skip → ch2

        # 输出头
        self.out_norm = get_groupnorm(ch)
        self.out      = nn.Conv2d(ch, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)                           # (B, time_dim)

        # Encoder
        e1 = self.enc1(x,              t_emb)              # (B, ch,  32, 32)
        e2 = self.enc2(self.pool(e1),  t_emb)              # (B, ch2, 16, 16)
        e2 = self.enc2_attn(e2)
        e3 = self.enc3(self.pool(e2),  t_emb)              # (B, ch4,  8,  8)
        e3 = self.enc3_attn(e3)

        # Bottleneck
        m = self.mid1(e3, t_emb)
        m = self.mid_attn1(m)
        m = self.mid2(m, t_emb)
        m = self.mid_attn2(m)

        # Decoder
        d2 = self.up2(m)                                   # (B, ch2, 16, 16)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), t_emb) # cat → ch4
        d2 = self.dec2_attn(d2)

        d1 = self.up1(d2)                                  # (B, ch,  32, 32)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), t_emb) # cat → ch2

        return self.out(F.silu(self.out_norm(d1)))
