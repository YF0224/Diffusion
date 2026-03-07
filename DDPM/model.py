import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================
# 时间步正弦嵌入（freqs 缓存为 buffer，不重复计算）
# ==================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half).float()
            / max(half - 1, 1)
        )
        self.register_buffer("freqs", freqs)   # 只算一次，存为 buffer

    def forward(self, t):
        args = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


# ==================================================
# GroupNorm 工具（num_groups 在构造时确定，不在 forward 里算）
# ==================================================
def get_groupnorm(num_channels, preferred=8):
    for g in range(preferred, 0, -1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


# ==================================================
# ResBlock（AdaGN：用时间 embedding 做 scale+shift）
# ==================================================
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1     = get_groupnorm(in_ch)
        self.conv1     = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)
        self.norm2     = get_groupnorm(out_ch)
        self.conv2     = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip      = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.time_proj(t_emb).chunk(2, dim=1)  # t_emb 已在 time_mlp 末尾激活
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
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
# Self-Attention（Linear Attention 替换 Softmax Attention）
# 复杂度 O(HW) 而非 O((HW)^2)，32×32 feature map 快 ~16x
# ==================================================
class LinearAttention(nn.Module):
    """
    Efficient linear attention for spatial feature maps.
    复杂度 O(HW·C) 而非 O((HW)^2·C)
    """
    def __init__(self, ch, heads=4):
        super().__init__()
        assert ch % heads == 0
        self.heads   = heads
        self.norm    = get_groupnorm(ch)
        self.to_qkv  = nn.Conv2d(ch, ch * 3, 1, bias=False)
        self.to_out  = nn.Sequential(
            nn.Conv2d(ch, ch, 1),
            get_groupnorm(ch),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        heads = self.heads
        head_dim = c // heads

        qkv = self.to_qkv(self.norm(x))                   # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)                     # each (B, C, H, W)

        # reshape to (B, heads, head_dim, HW)
        q = q.view(b, heads, head_dim, h * w)
        k = k.view(b, heads, head_dim, h * w)
        v = v.view(b, heads, head_dim, h * w)

        # softmax over spatial dim for q/k（linear attention kernel）
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        # context: (B, heads, head_dim, head_dim)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out     = torch.einsum("bhde,bhdn->bhen", context, q)  # (B, heads, head_dim, HW)

        out = out.reshape(b, c, h, w)
        return x + self.to_out(out)


# ==================================================
# UNet（base_ch=128，LinearAttention，ResStack n=2）
# ==================================================
class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=128, time_dim=256, res_blocks=2):
        super().__init__()
        ch  = base_ch        # 128
        ch2 = base_ch * 2    # 256
        ch4 = base_ch * 4    # 512

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
            nn.SiLU(),  # 预激活，ResBlock 内直接用无需重复 silu
        )

        # ---- Encoder ----
        self.enc1      = ResStack(in_ch, ch,  time_dim, n=res_blocks)  # 32×32
        self.enc2      = ResStack(ch,    ch2, time_dim, n=res_blocks)  # 16×16
        self.enc2_attn = LinearAttention(ch2, heads=4)
        self.enc3      = ResStack(ch2,   ch4, time_dim, n=res_blocks)  #  8×8
        self.enc3_attn = LinearAttention(ch4, heads=8)
        self.pool      = nn.MaxPool2d(2)

        # ---- Bottleneck ----
        self.mid1      = ResStack(ch4, ch4, time_dim, n=res_blocks)
        self.mid_attn1 = LinearAttention(ch4, heads=8)
        self.mid2      = ResStack(ch4, ch4, time_dim, n=res_blocks)
        self.mid_attn2 = LinearAttention(ch4, heads=8)

        # ---- Decoder ----
        self.up2       = nn.ConvTranspose2d(ch4, ch2, 2, stride=2)
        self.dec2      = ResStack(ch4, ch2, time_dim, n=res_blocks)
        self.dec2_attn = LinearAttention(ch2, heads=4)

        self.up1       = nn.ConvTranspose2d(ch2, ch,  2, stride=2)
        self.dec1      = ResStack(ch2, ch,  time_dim, n=res_blocks)

        self.out_norm  = get_groupnorm(ch)
        self.out       = nn.Conv2d(ch, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        e1 = self.enc1(x,               t_emb)
        e2 = self.enc2(self.pool(e1),   t_emb)
        e2 = self.enc2_attn(e2)
        e3 = self.enc3(self.pool(e2),   t_emb)
        e3 = self.enc3_attn(e3)

        m  = self.mid1(e3,  t_emb)
        m  = self.mid_attn1(m)
        m  = self.mid2(m,   t_emb)
        m  = self.mid_attn2(m)

        d2 = self.up2(m)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), t_emb)
        d2 = self.dec2_attn(d2)

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), t_emb)

        return self.out(F.silu(self.out_norm(d1)))
    