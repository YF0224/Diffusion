"""
训练用 UNet：输入 (x, t)，输出预测噪声 ε。
通用版：固定 2× 下/上采样，特征图随输入尺寸动态缩放，32/64/128/256 等任意分辨率直接用。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .time_embed import SinusoidalPosEmb


def _group_norm(ch: int, num_groups: int = 8) -> nn.GroupNorm:
    for g in range(min(num_groups, ch), 0, -1):
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)
        self.norm2 = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.time_proj(t_emb).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class LinearAttention(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        assert ch % heads == 0
        self.heads = heads
        self.norm = _group_norm(ch)
        self.to_qkv = nn.Conv2d(ch, ch * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(ch, ch, 1), _group_norm(ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        heads = self.heads
        head_dim = c // heads
        qkv = self.to_qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(b, heads, head_dim, h * w).softmax(dim=-1)
        k = k.view(b, heads, head_dim, h * w).softmax(dim=-2)
        v = v.view(b, heads, head_dim, h * w)
        ctx = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", ctx, q)
        out = out.reshape(b, c, h, w)
        return x + self.to_out(out)


# 默认深度：支持 32～256+，每层固定 2× 缩放
DEFAULT_CHANNEL_MULTS = (1, 2, 4, 8, 8, 8)  # 6 层，任意分辨率通用
DEFAULT_NUM_LEVELS = len(DEFAULT_CHANNEL_MULTS)


class SimpleUNet(nn.Module):
    """
    通用 UNet：固定 2× 下采样/上采样，不绑死分辨率。
    输入 x: (B, C, H, W)，t: (B,) long；输出 (B, C, H, W)。
    32/64/128/256 等都能直接用，特征图随 H,W 自动缩放。
    建议 H,W 为 2 的幂且 ≥ 32（内部 6 层，最小 32/2^5=1）。
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 128,
        time_dim: int = 256,
        num_res_blocks: int = 2,
        channel_mults: tuple = DEFAULT_CHANNEL_MULTS,
    ):
        super().__init__()
        self.chs = [base_ch * m for m in channel_mults]
        self.num_levels = len(self.chs)
        self.pool = nn.MaxPool2d(2)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
            nn.SiLU(),
        )

        # Encoder：每层固定 2× 下采样
        self.enc_blocks = nn.ModuleList()
        self.enc_attn = nn.ModuleList()
        in_c = in_ch
        for i, out_c in enumerate(self.chs):
            self.enc_blocks.append(self._make_res_stack(in_c, out_c, time_dim, num_res_blocks))
            self.enc_attn.append(LinearAttention(out_c, heads=4 if out_c <= base_ch * 2 else 8))
            in_c = out_c

        # Mid
        self.mid1 = self._make_res_stack(self.chs[-1], self.chs[-1], time_dim, num_res_blocks)
        self.mid_attn = LinearAttention(self.chs[-1], heads=8)
        self.mid2 = self._make_res_stack(self.chs[-1], self.chs[-1], time_dim, num_res_blocks)

        # Decoder：每层固定 2× 上采样
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        for i in range(self.num_levels - 1):
            skip_ch = self.chs[-2 - i]
            in_c = self.chs[-1 - i] + skip_ch
            self.ups.append(nn.ConvTranspose2d(self.chs[-1 - i], self.chs[-1 - i], 2, stride=2))
            self.dec_blocks.append(self._make_res_stack(in_c, skip_ch, time_dim, num_res_blocks))
            self.dec_attn.append(LinearAttention(skip_ch, heads=4 if skip_ch <= base_ch * 2 else 8))

        self.out_norm = _group_norm(self.chs[0])
        self.out = nn.Conv2d(self.chs[0], in_ch, 1)

    def _make_res_stack(self, in_ch: int, out_ch: int, time_dim: int, n: int) -> nn.Module:
        blocks = [ResBlock(in_ch, out_ch, time_dim)]
        for _ in range(n - 1):
            blocks.append(ResBlock(out_ch, out_ch, time_dim))
        return nn.ModuleList(blocks)

    def _forward_res_stack(self, x: torch.Tensor, t_emb: torch.Tensor, stack: nn.ModuleList) -> torch.Tensor:
        for blk in stack:
            x = blk(x, t_emb)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        skips = []

        h = x
        for i in range(self.num_levels):
            h = self._forward_res_stack(h, t_emb, self.enc_blocks[i])
            h = self.enc_attn[i](h)
            skips.append(h)
            if i < self.num_levels - 1:
                h = self.pool(h)

        h = self._forward_res_stack(h, t_emb, self.mid1)
        h = self.mid_attn(h)
        h = self._forward_res_stack(h, t_emb, self.mid2)

        for i in range(self.num_levels - 1):
            h = self.ups[i](h)
            h = torch.cat([h, skips[-2 - i]], dim=1)
            h = self._forward_res_stack(h, t_emb, self.dec_blocks[i])
            h = self.dec_attn[i](h)

        return self.out(F.silu(self.out_norm(h)))
