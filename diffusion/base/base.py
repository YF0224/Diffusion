"""
前向 / 逆向 抽象接口。
各方法（DDPM、DDIM、SDE、Flow 等）继承 DiffusionProcess 或实现相同接口，
在 diffusion/<name>/ 中实现具体调度与单步公式，便于集成与复用。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


@dataclass
class ForwardResult:
    """前向一步的结果：加噪后的 xt 与训练所需信息（如 noise、α_bar 等）。"""
    xt: torch.Tensor
    noise: torch.Tensor
    # 可扩展：alpha_bar_t, sigma_t 等，供不同 loss 使用


class DiffusionProcess(ABC):
    """
    扩散过程抽象：约定前向与逆向的调用方式。
    各 method 子类实现 forward_step / reverse_step / sample_loop。
    """

    @property
    @abstractmethod
    def T(self):
        """总扩散步数（离散）或等价步数。"""
        pass

    @abstractmethod
    def forward_step(self, x0: torch.Tensor, t: torch.Tensor, noise=None) -> ForwardResult:
        """
        前向加噪: x0 -> xt。
        t: (B,) 或标量，步索引。
        返回 ForwardResult(xt=..., noise=...)。
        """
        pass

    @abstractmethod
    def reverse_step(self, model, xt: torch.Tensor, t, **kwargs) -> torch.Tensor:
        """
        逆向一步: xt -> x_prev。
        model: 去噪/score/速度网络，由各 method 定义输入输出。
        """
        pass

    def sample_loop(self, model, shape, **kwargs) -> torch.Tensor:
        """
        完整采样: 从噪声到 x0。
        默认实现：从 x_T ~ N(0,I) 开始，逐步 reverse_step。
        子类可重写（如 DDIM 子序列、ODE 求解器）。
        """
        raise NotImplementedError("Subclass should implement sample_loop.")
