# 扩散基类：前向/逆向抽象，供各子模块继承与复用（时间嵌入在 models 中）
from .base import ForwardResult, DiffusionProcess

__all__ = ["ForwardResult", "DiffusionProcess"]
