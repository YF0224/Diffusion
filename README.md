# Diffusion

比较 DDPM、DDIM、SDE、Flow Matching 等扩散模型，统一 data / utils / output，各扩散过程在 `diffusion/` 下按需扩展。

---

## 数据准备

**不写统一下载脚本**，数据由自己准备、放到 `data/` 下即可。

- **CIFAR-10 / CIFAR-100**  
  - 训练时若检测到 `data/` 里没有对应数据，dataloader 会**自动下载**（无 torchvision 时用 pickle 版）。  
  - 也可先手动下载官方 [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)、[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)，解压到 `data/cifar-100-python/` 或 `data/cifar10/` 等（见 `utils/dataloader.py` 支持路径）；扩散逻辑在 `diffusion/`。
- **自定义数据集（如 256×256 图片）**  
  - 自己下载 / 爬取后，在 `data/` 下建一个子目录（如 `data/my_256/`），把图片放进去（可再分子目录当类别）。  
  - 在 config 里把 `data.dataset_name` 设为该目录名、`data.img_size` 设为 256 即可。

如需**显式**先下载 CIFAR 再训练，可运行：

```bash
python -m scripts.data.download_cifar
```

（会下载到 `data/`，之后训练直接用。）
