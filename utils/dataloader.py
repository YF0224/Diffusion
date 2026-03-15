"""
统一数据接口：所有方法从此模块获取 DataLoader。
数据根目录为项目下的 data/。支持 CIFAR、自定义图片文件夹。
当 torch/torchvision 版本不兼容时，自动走「不依赖 torchvision」的加载路径。
"""
import os
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# 项目根目录（Diffusion/）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# 是否能用 torchvision（在 get_dataloader 里按需检测）
_USE_TORCHVISION = None


def _check_torchvision():
    global _USE_TORCHVISION
    if _USE_TORCHVISION is not None:
        return _USE_TORCHVISION
    try:
        from torchvision import datasets, transforms  # noqa: F401
        _USE_TORCHVISION = True
    except Exception:
        _USE_TORCHVISION = False
    return _USE_TORCHVISION


# ---------------------------------------------------------------------------
# 不依赖 torchvision：CIFAR 用 pickle 读，transform 用 PIL + 手写归一化
# ---------------------------------------------------------------------------

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"


def _download_cifar(root: str, url: str, filename: str):
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, filename)
    if os.path.isfile(path):
        return path
    print(f"Downloading {filename}...", flush=True)
    urllib.request.urlretrieve(url, path)
    return path


def _load_cifar_pickle(root: str, name: str, train: bool):
    """CIFAR 解压后的目录：cifar-10 为 data_batch_1..5 / test_batch，cifar-100 为 train / test。"""
    data_list, label_list = [], []
    if name == "cifar-10":
        files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"] if train else ["test_batch"]
    else:
        files = ["train"] if train else ["test"]
    for f in files:
        p = os.path.join(root, f)
        if not os.path.isfile(p):
            continue
        with open(p, "rb") as fp:
            d = pickle.load(fp, encoding="bytes")
        key_data = b"data" if b"data" in d else "data"
        key_labels = b"labels" if b"labels" in d else (b"fine_labels" if b"fine_labels" in d else "fine_labels")
        data_list.append(d[key_data])
        label_list.append(d[key_labels])
    if not data_list:
        return None, None
    data = np.concatenate(data_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    return data, labels


def _cifar_data_exists(root_data: str, name: str, train: bool) -> bool:
    """检查是否已有解压好的 CIFAR 数据（data 里已下载好的情况）。"""
    if name == "cifar-10":
        need = ["data_batch_1", "test_batch"] if train else ["test_batch"]
    else:
        need = ["train", "test"]
    for f in need:
        if os.path.isfile(os.path.join(root_data, f)):
            return True
    return False


class CIFARNoTv(Dataset):
    """不依赖 torchvision 的 CIFAR：用 pickle 读，PIL 做 resize，手写归一化。"""

    def __init__(self, root: str, name: str, train: bool, img_size: int = 32):
        self.train = train
        self.img_size = img_size
        self.name = name
        if name == "cifar-10":
            url, tarball = CIFAR10_URL, "cifar-10-python.tar.gz"
            self.subdir = "cifar-10-batches-py"
        else:
            url, tarball = CIFAR100_URL, "cifar-100-python.tar.gz"
            self.subdir = "cifar-100-python"
        root_data = os.path.join(root, self.subdir)
        # 若 data 里已有解压好的，直接用（支持 data/cifar100/... 或 data/cifar-100-python）
        if _cifar_data_exists(root_data, name, train):
            pass
        elif os.path.isdir(root) and _cifar_data_exists(root, name, train):
            root_data = root
        elif name == "cifar-100" and _cifar_data_exists(os.path.join(DATA_ROOT, "cifar-100-python"), name, train):
            root_data = os.path.join(DATA_ROOT, "cifar-100-python")
        elif name == "cifar-10" and _cifar_data_exists(os.path.join(DATA_ROOT, "cifar-10-batches-py"), name, train):
            root_data = os.path.join(DATA_ROOT, "cifar-10-batches-py")
        else:
            os.makedirs(root, exist_ok=True)
            path = _download_cifar(root, url, tarball)
            if not os.path.isdir(root_data):
                print(f"Extracting {tarball}...", flush=True)
                with tarfile.open(path, "r:gz") as tf:
                    tf.extractall(root)
        self.data, self.labels = _load_cifar_pickle(root_data, name, train)
        if self.data is None:
            raise FileNotFoundError(f"CIFAR 数据未找到: {root_data}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        # data: (N, 3072) RGB 逐行 32*32*3
        x = self.data[i].reshape(3, 32, 32).transpose(1, 2, 0)  # (32,32,3)
        img = Image.fromarray(x)
        if self.img_size != 32:
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        x = np.array(img)
        x = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0
        x = (x - 0.5) / 0.5
        if self.train and np.random.rand() > 0.5:
            x = x.flip(-1)
        return x, int(self.labels[i])


def _transform_pil_to_tensor(img_size: int, train: bool):
    """返回函数 f(pil_image) -> tensor [-1,1]，不依赖 torchvision。"""

    def _resize_and_normalize(img: Image.Image):
        img = img.resize((img_size, img_size), Image.BILINEAR)
        x = np.array(img)
        if x.ndim == 2:
            x = np.stack([x] * 3, axis=-1)
        x = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0
        x = (x - 0.5) / 0.5
        if train and np.random.rand() > 0.5:
            x = x.flip(-1)
        return x

    return _resize_and_normalize


class ImageFolderDataset(Dataset):
    """
    自定义图片数据集：从文件夹读图。
    使用 PIL + 手写 resize/归一化 时，transform 为可调用对象 f(pil) -> tensor。
    """

    def __init__(self, root: str, transform=None, extensions=("jpg", "jpeg", "png", "bmp", "webp")):
        self.root = Path(root)
        self.transform = transform
        self.extensions = set(e.lower() for e in extensions)
        self.samples = []
        if self.root.is_dir():
            for p in self.root.rglob("*"):
                if p.suffix.lower().lstrip(".") in self.extensions:
                    self.samples.append(str(p))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img = Image.open(self.samples[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


def get_dataloader(
    dataset_name="cifar-100-python",
    batch_size=128,
    train=True,
    num_workers=4,
    img_size=32,
):
    """
    根据数据集名返回 DataLoader。
    - 若当前环境 torchvision 可用：cifar10/cifar100 用 torchvision 加载。
    - 若 torchvision 不可用（如版本冲突）：自动用「无 torchvision」路径（CIFAR 用 pickle，自定义用 PIL）。
    """
    root = os.path.join(DATA_ROOT, dataset_name)

    if _check_torchvision():
        from torchvision import datasets, transforms
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        common = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]
        if train:
            common.insert(0, transforms.RandomHorizontalFlip())
        transform = transforms.Compose(common)
        if dataset_name.lower() == "cifar100":
            dataset = datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
        elif dataset_name.lower() == "cifar10":
            dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        else:
            if not os.path.isdir(root):
                raise FileNotFoundError(f"数据集目录不存在: {root}")
            dataset = ImageFolderDataset(root, transform=transform)
    else:
        # 不依赖 torchvision
        if dataset_name.lower() == "cifar100":
            dataset = CIFARNoTv(root, "cifar-100", train, img_size)
        elif dataset_name.lower() == "cifar10":
            dataset = CIFARNoTv(root, "cifar-10", train, img_size)
        else:
            if not os.path.isdir(root):
                raise FileNotFoundError(
                    f"数据集目录不存在: {root}。"
                    "请创建 data/ 下对应目录并放入图片，或使用 cifar10 / cifar100。"
                )
            transform = _transform_pil_to_tensor(img_size, train)
            dataset = ImageFolderDataset(root, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )
