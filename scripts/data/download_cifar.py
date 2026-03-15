"""
可选：显式下载 CIFAR-10 / CIFAR-100 到 data/，再训练时直接用。
数据本来也可以由 dataloader 在首次加载时自动下载，此脚本只是方便「先下好再跑」。
"""
import argparse
import os
import sys
import urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.dataloader import DATA_ROOT

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"


def download(url: str, path: str):
    if os.path.isfile(path):
        print(f"已存在，跳过: {path}")
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"下载: {url} -> {path}")
    urllib.request.urlretrieve(url, path)
    return path


def main():
    parser = argparse.ArgumentParser(description="下载 CIFAR 到 data/")
    parser.add_argument("--which", choices=["10", "100", "both"], default="both", help="下载 CIFAR-10 / 100 / 都下")
    args = parser.parse_args()
    if args.which in ("10", "both"):
        path = download(CIFAR10_URL, os.path.join(DATA_ROOT, "cifar10", "cifar-10-python.tar.gz"))
        if not os.path.isdir(os.path.join(DATA_ROOT, "cifar10", "cifar-10-batches-py")):
            import tarfile
            print("解压 CIFAR-10...")
            with tarfile.open(path, "r:gz") as tf:
                tf.extractall(os.path.join(DATA_ROOT, "cifar10"))
    if args.which in ("100", "both"):
        path = download(CIFAR100_URL, os.path.join(DATA_ROOT, "cifar100", "cifar-100-python.tar.gz"))
        if not os.path.isdir(os.path.join(DATA_ROOT, "cifar100", "cifar-100-python")):
            import tarfile
            print("解压 CIFAR-100...")
            with tarfile.open(path, "r:gz") as tf:
                tf.extractall(os.path.join(DATA_ROOT, "cifar100"))
    print("完成。训练时 config 里 dataset_name 用 cifar10 或 cifar100 即可。")


if __name__ == "__main__":
    main()
