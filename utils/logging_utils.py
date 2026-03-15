"""
训练日志与输出路径：统一写到 output/<method>/ 下。
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")


def get_save_dir(method_name, subdir=None):
    """output/<method_name>/ 或 output/<method_name>/<subdir>/"""
    path = os.path.join(OUTPUT_ROOT, method_name)
    if subdir:
        path = os.path.join(path, subdir)
    os.makedirs(path, exist_ok=True)
    return path


class LossLogger:
    """记录 epoch / step 的 loss，写 csv 并可选画曲线。"""

    def __init__(self, save_dir, log_name="loss_log.csv"):
        self.save_dir = save_dir
        self.csv_path = os.path.join(save_dir, log_name)
        self.rows = []
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "avg_loss"])

    def log(self, epoch, avg_loss):
        self.rows.append((epoch, avg_loss))
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_loss:.6f}"])

    def plot(self, title="Training Loss", ylabel="Loss", out_name="loss_curve.png"):
        if not self.rows:
            return
        epochs = [r[0] for r in self.rows]
        losses = [r[1] for r in self.rows]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, losses, linewidth=1.5, color="#2563EB")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(self.save_dir, out_name), dpi=150)
        plt.close(fig)
