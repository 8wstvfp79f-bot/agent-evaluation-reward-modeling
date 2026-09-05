"""
Bradley-Terry 偏好学习演示。

本脚本刻意保持小而直观，方便学习：
- 四个选项：A、B、C、D
- 模拟偏好数据：A > B > C > D
- 为每个选项学习一个隐藏 reward score
- 使用 NumPy 手写梯度下降
- 使用 matplotlib 保存训练曲线

运行方式：
    python preference_learning.py
"""

from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
BT_CURVE_PATH = RESULTS_DIR / "bt_training_curve.png"

OPTIONS = ["A", "B", "C", "D"]
OPTION_TO_INDEX = {name: index for index, name in enumerate(OPTIONS)}

# 模拟成对偏好数据：胜者 > 败者。
PREFERENCE_PAIRS = [
    ("A", "B"),
    ("A", "C"),
    ("B", "C"),
    ("A", "D"),
    ("B", "D"),
    ("C", "D"),
]

LEARNING_RATE = 0.5
NUM_EPOCHS = 1000
PRINT_EVERY = 100


def sigmoid(x):
    """数值稳定版 sigmoid。"""
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def bradley_terry_probability(scores, winner_index, loser_index):
    """
    计算 P(胜者 > 败者)。

    这与下面的 Bradley-Terry 形式等价：
        exp(r_winner) / (exp(r_winner) + exp(r_loser))

    这里使用 sigmoid 形式，数值上更稳定：
        sigmoid(r_winner - r_loser)
    """
    return sigmoid(scores[winner_index] - scores[loser_index])


def compute_loss_and_gradient(scores, preference_pairs):
    """
    计算负对数似然 loss 及其梯度。

    对每一条偏好 胜者 > 败者：
        loss = -log P(胜者 > 败者)
    """
    loss = 0.0
    gradient = np.zeros_like(scores)

    for winner, loser in preference_pairs:
        winner_index = OPTION_TO_INDEX[winner]
        loser_index = OPTION_TO_INDEX[loser]

        probability = bradley_terry_probability(
            scores,
            winner_index,
            loser_index,
        )
        loss += -np.log(probability + 1e-12)

        # 对分数差 (r_w - r_l) 求导，梯度为 probability - 1。
        delta_gradient = probability - 1.0
        gradient[winner_index] += delta_gradient
        gradient[loser_index] -= delta_gradient

    loss /= len(preference_pairs)
    gradient /= len(preference_pairs)
    return loss, gradient


def format_scores(scores):
    """格式化当前 reward scores，方便在终端打印。"""
    return " | ".join(
        f"r{name}={scores[index]: .4f}"
        for index, name in enumerate(OPTIONS)
    )


def train():
    """
    使用梯度下降为每个选项学习一个隐藏分数。

    Bradley-Terry 分数只在相对大小上有意义，整体平移不影响偏好概率。
    因此每次更新后把分数中心化，让打印出来的数值更容易阅读。
    """
    scores = np.zeros(len(OPTIONS), dtype=float)
    losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        loss, gradient = compute_loss_and_gradient(scores, PREFERENCE_PAIRS)
        scores -= LEARNING_RATE * gradient
        scores -= np.mean(scores)
        losses.append(loss)

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print(
                f"epoch={epoch:4d} | loss={loss:.6f} | "
                f"{format_scores(scores)}"
            )

    return scores, losses


def plot_training_curve(losses, save_path):
    """保存 Bradley-Terry 训练 loss 曲线。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, losses, linewidth=1.5, label="BT loss")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("Bradley-Terry Preference Learning")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\nTraining curve saved to: {save_path}")


def print_final_scores(scores):
    """打印最终学到的 reward scores。"""
    print("\nFinal reward scores")
    print("-" * 24)
    for index, name in enumerate(OPTIONS):
        print(f"{name} score: {scores[index]:.6f}")


def main():
    print("Bradley-Terry Preference Learning Demo")
    print("Preferences: A > B, A > C, B > C, A > D, B > D, C > D")
    print("-" * 72)

    scores, losses = train()
    plot_training_curve(losses, BT_CURVE_PATH)
    print_final_scores(scores)


if __name__ == "__main__":
    main()
