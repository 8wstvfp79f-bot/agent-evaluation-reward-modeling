"""
Evaluation Pipeline V1 评测流水线。

本脚本复用当前项目已有的 GridWorld、QLearningAgent、RewardModel
和 Active Querying 中的候选轨迹生成逻辑，完成一批 agent trajectories
的自动评测：
- 先在 GridWorld 中训练一个 Q-Learning agent
- 再生成多条候选 trajectory
- 使用 RewardModel 为每条 trajectory 计算 reward_score
- 按 reward_score 从高到低排序
- 保存 CSV，并在终端打印 Top-K 轨迹

运行方式：
    python src/evaluation_pipeline.py
"""

from pathlib import Path
import csv
import random

import numpy as np

from active_querying import (
    RANDOM_SEED,
    format_actions,
    format_state_path,
    generate_candidate_trajectories,
    train_agent_for_active_querying,
)
from env import GridWorld
from q_learning import QLearningAgent
from reward_model import RewardModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TRAJECTORY_SCORES_PATH = RESULTS_DIR / "trajectory_scores.csv"

NUM_TRAJECTORIES = 50


def evaluate_trajectories(trajectories, reward_model):
    """
    使用 RewardModel 为每条 trajectory 打分，并按分数从高到低排序。

    Args:
        trajectories: trajectory 字典列表。
        reward_model: 用于计算 reward_score 的 RewardModel 实例。

    Returns:
        list[dict]: 按 reward_score 降序排列后的 trajectories。
    """
    for trajectory in trajectories:
        trajectory["reward_score"] = reward_model.score_trajectory(trajectory)

    return sorted(
        trajectories,
        key=lambda trajectory: trajectory["reward_score"],
        reverse=True,
    )


def save_trajectory_scores(sorted_trajectories, save_path):
    """将排序后的 trajectory 分数和统计信息保存为 CSV 文件。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "trajectory_id",
        "reward_score",
        "total_reward",
        "total_benefit",
        "total_cost",
        "total_risk",
        "num_steps",
        "states",
        "actions",
    ]

    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for rank, trajectory in enumerate(sorted_trajectories, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "trajectory_id": trajectory["trajectory_id"],
                    "reward_score": trajectory["reward_score"],
                    "total_reward": trajectory["total_reward"],
                    "total_benefit": trajectory["total_benefit"],
                    "total_cost": trajectory["total_cost"],
                    "total_risk": trajectory["total_risk"],
                    "num_steps": len(trajectory["actions"]),
                    "states": format_state_path(trajectory["states"]),
                    "actions": format_actions(trajectory["actions"]),
                }
            )


def print_top_trajectories(sorted_trajectories, top_k=5):
    """在终端打印 reward_score 最高的 Top-K trajectories。"""
    print(f"Top {top_k} Trajectories")
    print("-" * 72)

    for rank, trajectory in enumerate(sorted_trajectories[:top_k], start=1):
        print(f"Rank: {rank}")
        print(f"Trajectory ID: {trajectory['trajectory_id']}")
        print(f"Reward Score: {trajectory['reward_score']:.3f}")
        print(f"Total Benefit: {trajectory['total_benefit']:.3f}")
        print(f"Total Cost: {trajectory['total_cost']:.3f}")
        print(f"Total Risk: {trajectory['total_risk']:.3f}")
        print(f"Num Steps: {len(trajectory['actions'])}")
        print("-" * 72)


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    env = GridWorld()
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.2)

    train_agent_for_active_querying(env, agent)
    trajectories = generate_candidate_trajectories(
        env,
        agent,
        num_trajectories=NUM_TRAJECTORIES,
    )

    reward_model = RewardModel()
    sorted_trajectories = evaluate_trajectories(trajectories, reward_model)

    save_trajectory_scores(sorted_trajectories, TRAJECTORY_SCORES_PATH)
    print_top_trajectories(sorted_trajectories)
    print(f"Trajectory scores saved to: {TRAJECTORY_SCORES_PATH}")


if __name__ == "__main__":
    main()
