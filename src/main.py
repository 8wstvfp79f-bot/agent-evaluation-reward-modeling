"""
Q-Learning 训练入口

在 GridWorld 环境中训练 Q-Learning 智能体，记录每轮奖励并绘制曲线。
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无 GUI 后端，仅保存图片
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from env import GridWorld
from q_learning import QLearningAgent
from reward_utils import (
    calculate_multi_rewards,
    calculate_weighted_reward,
    is_adjacent_to_obstacle,
)

# ---------------------------------------------------------------------------
# 训练配置
# ---------------------------------------------------------------------------
NUM_EPISODES = 300
MAX_STEPS = 100

# 项目根目录：src/ 的上一级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
REWARD_CURVE_PATH = RESULTS_DIR / "reward_curve.png"
MULTI_REWARD_STATS_PATH = RESULTS_DIR / "multi_reward_stats.csv"
COMPARE_REWARDS_PATH = RESULTS_DIR / "compare_rewards.png"
WEIGHT_EXPERIMENTS_CSV_PATH = RESULTS_DIR / "weight_experiments.csv"
WEIGHT_EXPERIMENT_REWARDS_PATH = RESULTS_DIR / "weight_experiment_rewards.png"
TRAJECTORY_TXT_PATH = RESULTS_DIR / "trajectory.txt"
TRAJECTORY_PNG_PATH = RESULTS_DIR / "trajectory.png"

ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}
WEIGHT_EXPERIMENTS = [
    {
        "name": "benefit-focused",
        "slug": "benefit_focused",
        "benefit_weight": 0.8,
        "cost_weight": 0.1,
        "risk_weight": 0.1,
    },
    {
        "name": "balanced",
        "slug": "balanced",
        "benefit_weight": 0.5,
        "cost_weight": 0.3,
        "risk_weight": 0.2,
    },
    {
        "name": "risk-averse",
        "slug": "risk_averse",
        "benefit_weight": 0.3,
        "cost_weight": 0.2,
        "risk_weight": 0.5,
    },
]


def train(env, agent, num_episodes, max_steps):
    """
    运行 Q-Learning 训练循环。

    Args:
        env:          GridWorld 环境实例
        agent:        QLearningAgent 实例
        num_episodes: 训练轮数
        max_steps:    每轮最大步数

    Returns:
        list: 每轮的累计奖励
    """
    episode_rewards = []
    multi_reward_stats = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        total_benefit = 0.0
        total_cost = 0.0
        total_risk = 0.0

        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            benefit_reward, cost_reward, risk_reward = calculate_multi_rewards(
                env, state, next_state
            )
            total_reward += reward
            total_benefit += benefit_reward
            total_cost += cost_reward
            total_risk += risk_reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)
        multi_reward_stats.append(
            {
                "episode": episode,
                "total_reward": total_reward,
                "total_benefit": total_benefit,
                "total_cost": total_cost,
                "total_risk": total_risk,
            }
        )

        if episode % 50 == 0 or episode == 1:
            print(f"Episode {episode:3d}/{num_episodes} | reward = {total_reward:.1f}")

    return episode_rewards, multi_reward_stats


def train_with_weights(
    env,
    agent,
    num_episodes,
    max_steps,
    experiment_name,
    benefit_weight,
    cost_weight,
    risk_weight,
):
    """使用加权后的多目标 reward 训练单个 Q-Learning agent。"""
    episode_weighted_rewards = []
    experiment_stats = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_weighted_reward = 0.0
        total_benefit = 0.0
        total_cost = 0.0
        total_risk = 0.0

        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, _, done = env.step(action)

            benefit_reward, cost_reward, risk_reward = calculate_multi_rewards(
                env, state, next_state
            )
            weighted_reward = calculate_weighted_reward(
                benefit_reward,
                cost_reward,
                risk_reward,
                benefit_weight,
                cost_weight,
                risk_weight,
            )

            agent.update(state, action, weighted_reward, next_state, done)

            total_weighted_reward += weighted_reward
            total_benefit += benefit_reward
            total_cost += cost_reward
            total_risk += risk_reward
            state = next_state

            if done:
                break

        episode_weighted_rewards.append(total_weighted_reward)
        experiment_stats.append(
            {
                "experiment": experiment_name,
                "episode": episode,
                "weighted_reward": total_weighted_reward,
                "total_benefit": total_benefit,
                "total_cost": total_cost,
                "total_risk": total_risk,
            }
        )

        if episode % 50 == 0 or episode == 1:
            print(
                f"{experiment_name:15s} | Episode {episode:3d}/{num_episodes} "
                f"| weighted_reward = {total_weighted_reward:.2f}"
            )

    return episode_weighted_rewards, experiment_stats


def save_multi_reward_stats(multi_reward_stats, save_path):
    """保存每轮 episode 的 benefit / cost / risk 统计结果。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "total_reward",
        "total_benefit",
        "total_cost",
        "total_risk",
    ]

    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(multi_reward_stats)

    print(f"多目标 reward 统计已保存至: {save_path}")


def save_weight_experiment_stats(experiment_stats, save_path):
    """保存三组加权 reward 实验的逐 episode 统计结果。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "episode",
        "weighted_reward",
        "total_benefit",
        "total_cost",
        "total_risk",
    ]

    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(experiment_stats)

    print(f"加权 reward 实验统计已保存至: {save_path}")


def plot_reward_curve(episode_rewards, save_path):
    """
    绘制并保存每轮奖励曲线。

    Args:
        episode_rewards: 每轮累计奖励列表
        save_path:       图片保存路径
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    episodes = range(1, len(episode_rewards) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, episode_rewards, linewidth=1.2, alpha=0.85, label="Episode reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning Training Reward Curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\nReward curve saved to: {save_path}")


def plot_compare_rewards(multi_reward_stats, save_path):
    """在同一张图中绘制 benefit / cost / risk reward 曲线。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    episodes = [row["episode"] for row in multi_reward_stats]
    benefit_rewards = [row["total_benefit"] for row in multi_reward_stats]
    cost_rewards = [row["total_cost"] for row in multi_reward_stats]
    risk_rewards = [row["total_risk"] for row in multi_reward_stats]

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, benefit_rewards, linewidth=1.2, label="Benefit reward")
    plt.plot(episodes, cost_rewards, linewidth=1.2, label="Cost reward")
    plt.plot(episodes, risk_rewards, linewidth=1.2, label="Risk reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Benefit / Cost / Risk Reward Curves")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"多目标 reward 对比图已保存至: {save_path}")


def plot_weight_experiment_rewards(weighted_rewards_by_experiment, save_path):
    """在同一张图中对比三组权重实验的 weighted reward 曲线。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    for experiment_name, weighted_rewards in weighted_rewards_by_experiment.items():
        episodes = range(1, len(weighted_rewards) + 1)
        plt.plot(
            episodes,
            weighted_rewards,
            linewidth=1.2,
            alpha=0.85,
            label=experiment_name,
        )

    plt.xlabel("Episode")
    plt.ylabel("Weighted Reward")
    plt.title("Weighted Reward Experiments")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"加权 reward 实验曲线已保存至: {save_path}")


def run_weight_experiments(num_episodes, max_steps):
    """依次运行三组权重实验，并保存统计、曲线和 greedy trajectory。"""
    all_experiment_stats = []
    weighted_rewards_by_experiment = {}

    print("\nWeighted Reward Experiments")
    print("-" * 40)

    for config in WEIGHT_EXPERIMENTS:
        env = GridWorld()
        agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.2)

        episode_weighted_rewards, experiment_stats = train_with_weights(
            env,
            agent,
            num_episodes,
            max_steps,
            config["name"],
            config["benefit_weight"],
            config["cost_weight"],
            config["risk_weight"],
        )

        weighted_rewards_by_experiment[config["name"]] = episode_weighted_rewards
        all_experiment_stats.extend(experiment_stats)

        trajectory = collect_greedy_trajectory(env, agent, max_steps)
        save_trajectory(
            trajectory,
            RESULTS_DIR / f"trajectory_{config['slug']}.txt",
        )

    save_weight_experiment_stats(
        all_experiment_stats,
        WEIGHT_EXPERIMENTS_CSV_PATH,
    )
    plot_weight_experiment_rewards(
        weighted_rewards_by_experiment,
        WEIGHT_EXPERIMENT_REWARDS_PATH,
    )


def collect_greedy_trajectory(env, agent, max_steps):
    """Run the learned greedy policy from start and record visited states."""
    state = env.reset()
    trajectory = [state]

    for _ in range(max_steps):
        action = agent.get_best_action(state)
        next_state, _, done = env.step(action)
        trajectory.append(next_state)
        state = next_state

        if done:
            break

    return trajectory


def format_trajectory(trajectory):
    """Format as [(r,c), (r,c), ...] for a compact results file."""
    return "[" + ", ".join(f"({row},{col})" for row, col in trajectory) + "]"


def save_trajectory(trajectory, save_path):
    """Save the greedy trajectory to a text file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(format_trajectory(trajectory), encoding="utf-8")
    print(f"Trajectory saved to: {save_path}")


def plot_trajectory(env, trajectory, save_path):
    """Draw the GridWorld and overlay the greedy trajectory."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.set_xlim(0, env.cols)
    ax.set_ylim(env.rows, 0)
    ax.set_aspect("equal")
    ax.set_xticks(range(env.cols + 1))
    ax.set_yticks(range(env.rows + 1))
    ax.grid(True, color="#94a3b8", linewidth=1.0)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for row in range(env.rows):
        for col in range(env.cols):
            cell = (row, col)
            facecolor = "#f8fafc"
            label = ""
            label_color = "#0f172a"

            if cell == env.start:
                facecolor = "#bbf7d0"
                label = "S"
            elif cell == env.goal:
                facecolor = "#fde68a"
                label = "G"
            elif cell in env.obstacles:
                facecolor = "#475569"
                label = "#"
                label_color = "#ffffff"

            ax.add_patch(
                Rectangle(
                    (col, row),
                    1,
                    1,
                    facecolor=facecolor,
                    edgecolor="#334155",
                    linewidth=1.2,
                )
            )

            if label:
                ax.text(
                    col + 0.5,
                    row + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                    color=label_color,
                    zorder=5,
                )

    if len(trajectory) > 1:
        points = [(col + 0.5, row + 0.5) for row, col in trajectory]
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        ax.plot(
            xs,
            ys,
            color="#2563eb",
            linewidth=2.2,
            marker="o",
            markersize=4,
            zorder=3,
        )

        for start, end in zip(points, points[1:]):
            if start == end:
                continue
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#2563eb",
                    "lw": 1.8,
                    "shrinkA": 8,
                    "shrinkB": 8,
                },
                zorder=4,
            )

    ax.set_title("Greedy Policy Trajectory")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Trajectory plot saved to: {save_path}")


def print_q_table(agent):
    """格式化打印 Q 表。"""
    print("\n" + "=" * 60)
    print("Q-Table")
    print("=" * 60)

    if not agent.q_table:
        print("(empty)")
        return

    for state in sorted(agent.q_table.keys()):
        q_values = agent.get_q_values(state)
        print(f"\nState {state}:")
        for action, q in q_values.items():
            print(f"  {ACTION_NAMES[action]:5s} (a={action}): {q:8.4f}")

    print("\n" + "-" * 60)
    print("Learned policy (greedy):")
    policy = agent.get_policy()
    for state in sorted(policy.keys()):
        action = policy[state]
        print(f"  {state} -> {ACTION_NAMES[action]}")


def main():
    env = GridWorld()
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.2)

    print("Q-Learning Training on GridWorld")
    print(f"Episodes: {NUM_EPISODES} | {agent}")
    print("-" * 40)

    episode_rewards, multi_reward_stats = train(env, agent, NUM_EPISODES, MAX_STEPS)
    plot_reward_curve(episode_rewards, REWARD_CURVE_PATH)
    save_multi_reward_stats(multi_reward_stats, MULTI_REWARD_STATS_PATH)
    plot_compare_rewards(multi_reward_stats, COMPARE_REWARDS_PATH)
    trajectory = collect_greedy_trajectory(env, agent, MAX_STEPS)
    save_trajectory(trajectory, TRAJECTORY_TXT_PATH)
    plot_trajectory(env, trajectory, TRAJECTORY_PNG_PATH)
    print_q_table(agent)
    run_weight_experiments(NUM_EPISODES, MAX_STEPS)

    print(f"\nBaseline training finished. Mean reward (last 50): "
          f"{sum(episode_rewards[-50:]) / min(50, len(episode_rewards)):.2f}")


if __name__ == "__main__":
    main()
