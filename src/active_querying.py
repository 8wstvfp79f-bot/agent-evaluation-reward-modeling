"""
主动查询 / 最大分歧演示。

本脚本复用当前项目的 GridWorld、QLearningAgent 和多目标 reward 逻辑：
- 先在 GridWorld 中训练一个 Q-Learning agent
- 再从训练后的 agent 生成 20 条候选 trajectory
- 每条 trajectory 记录路径、动作、原始总 reward、多目标 reward 分量和 reward_score
- 使用 RewardModel 输出的 reward_score 计算 Bradley-Terry 概率
- 选择最接近 P(A > B) = 0.5 的 pair，作为最值得人工标注的数据

运行方式：
    python src/active_querying.py
"""

from pathlib import Path
import csv
import random

import numpy as np

from env import GridWorld
from q_learning import QLearningAgent
from reward_model import RewardModel
from reward_utils import calculate_multi_rewards


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
ACTIVE_QUERY_PATH = RESULTS_DIR / "active_querying.txt"
PAIR_CSV_PATH = RESULTS_DIR / "active_querying_pairs.csv"

NUM_TRAJECTORIES = 20
NUM_TRAINING_EPISODES = 300
MAX_STEPS = 100
RANDOM_SEED = 7

BENEFIT_WEIGHT = 0.5
COST_WEIGHT = 0.3
RISK_WEIGHT = 0.2

reward_model = RewardModel(
    benefit_weight=BENEFIT_WEIGHT,
    cost_weight=COST_WEIGHT,
    risk_weight=RISK_WEIGHT,
)

ACTION_NAMES = GridWorld.ACTION_NAMES


def sigmoid(x):
    """数值稳定版 sigmoid。"""
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def bradley_terry_probability(score_a, score_b):
    """
    计算 P(A > B)。

    这里使用 Bradley-Terry 的 sigmoid 形式：
        P(A > B) = sigmoid(score_A - score_B)
    """
    return sigmoid(score_a - score_b)


def train_agent_for_active_querying(
    env,
    agent,
    num_episodes=NUM_TRAINING_EPISODES,
    max_steps=MAX_STEPS,
):
    """
    训练一个用于生成候选 trajectory 的 Q-Learning agent。

    这里保留和 main.py 相同的基础 Q-Learning 交互方式：
    agent 通过 env.step() 得到原始环境 reward，并用 agent.update() 更新 Q-table。
    """
    for _ in range(num_episodes):
        state = env.reset()

        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

            if done:
                break


def choose_candidate_action(agent, state, exploration_rate, rng):
    """
    为候选 trajectory 选择动作。

    exploration_rate 越高，越容易偏离当前 greedy policy，
    因而可以生成更丰富的候选轨迹，而不是 20 条完全相同的最优路径。
    """
    if rng.random() < exploration_rate:
        return int(rng.choice(QLearningAgent.ACTIONS))

    q_values = agent.get_q_values(state)
    max_q = max(q_values.values())
    best_actions = [
        action for action, q_value in q_values.items() if q_value == max_q
    ]
    return min(best_actions)


def summarize_trajectory(
    trajectory_id,
    states,
    actions,
    total_reward,
    total_benefit,
    total_cost,
    total_risk,
):
    """把一条 rollout 轨迹整理成主动查询需要的结构化记录。"""
    trajectory = {
        "trajectory_id": trajectory_id,
        "states": states,
        "actions": actions,
        "total_reward": total_reward,
        "total_benefit": total_benefit,
        "total_cost": total_cost,
        "total_risk": total_risk,
    }
    reward_score = reward_model.score_trajectory(trajectory)
    trajectory["reward_score"] = reward_score
    trajectory["weighted_score"] = reward_score
    return trajectory


def rollout_candidate_trajectory(env, agent, exploration_rate, rng):
    """从起点 rollout 一条候选 trajectory，并累计多目标 reward。"""
    state = env.reset()
    states = [state]
    actions = []
    total_reward = 0.0
    total_benefit = 0.0
    total_cost = 0.0
    total_risk = 0.0

    for _ in range(MAX_STEPS):
        action = choose_candidate_action(
            agent,
            state,
            exploration_rate,
            rng,
        )
        next_state, reward, done = env.step(action)
        benefit_reward, cost_reward, risk_reward = calculate_multi_rewards(
            env,
            state,
            next_state,
        )

        actions.append(ACTION_NAMES[action])
        states.append(next_state)
        total_reward += reward
        total_benefit += benefit_reward
        total_cost += cost_reward
        total_risk += risk_reward
        state = next_state

        if done:
            break

    return states, actions, total_reward, total_benefit, total_cost, total_risk


def trajectory_signature(states, actions):
    """生成轨迹签名，用于避免候选集中出现完全重复的 rollout。"""
    return tuple(states), tuple(actions)


def generate_candidate_trajectories(env, agent, num_trajectories=NUM_TRAJECTORIES):
    """
    生成候选 trajectories。

    每条轨迹都从 GridWorld 起点出发。低 exploration_rate 的轨迹更接近
    learned greedy policy；高 exploration_rate 的轨迹会包含更多探索动作。
    这里会尽量保留不同路径的候选轨迹，避免人工标注比较完全重复的样本。
    """
    rng = np.random.default_rng(RANDOM_SEED)
    planned_rates = np.linspace(0.0, 0.65, num_trajectories)
    trajectories = []
    seen_signatures = set()
    attempt = 0
    max_attempts = num_trajectories * 50

    while len(trajectories) < num_trajectories and attempt < max_attempts:
        if attempt < len(planned_rates):
            exploration_rate = planned_rates[attempt]
        else:
            exploration_rate = float(rng.uniform(0.05, 0.75))

        (
            states,
            actions,
            total_reward,
            total_benefit,
            total_cost,
            total_risk,
        ) = rollout_candidate_trajectory(
            env,
            agent,
            exploration_rate,
            rng,
        )

        signature = trajectory_signature(states, actions)
        attempt += 1

        if signature in seen_signatures:
            continue

        seen_signatures.add(signature)
        trajectory_index = len(trajectories) + 1
        trajectories.append(
            summarize_trajectory(
                trajectory_id=f"T{trajectory_index:02d}",
                states=states,
                actions=actions,
                total_reward=total_reward,
                total_benefit=total_benefit,
                total_cost=total_cost,
                total_risk=total_risk,
            )
        )

    # 小地图里的唯一轨迹数量通常足够。这里保留兜底逻辑，保证函数始终返回
    # num_trajectories 条候选轨迹。
    while len(trajectories) < num_trajectories:
        (
            states,
            actions,
            total_reward,
            total_benefit,
            total_cost,
            total_risk,
        ) = rollout_candidate_trajectory(
            env,
            agent,
            exploration_rate=0.75,
            rng=rng,
        )

        trajectory_index = len(trajectories) + 1
        trajectories.append(
            summarize_trajectory(
                trajectory_id=f"T{trajectory_index:02d}",
                states=states,
                actions=actions,
                total_reward=total_reward,
                total_benefit=total_benefit,
                total_cost=total_cost,
                total_risk=total_risk,
            )
        )

    return trajectories


def compute_pair_uncertainties(trajectories):
    """
    为每一对 trajectory 计算 preference probability 和 uncertainty。

    这里使用 trajectory 的 reward_score 作为 Bradley-Terry score。
    uncertainty = abs(probability - 0.5)
    数值越小，说明模型越接近五五开，也就是越不确定。
    """
    pairs = []

    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            trajectory_a = trajectories[i]
            trajectory_b = trajectories[j]
            score_a = trajectory_a["reward_score"]
            score_b = trajectory_b["reward_score"]

            probability = bradley_terry_probability(score_a, score_b)
            uncertainty = abs(probability - 0.5)

            pairs.append(
                {
                    "trajectory_a": trajectory_a["trajectory_id"],
                    "trajectory_b": trajectory_b["trajectory_id"],
                    "score_a": score_a,
                    "score_b": score_b,
                    "probability": float(probability),
                    "uncertainty": float(uncertainty),
                }
            )

    return pairs


def select_maximum_disagreement_pair(pairs):
    """选择 uncertainty 最小的 pair，作为 Maximum Disagreement 查询。"""
    return min(pairs, key=lambda pair: pair["uncertainty"])


def save_pair_ranking(pairs, save_path):
    """把所有 trajectory pair 按 uncertainty 从小到大保存到 CSV。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "trajectory_a",
        "trajectory_b",
        "score_a",
        "score_b",
        "probability",
        "uncertainty",
    ]

    sorted_pairs = sorted(pairs, key=lambda pair: pair["uncertainty"])

    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_pairs)

    return sorted_pairs


def format_state_path(states):
    """把 state 路径格式化成紧凑的一行。"""
    return "[" + ", ".join(f"({row},{col})" for row, col in states) + "]"


def format_actions(actions):
    """把动作序列格式化成紧凑的一行。"""
    return "[" + ", ".join(actions) + "]"


def format_trajectory_detail(label, trajectory):
    """格式化单条 trajectory 的路径、动作和 reward 统计。"""
    return "\n".join(
        [
            f"{label}: {trajectory['trajectory_id']}",
            f"states: {format_state_path(trajectory['states'])}",
            f"actions: {format_actions(trajectory['actions'])}",
            f"total_reward: {trajectory['total_reward']:.3f}",
            f"total_benefit: {trajectory['total_benefit']:.3f}",
            f"total_cost: {trajectory['total_cost']:.3f}",
            f"total_risk: {trajectory['total_risk']:.3f}",
            f"reward_score: {trajectory['reward_score']:.3f}",
            f"weighted_score: {trajectory['weighted_score']:.3f}",
        ]
    )


def format_best_pair(best_pair, trajectories):
    """格式化最值得人工标注的 trajectory pair。"""
    trajectory_by_id = {
        trajectory["trajectory_id"]: trajectory
        for trajectory in trajectories
    }
    trajectory_a = trajectory_by_id[best_pair["trajectory_a"]]
    trajectory_b = trajectory_by_id[best_pair["trajectory_b"]]

    lines = [
        "Active Querying / Maximum Disagreement Demo",
        "-" * 56,
        "Most useful trajectory pair for human labeling:",
        "",
        format_trajectory_detail("trajectory_a", trajectory_a),
        "",
        format_trajectory_detail("trajectory_b", trajectory_b),
        "",
        f"P(A > B): {best_pair['probability']:.6f}",
        f"uncertainty: {best_pair['uncertainty']:.6f}",
    ]
    return "\n".join(lines)


def save_active_query(best_pair, trajectories, save_path):
    """把 Maximum Disagreement 的查询结果保存到文本文件。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = format_best_pair(best_pair, trajectories)
    save_path.write_text(output_text + "\n", encoding="utf-8")
    return output_text


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    env = GridWorld()
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.2)

    train_agent_for_active_querying(env, agent)
    trajectories = generate_candidate_trajectories(env, agent)
    pairs = compute_pair_uncertainties(trajectories)
    best_pair = select_maximum_disagreement_pair(pairs)

    save_pair_ranking(pairs, PAIR_CSV_PATH)
    output_text = save_active_query(best_pair, trajectories, ACTIVE_QUERY_PATH)

    print(output_text)
    print()
    print(f"Active query saved to: {ACTIVE_QUERY_PATH}")
    print(f"All pair rankings saved to: {PAIR_CSV_PATH}")


if __name__ == "__main__":
    main()
