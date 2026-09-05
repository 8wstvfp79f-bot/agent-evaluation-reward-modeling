"""
多目标 reward 工具函数。

这些函数只依赖 GridWorld 环境本身，不负责训练、绘图或文件输出。
main.py 和 active_querying.py 都可以复用这里的逻辑，避免重复实现。
"""


def is_adjacent_to_obstacle(env, state):
    """判断当前位置是否与障碍物上下左右相邻。"""
    row, col = state
    for obstacle_row, obstacle_col in env.obstacles:
        if abs(row - obstacle_row) + abs(col - obstacle_col) == 1:
            return True
    return False


def calculate_multi_rewards(env, state, next_state):
    """计算多目标 reward 分量，不改变原有环境 reward。"""
    moved = next_state != state
    benefit_reward = env.REWARD_GOAL if next_state == env.goal else 0
    cost_reward = env.REWARD_STEP if moved else 0
    risk_reward = -3 if is_adjacent_to_obstacle(env, next_state) else 0
    return benefit_reward, cost_reward, risk_reward


def calculate_weighted_reward(
    benefit_reward,
    cost_reward,
    risk_reward,
    benefit_weight,
    cost_weight,
    risk_weight,
):
    """按给定权重把多目标 reward 分量合成为一个 weighted score。"""
    return (
        benefit_weight * benefit_reward
        + cost_weight * cost_reward
        + risk_weight * risk_reward
    )
