"""
Q-Learning 智能体

基于表格（Tabular）的 Q-Learning 算法实现。
Q 表使用 Python 字典存储，适用于离散状态、离散动作的小规模环境。
"""

import random


class QLearningAgent:
    """
    Q-Learning 智能体。

    使用 epsilon-greedy 策略在「探索」与「利用」之间平衡：
        - 以 epsilon 概率随机选动作（探索未知区域）
        - 以 1-epsilon 概率选 Q 值最大的动作（利用已有知识）

    Q 表结构：
        q_table[state][action] = Q值
        例如 q_table[(0, 0)][2] 表示在 (0,0) 状态下执行动作 2 的 Q 值

    动作编号（与 GridWorld 一致）：
        0 = up, 1 = down, 2 = left, 3 = right
    """

    # 合法动作集合
    ACTIONS = (0, 1, 2, 3)

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        """
        初始化 Q-Learning 智能体。

        Args:
            alpha (float):   学习率，控制每次更新对 Q 值的修正幅度
                             越大学得越快，但可能不稳定；默认 0.1
            gamma (float):   折扣因子，衡量未来奖励的重要性
                             接近 1 表示更重视长期回报；默认 0.9
            epsilon (float): 探索率，随机选动作的概率；默认 0.2
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q 表：{ state: { action: q_value } }
        # 状态未访问过时不预分配，按需 lazy 初始化
        self.q_table = {}

    # -------------------------------------------------------------------------
    # 核心接口
    # -------------------------------------------------------------------------

    def get_q_values(self, state):
        """
        返回给定状态下所有动作的 Q 值。

        若该状态尚未出现在 Q 表中，则所有动作的 Q 值默认为 0.0。

        Args:
            state: 环境状态，例如 GridWorld 中的 (row, col) 元组

        Returns:
            dict: { action(int): q_value(float) }
                  包含全部 4 个动作，未访问过的 Q 值为 0.0
        """
        state_q = self.q_table.get(state, {})
        return {action: state_q.get(action, 0.0) for action in self.ACTIONS}

    def choose_action(self, state):
        """
        使用 epsilon-greedy 策略选择动作。

        策略逻辑：
            1. 生成 [0, 1) 均匀随机数 u
            2. 若 u < epsilon → 从 ACTIONS 中均匀随机选一个（探索）
            3. 否则 → 选 Q 值最大的动作（利用）
               若有多个动作 Q 值相同且均为最大，则从中随机选一个

        Args:
            state: 当前状态

        Returns:
            int: 选中的动作编号 (0~3)
        """
        # ----- 探索：以 epsilon 概率随机动作 -----
        if random.random() < self.epsilon:
            return random.choice(self.ACTIONS)

        # ----- 利用：选 Q 值最大的动作 -----
        q_values = self.get_q_values(state)
        max_q = max(q_values.values())

        # 收集所有达到最大 Q 值的动作，打破平局
        best_actions = [
            action for action, q in q_values.items() if q == max_q
        ]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """
        根据 Q-Learning 更新规则修正 Q(s, a)。

        更新公式（Bellman 方程）：
            Q(s,a) ← Q(s,a) + α * [ r + γ * max_a' Q(s',a') - Q(s,a) ]

        当 done=True（episode 结束）时，没有下一状态的价值，因此：
            target = r
        否则：
            target = r + γ * max_a' Q(s',a')

        Args:
            state:      当前状态 s
            action:     执行的动作 a
            reward:     获得的即时奖励 r
            next_state: 转移后的下一状态 s'
            done:       是否终止（到达终点等）

        Returns:
            float: 本次更新后的 Q(s, a) 值
        """
        # 确保 state 在 Q 表中有条目
        if state not in self.q_table:
            self.q_table[state] = {}

        current_q = self.q_table[state].get(action, 0.0)

        # 计算 TD 目标（Temporal Difference target）
        if done:
            # 终止状态：未来价值为 0
            target = reward
        else:
            next_q_values = self.get_q_values(next_state)
            max_next_q = max(next_q_values.values())
            target = reward + self.gamma * max_next_q

        # TD 误差与 Q 值更新
        td_error = target - current_q
        new_q = current_q + self.alpha * td_error
        self.q_table[state][action] = new_q

        return new_q

    # -------------------------------------------------------------------------
    # 辅助方法
    # -------------------------------------------------------------------------

    def get_best_action(self, state):
        """
        纯贪婪策略：始终选 Q 值最大的动作（不含随机探索）。

        用于评估训练后的策略，而非训练过程中的动作选择。

        Args:
            state: 当前状态

        Returns:
            int: Q 值最大的动作编号
        """
        q_values = self.get_q_values(state)
        max_q = max(q_values.values())
        best_actions = [
            action for action, q in q_values.items() if q == max_q
        ]
        return random.choice(best_actions)

    def get_policy(self):
        """
        导出当前学到的确定性策略（每个状态选 Q 最大的动作）。

        Returns:
            dict: { state: best_action(int) }
        """
        policy = {}
        for state in self.q_table:
            policy[state] = self.get_best_action(state)
        return policy

    def __repr__(self):
        return (
            f"QLearningAgent(alpha={self.alpha}, gamma={self.gamma}, "
            f"epsilon={self.epsilon}, states={len(self.q_table)})"
        )


# =============================================================================
# 简单自测：在 GridWorld 上训练若干 episode
# =============================================================================
if __name__ == "__main__":
    from env import GridWorld

    env = GridWorld()
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.2)

    num_episodes = 500
    max_steps = 100

    print(f"开始训练: {num_episodes} episodes")
    print(agent)
    print()

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            if done:
                break

        # 每 100 个 episode 打印一次进度
        if episode % 100 == 0:
            print(f"Episode {episode:4d} | 本局奖励: {total_reward:6.1f} | "
                  f"已学状态数: {len(agent.q_table)}")

    print("\n训练完成，学到的策略（纯贪婪）：")
    policy = agent.get_policy()
    for state in sorted(policy.keys()):
        action_names = {0: "up", 1: "down", 2: "left", 3: "right"}
        print(f"  状态 {state} → {action_names[policy[state]]}")

    # 用纯贪婪策略跑一局验证
    print("\n验证：用贪婪策略从起点到终点")
    state = env.reset()
    env.render()
    total_reward = 0
    for step in range(max_steps):
        action = agent.get_best_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        print(f"  Step {step + 1}: action={GridWorld.ACTION_NAMES[action]}, "
              f"reward={reward}, state={next_state}")
        state = next_state
        if done:
            env.render()
            print(f"  累计奖励: {total_reward}")
            break
