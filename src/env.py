"""
GridWorld 网格世界环境

一个简化的离散状态、离散动作的强化学习环境。
智能体在 6×7 的网格中移动，目标是到达终点，同时避开障碍。
"""


class GridWorld:
    """
    6 行 7 列的网格世界环境。

    坐标约定：
        - state 为 (row, col) 元组，row 范围 [0, 5]，col 范围 [0, 6]
        - 左上角为 (0, 0)，向右 col 增大，向下 row 增大

    地图布局（S=起点, G=终点, #=障碍, .=空地）：
        S . . . # . .
        . # . . # . .
        . # . # . . .
        . . . # . # .
        # . # . . . .
        . . . . # . G
    """

    # -------------------------------------------------------------------------
    # 动作常量：支持字符串或整数两种形式传入 step()
    # -------------------------------------------------------------------------
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

    # 整数动作与字符串动作的映射表
    ACTION_NAMES = {0: UP, 1: DOWN, 2: LEFT, 3: RIGHT}

    # 每个动作对应的 (row 变化, col 变化)
    ACTION_DELTAS = {
        UP: (-1, 0),
        DOWN: (1, 0),
        LEFT: (0, -1),
        RIGHT: (0, 1),
    }

    # -------------------------------------------------------------------------
    # 奖励常量
    # -------------------------------------------------------------------------
    REWARD_GOAL = 10       # 到达终点
    REWARD_STEP = -1       # 普通合法移动
    REWARD_COLLISION = -5  # 撞墙或撞障碍（位置不变）

    def __init__(self):
        """初始化环境参数与地图配置。"""
        self.rows = 6
        self.cols = 7

        # 固定地图元素位置
        self.start = (0, 0)
        self.goal = (5, 6)
        self.obstacles = {
            (0, 4),
            (1, 1),
            (1, 4),
            (2, 1),
            (2, 3),
            (3, 3),
            (3, 5),
            (4, 0),
            (4, 2),
            (5, 4),
        }

        # 当前智能体所在位置，由 reset() 设置
        self.state = None

    def reset(self):
        """
        重置环境，将智能体放回起点。

        Returns:
            tuple: 初始状态 (row, col)，即 (0, 0)
        """
        self.state = self.start
        return self.state

    def step(self, action):
        """
        执行一步动作，更新状态并返回结果。

        Args:
            action: 动作，可为字符串 ("up"/"down"/"left"/"right")
                    或整数 (0/1/2/3)

        Returns:
            next_state (tuple): 执行动作后的状态 (row, col)
            reward (float):     本步获得的奖励
            done (bool):        是否到达终点（episode 结束）

        Raises:
            ValueError: 传入非法动作时抛出
        """
        # 若传入整数动作，先转换为字符串
        action = self._normalize_action(action)

        # 根据动作计算目标位置
        dr, dc = self.ACTION_DELTAS[action]
        row, col = self.state
        target_row = row + dr
        target_col = col + dc
        target = (target_row, target_col)

        # ----- 碰撞检测：越界（撞墙）或进入障碍 -----
        if not self._is_valid_cell(target) or target in self.obstacles:
            # 位置不变，给予碰撞惩罚
            return self.state, self.REWARD_COLLISION, False

        # ----- 合法移动：更新状态 -----
        self.state = target
        next_state = self.state

        # ----- 到达终点 -----
        if next_state == self.goal:
            return next_state, self.REWARD_GOAL, True

        # ----- 普通移动 -----
        return next_state, self.REWARD_STEP, False

    # -------------------------------------------------------------------------
    # 辅助方法
    # -------------------------------------------------------------------------

    def _normalize_action(self, action):
        """
        将动作统一转换为字符串形式。

        Args:
            action: 字符串或整数动作

        Returns:
            str: "up" / "down" / "left" / "right"

        Raises:
            ValueError: 动作不在合法范围内
        """
        if isinstance(action, int):
            if action not in self.ACTION_NAMES:
                raise ValueError(
                    f"非法整数动作 {action}，合法值为 0~3"
                )
            return self.ACTION_NAMES[action]

        if action not in self.ACTION_DELTAS:
            raise ValueError(
                f"非法动作 '{action}'，合法值为 "
                f"{list(self.ACTION_DELTAS.keys())} 或 0~3"
            )
        return action

    def _is_valid_cell(self, cell):
        """
        判断坐标是否在网格范围内（未越界）。

        Args:
            cell (tuple): (row, col)

        Returns:
            bool: 在范围内为 True，越界为 False
        """
        row, col = cell
        return 0 <= row < self.rows and 0 <= col < self.cols

    def render(self):
        """
        在终端打印当前地图（调试用，可选）。

        示例输出：
            S . . . # . .
            . # . . # . .
            . # . # . . .
            . . . # . # .
            # . # . . . .
            . . . . # . G
        其中 A 表示智能体当前位置。
        """
        grid = [["." for _ in range(self.cols)] for _ in range(self.rows)]

        # 标记固定元素
        gr, gc = self.goal
        grid[gr][gc] = "G"

        for obstacle_row, obstacle_col in self.obstacles:
            grid[obstacle_row][obstacle_col] = "#"

        # 标记智能体（若在起点/终点则优先显示 S/G）
        sr, sc = self.state
        if self.state == self.start:
            grid[sr][sc] = "S"
        elif self.state == self.goal:
            grid[sr][sc] = "G"
        else:
            grid[sr][sc] = "A"

        print("-" * (self.cols * 2 + 1))
        for row in grid:
            print("|" + " ".join(row) + "|")
        print("-" * (self.cols * 2 + 1))


# =============================================================================
# 简单自测：直接运行本文件可验证环境逻辑
# =============================================================================
if __name__ == "__main__":
    env = GridWorld()
    state = env.reset()
    print(f"初始状态: {state}")
    env.render()

    # 演示：向右走两步，再向下
    demo_actions = [
        "down", "down", "down", "right", "down", "down", "right",
        "right", "up", "right", "right", "right", "down",
    ]
    total_reward = 0

    for i, action in enumerate(demo_actions, start=1):
        next_state, reward, done = env.step(action)
        total_reward += reward
        print(f"\nStep {i}: action={action}")
        print(f"  next_state={next_state}, reward={reward}, done={done}")
        env.render()
        if done:
            print(f"\n到达终点！累计奖励: {total_reward}")
            break
