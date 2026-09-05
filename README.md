# Agent Evaluation & Reward Modeling

## 项目概览

Agent Evaluation & Reward Modeling 是一个教学型项目，用来把多目标强化学习
（Multi-Objective Reinforcement Learning, MORL）和偏好对齐
（Preference Alignment）串起来。

项目使用一个 6x7 的 GridWorld，并包含多个障碍物。Q-Learning agent
需要学习如何从起点到达终点；同时，项目会把每条轨迹拆成 benefit、
cost、risk 三类 reward 分量。之后这些分量会被 RewardModel 转换成
`reward_score`，再通过 Bradley-Terry 模型计算轨迹偏好概率，最后由
Active Querying 选出最值得人工标注的 trajectory pair。

整体目标可以理解为：

```text
多目标强化学习
+ 偏好对齐
= 轨迹生成、Reward Model 打分、偏好概率计算、主动查询选择
```

## 核心组件

### Q-Learning 训练

`src/q_learning.py` 实现了一个表格型 Q-Learning agent，使用
epsilon-greedy 策略在探索和利用之间切换。`src/main.py` 负责训练 agent，
并保存 reward 曲线、greedy trajectory 和策略结果。

### 多目标 Rewards

`src/reward_utils.py` 负责计算多目标 reward 分量：

- `benefit`：到达终点带来的收益
- `cost`：移动步数带来的成本
- `risk`：靠近障碍物带来的风险惩罚

这样每条 trajectory 不只是一个单一 reward，而是包含更丰富的行为信息。

### Pareto 权衡

项目通过不同权重组合展示 Pareto 风格的目标权衡。不同 benefit、cost、
risk 权重会偏好不同策略，例如更短路径、更安全路径，或者更重视到达终点。

### Bradley-Terry 偏好模型

`src/preference_learning.py` 演示了 Bradley-Terry 偏好学习。Active Querying
中也使用同样的概率形式：

```text
P(A > B) = sigmoid(score_A - score_B)
```

### Active Querying 主动查询

`src/active_querying.py` 会从训练后的 agent 中生成候选 trajectories，
使用 `RewardModel` 为每条 trajectory 计算 `reward_score`，然后遍历所有
trajectory pair，选择最接近 `P(A > B) = 0.5` 的 pair。

这个 pair 就是 Maximum Disagreement pair：模型最不确定哪条轨迹更好，
因此最适合交给人工标注。

### Reward Model 奖励模型

`src/reward_model.py` 定义了一个轻量级 `RewardModel` 类。它会把 trajectory
级别的 benefit、cost、risk 累计值转换成单一的 `reward_score`。当前版本
故意保持简单，内部复用 `reward_utils.calculate_weighted_reward`。

## 项目结构

```text
agent-evaluation-reward-modeling/
|-- README.md
|-- requirements.txt
|-- docs/
|   `-- architecture.md
|-- frontend/
|   |-- index.html
|   |-- styles.css
|   `-- app.js
|-- results/
|   |-- active_querying.txt
|   |-- active_querying_pairs.csv
|   |-- reward_curve.png
|   |-- trajectory.txt
|   `-- ...
`-- src/
    |-- active_querying.py
    |-- env.py
    |-- evaluation_pipeline.py
    |-- main.py
    |-- preference_learning.py
    |-- q_learning.py
    |-- reward_model.py
    `-- reward_utils.py
```

## 如何运行

训练 Q-Learning 和 MORL 实验：

```bash
python src/main.py
```

运行 Active Querying：

```bash
python src/active_querying.py
```

运行 Reward Model demo：

```bash
python src/reward_model.py
```

如果 Windows 上 `python` 命令不可用，可以使用：

```bash
py -3 src/main.py
py -3 src/active_querying.py
py -3 src/reward_model.py
```

## 前端演示

项目提供一个不依赖额外框架的结果仪表盘，用来查看Q-Learning训练曲线、三组多目标权重实验、trajectory排名和Bradley-Terry主动查询结果。在项目根目录执行：

```powershell
python -m http.server 8081
```

然后打开`http://127.0.0.1:8081/frontend/`。前端读取`results/`里的现有CSV与PNG，不会重新训练，也不会修改实验结果。

页面不是模型训练器，而是实验结果阅读器：左侧选择 benefit/cost/risk 权重方案，右侧查看奖励曲线、轨迹得分排行，以及 Bradley-Terry 概率最接近 0.5 的主动查询样本。

Windows 也可以在项目根目录运行 `.\start_frontend.ps1`，无需安装 Node.js。

## 示例输出

### 轨迹 Reward Score

RewardModel 会把 trajectory 的多目标统计合成为一个标量分数：

```text
total_benefit: 10
total_cost: -15
total_risk: -33
reward_score: -6.100
```

默认权重下的计算公式是：

```text
reward_score = 0.5 * total_benefit
             + 0.3 * total_cost
             + 0.2 * total_risk
```

### Active Querying 结果

`results/active_querying.txt` 会展示最值得人工标注的 trajectory pair：

```text
Most useful trajectory pair for human labeling:

trajectory_a: T02
reward_score: -6.100

trajectory_b: T08
reward_score: -6.100

P(A > B): 0.500000
uncertainty: 0.000000
```

概率越接近 0.5，说明模型越不确定。Active Querying 会优先选择这样的
pair，因为人工标注它们最有信息量。
