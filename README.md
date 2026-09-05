# 智能体评估与奖励建模实验

## 项目概览

本项目是一个教学型智能体评估与奖励建模实验，用来串联多目标强化学习
（Multi-Objective Reinforcement Learning，MORL）和偏好对齐。

项目使用一个包含多个障碍物的 6×7 GridWorld。Q-Learning 智能体
需要学习如何从起点到达终点；同时，项目会把每条轨迹拆成 benefit、
cost、risk 三类奖励分量。之后这些分量会被 `RewardModel` 转换成
`reward_score`，再通过 Bradley-Terry 模型计算轨迹偏好概率，最后由
主动查询选出最值得人工标注的轨迹对。

整体目标可以理解为：

```text
多目标强化学习
+ 偏好对齐
= 轨迹生成、奖励模型打分、偏好概率计算、主动查询选择
```

## 核心组件

### Q-Learning 训练

`src/q_learning.py` 实现了一个表格型 Q-Learning 智能体，使用
epsilon-greedy 策略在探索和利用之间切换。`src/main.py` 负责训练智能体，
并保存奖励曲线、贪心轨迹和策略结果。

### 多目标奖励

`src/reward_utils.py` 负责计算多目标奖励分量：

- `benefit`：到达终点带来的收益
- `cost`：移动步数带来的成本
- `risk`：靠近障碍物带来的风险惩罚

这样每条轨迹不只有一个单一奖励，还包含更丰富的行为信息。

### 多目标权衡

项目通过不同权重组合展示 Pareto 风格的目标权衡。不同 benefit、cost、
risk 权重会偏好不同策略，例如更短路径、更安全路径，或者更重视到达终点。

### Bradley-Terry 偏好模型

`src/preference_learning.py` 演示了 Bradley-Terry 偏好学习。主动查询
中也使用同样的概率形式：

```text
P(A > B) = sigmoid(score_A - score_B)
```

### 主动查询

`src/active_querying.py` 会从训练后的智能体中生成候选轨迹，
使用 `RewardModel` 为每条轨迹计算 `reward_score`，然后遍历所有轨迹对，
选择最接近 `P(A > B) = 0.5` 的轨迹对。

这个轨迹对就是最大分歧样本：模型最不确定哪条轨迹更好，
因此最适合交给人工标注。

### 奖励模型

`src/reward_model.py` 定义了一个轻量级 `RewardModel` 类。它会把轨迹
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

## 推荐运行顺序

这些脚本多数可以独立运行，但按下面顺序执行最容易理解完整数据流；前端依赖 `results/` 中已经生成的 CSV、文本和图片。

### 1. 安装依赖

```powershell
python -m pip install -r requirements.txt
```

### 2. 运行 Q-Learning 与多目标权重实验

```bash
python src/main.py
```

### 3. 运行奖励模型示例

```bash
python src/reward_model.py
```

### 4. 运行主动查询

```bash
python src/active_querying.py
```

### 5. 生成统一轨迹评估结果

```bash
python src/evaluation_pipeline.py
```

如果 Windows 上 `python` 命令不可用，可以把命令中的 `python` 改为 `py -3`。

### 6. 启动结果前端

```powershell
./start_frontend.ps1
```

也可以手动启动静态文件服务：

```powershell
python -m http.server 8081
```

浏览器打开 `http://127.0.0.1:8081/frontend/`。

### 最小运行流程

如果只想查看仓库已经保存的结果，可以跳过训练脚本，直接启动前端。如果想从头生成主要实验结果，依次执行：

```bash
python src/main.py
python src/active_querying.py
python src/evaluation_pipeline.py
```

## 前端演示

项目提供一个不依赖额外框架的结果仪表盘，用来查看 Q-Learning 训练曲线、三组多目标权重实验、轨迹排名和 Bradley-Terry 主动查询结果。前端读取 `results/` 中的现有 CSV 与 PNG，不会重新训练，也不会修改实验结果。

页面不是模型训练器，而是实验结果阅读器：左侧选择 benefit/cost/risk 权重方案，右侧查看奖励曲线、轨迹得分排行，以及 Bradley-Terry 概率最接近 0.5 的主动查询样本。

启动脚本只提供静态网页服务，无需安装 Node.js。

## 示例输出

### 轨迹奖励分数

`RewardModel` 会把轨迹的多目标统计合成为一个标量分数：

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

### 主动查询结果

`results/active_querying.txt` 会展示最值得人工标注的轨迹对：

```text
Most useful trajectory pair for human labeling:

trajectory_a: T02
reward_score: -6.100

trajectory_b: T08
reward_score: -6.100

P(A > B): 0.500000
uncertainty: 0.000000
```

概率越接近 0.5，说明模型越不确定。主动查询会优先选择这样的轨迹对，
因为人工标注它们最有信息量。
