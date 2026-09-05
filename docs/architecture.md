# 项目架构

```text
GridWorld 环境
    |
    v
Q-Learning 训练
    |
    v
轨迹生成
    |
    v
奖励模型打分
    |
    v
Bradley-Terry 偏好概率
    |
    v
主动查询
```

## 各层说明

### GridWorld 环境

定义 6×7 网格环境，包括起点、终点、多个障碍物、动作规则和原始环境
奖励。它是整个项目中所有轨迹的来源。

### Q-Learning 训练

在 GridWorld 中学习一个表格型策略。智能体通过和环境交互更新
`Q(state, action)`，逐渐学会从起点移动到终点。

### 轨迹生成

使用训练后的智能体生成候选轨迹。每条轨迹会记录经过的状态、执行的动作、
原始总奖励，以及 benefit、cost、risk 三个多目标奖励分量。

### 奖励模型打分

把轨迹级别的 benefit、cost、risk 累计值转换成单一的
`reward_score`。当前版本使用简单的固定权重 `RewardModel`。

### Bradley-Terry 偏好概率

把两条轨迹的奖励分数差值转换成偏好概率：

```text
P(A > B) = sigmoid(score_A - score_B)
```

### 主动查询

遍历所有轨迹对，选择概率最接近 0.5 的轨迹对。它表示模型最不确定哪条轨迹更好，因此最值得交给人工标注。
