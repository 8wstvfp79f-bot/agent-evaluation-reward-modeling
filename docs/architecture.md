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
Reward Model 打分
    |
    v
Bradley-Terry 偏好概率
    |
    v
Active Querying 主动查询
```

## 各层说明

### GridWorld 环境

定义 6x7 网格环境，包括起点、终点、多个障碍物、动作规则和原始环境
reward。它是整个项目中所有 trajectory 的来源。

### Q-Learning 训练

在 GridWorld 中学习一个表格型策略。agent 通过和环境交互更新
`Q(state, action)`，逐渐学会从起点移动到终点。

### 轨迹生成

使用训练后的 agent 生成候选 trajectories。每条 trajectory 会记录经过的
states、执行的 actions、原始 total reward，以及 benefit、cost、risk 三个
多目标 reward 分量。

### Reward Model 打分

把 trajectory 级别的 benefit、cost、risk 累计值转换成单一的
`reward_score`。当前版本是一个简单的加权 RewardModel。

### Bradley-Terry 偏好概率

把两个 trajectory 的 reward score 差值转换成偏好概率：

```text
P(A > B) = sigmoid(score_A - score_B)
```

### Active Querying 主动查询

遍历所有 trajectory pair，选择概率最接近 0.5 的 pair。这个 pair 就是
Maximum Disagreement pair，表示模型最不确定哪条轨迹更好，因此最值得
交给人工标注。
