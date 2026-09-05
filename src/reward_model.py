"""
Reward Model 模块。

这个文件实现一个简化版 RewardModel：
- 不训练神经网络
- 不引入 PyTorch 或 LangChain
- 只把 trajectory 的 benefit / cost / risk 分量合成为 reward_score

真正的加权计算复用 reward_utils.calculate_weighted_reward，
这样 RewardModel 和项目中已有的多目标 reward 逻辑保持一致。
"""

from reward_utils import calculate_weighted_reward


class RewardModel:
    """
    简化版 Reward Model。

    在完整 RLHF 或偏好学习系统中，reward model 通常会从人类偏好数据中学习。
    这里先用固定权重模拟 reward model 的打分行为，方便后续接入评估流程。
    """

    def __init__(
        self,
        benefit_weight=0.5,
        cost_weight=0.3,
        risk_weight=0.2,
    ):
        """保存 benefit / cost / risk 三个分量的权重。"""
        self.benefit_weight = benefit_weight
        self.cost_weight = cost_weight
        self.risk_weight = risk_weight

    def score_components(self, total_benefit, total_cost, total_risk):
        """
        根据累计 benefit / cost / risk 分量计算 reward_score。

        这里不重复实现加权公式，而是复用 reward_utils.calculate_weighted_reward。
        """
        return calculate_weighted_reward(
            total_benefit,
            total_cost,
            total_risk,
            self.benefit_weight,
            self.cost_weight,
            self.risk_weight,
        )

    def score_trajectory(self, trajectory):
        """
        为一条 trajectory 计算 reward_score。

        trajectory 至少需要包含：
            total_benefit
            total_cost
            total_risk
        """
        return self.score_components(
            trajectory["total_benefit"],
            trajectory["total_cost"],
            trajectory["total_risk"],
        )

    def explain_score(self, trajectory):
        """
        返回 reward_score 的可解释信息。

        这个结构方便后续 Evaluation Pipeline 或 README 展示：
        既能看到 trajectory 的 reward 分量，也能看到当前 RewardModel 使用的权重。
        """
        reward_score = self.score_trajectory(trajectory)
        return {
            "total_benefit": trajectory["total_benefit"],
            "total_cost": trajectory["total_cost"],
            "total_risk": trajectory["total_risk"],
            "benefit_weight": self.benefit_weight,
            "cost_weight": self.cost_weight,
            "risk_weight": self.risk_weight,
            "reward_score": reward_score,
        }


def main():
    """直接运行本文件时，演示 RewardModel 如何给一条 trajectory 打分。"""
    trajectory = {
        "trajectory_id": "demo_trajectory",
        "total_benefit": 10,
        "total_cost": -5,
        "total_risk": -6,
    }

    reward_model = RewardModel()
    reward_score = reward_model.score_trajectory(trajectory)
    explanation = reward_model.explain_score(trajectory)

    print("Reward Model Demo")
    print("-" * 40)
    print(f"trajectory_id: {trajectory['trajectory_id']}")
    print(f"reward_score: {reward_score:.3f}")
    print()
    print("explain_score:")
    for key, value in explanation.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
