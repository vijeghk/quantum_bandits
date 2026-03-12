import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple
from config import Config


class CombinatorialBanditEnv:
    """Combinatorial bandit environment"""

    def __init__(self, config, μ, θ):
        self.config = config
        self.n_arms = len(μ)
        self.combo_size = config.COMBO_SIZE

        # 真实参数
        self.true_μ = torch.tensor(μ, dtype=torch.float32)
        self.true_θ = torch.tensor(θ, dtype=torch.float32)

        # 计算最优组合
        self.optimal_combo, self.optimal_reward = self._find_optimal_combo()

    def _sigmoid(self, x):
        """Sigmoid"""
        return 1 / (1 + torch.exp(-x))

    def _compute_reward(self, combo: List[int]) -> float:
        """计算组合的期望奖励"""
        combo_tensor = torch.zeros(self.n_arms, dtype=torch.float32)
        combo_tensor[combo] = 1.0

        # 线性项
        linear_term = torch.sum(self.true_μ * combo_tensor)

        # 交互项
        inter_term = 0
        for i in range(self.n_arms):
            for j in range(i+1, self.n_arms):
                if combo_tensor[i] and combo_tensor[j]:
                    inter_term += self.true_θ[i, j]

        # 总期望
        expected = linear_term + inter_term
        return self._sigmoid(expected).item()

    def _find_optimal_combo(self):
        """寻找最优组合（穷举搜索）"""
        from itertools import combinations

        best_combo = None
        best_reward = -float('inf')

        all_combos = list(combinations(range(self.n_arms), self.combo_size))
        for combo in all_combos:
            reward = self._compute_reward(list(combo))
            if reward > best_reward:
                best_reward = reward
                best_combo = combo

        return list(best_combo), best_reward

    def step(self, combo: List[int]) -> Tuple[float, bool]:
        """
        执行一步动作

        Args:
            combo: 选择的组合

        Returns:
            reward: 观测到的奖励（带噪声）
            optimal: 是否是最优组合
        """
        # 计算期望奖励
        expected_reward = self._compute_reward(combo)

        # 添加高斯噪声
        noise = np.random.normal(0, 0.05)
        observed_reward = np.clip(expected_reward + noise, 0, 1)

        # 检查是否最优
        is_optimal = sorted(combo) == sorted(self.optimal_combo)

        return observed_reward, is_optimal

    def get_regret(self, combo: List[int]) -> float:
        """计算遗憾"""
        expected_reward = self._compute_reward(combo)
        return self.optimal_reward - expected_reward

    def get_context(self):
        """获取上下文信息（用于上下文赌博机）"""
        # 这里可以返回分子特征作为上下文
        return self.true_μ.numpy(), self.true_θ.numpy()

class NonstationaryEnv(CombinatorialBanditEnv):
    """非平稳环境 - 接口完全兼容基类，内部维护轮次计数器"""

    def __init__(self, config, μ, θ, change_points=3):
        super().__init__(config, μ, θ)
        self.change_points = change_points
        self.current_phase = 0
        self.current_round = 0  # 内部轮次计数器

        # 生成变化点
        total_rounds = config.N_ROUNDS
        change_intervals = total_rounds // (change_points + 1)
        self.changes = [i * change_intervals for i in range(1, change_points + 1)]

    def step(self, combo: List[int]):
        """
        完全兼容基类签名
        返回: (reward, is_optimal)
        """
        self.current_round += 1

        # 检查是否需要改变环境
        if self.current_round in self.changes:
            self._change_environment()
            self.current_phase += 1

        return super().step(combo)

    def _change_environment(self):
        """改变环境参数（同原实现）"""
        perturbation = torch.randn_like(self.true_μ) * 0.1
        self.true_μ += perturbation
        self.true_μ = torch.clamp(self.true_μ, 0, 1)
        self.optimal_combo, self.optimal_reward = self._find_optimal_combo()

class DelayedFeedbackEnv(CombinatorialBanditEnv):
    """延迟反馈环境 - 接口完全兼容基类，返回立即奖励，内部记录延迟队列"""

    def __init__(self, config, μ, θ, max_delay=10):
        super().__init__(config, μ, θ)
        self.max_delay = max_delay
        self.feedback_queue = []          # 存储延迟反馈信息
        self.current_round = 0

    def step(self, combo: List[int]):
        """
        完全兼容基类签名
        返回: (reward, is_optimal)  奖励为**立即观测值**，延迟信息仅内部记录
        """
        self.current_round += 1

        # 1. 计算期望奖励并添加噪声
        expected_reward = self._compute_reward(combo)
        noise = np.random.normal(0, 0.05)
        observed_reward = np.clip(expected_reward + noise, 0, 1)

        # 2. 检查是否最优
        is_optimal = sorted(combo) == sorted(self.optimal_combo)

        print(
            f"[DEBUG] combo={combo}, sorted={sorted(combo)}, optimal={self.optimal_combo}, sorted_optimal={sorted(self.optimal_combo)}")

        # 3. 模拟延迟反馈：随机延迟，并存入队列
        delay = np.random.randint(1, self.max_delay + 1)
        self.feedback_queue.append({
            'combo': combo.copy(),
            'reward': observed_reward,
            'delay': delay,
            'round_selected': self.current_round
        })

        # 4. 清理已到期的反馈（仅用于内部统计，不影响返回值）
        self._clean_feedback_queue()

        # 5. 返回**立即**奖励（符合基类契约）
        return observed_reward, is_optimal

    def _clean_feedback_queue(self):
        """减少延迟计数，移除已到期的反馈"""
        remaining = []
        for fb in self.feedback_queue:
            fb['delay'] -= 1
            if fb['delay'] > 0:
                remaining.append(fb)
        self.feedback_queue = remaining

    def get_pending_feedback(self):
        """获取尚未交付的反馈数量（辅助函数）"""
        return len(self.feedback_queue)
    