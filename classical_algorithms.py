"""
Classical Combinatorial Bandit Algorithms
"""
import numpy as np
import itertools
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct
from config import Config
from abc import ABC, abstractmethod

# ---------- Base class ----------
class BaseBandit(ABC):
    def __init__(self, config, n_arms):
        self.config = config
        self.n_arms = n_arms
        self.combo_size = config.COMBO_SIZE
        self.arm_counts = np.zeros(n_arms)
        self.total_reward = np.zeros(n_arms)

    @abstractmethod
    def select_combo(self):
        pass

    @abstractmethod
    def update(self, combo, reward):
        pass

    def _combo_to_key(self, combo):
        return tuple(sorted(combo))


# Combinatorial CMAB Algorithms
class CombinatorialUCB(BaseBandit):
    """Combinatorial UCB (CUCB)"""
    def __init__(self, config, n_arms):
        super().__init__(config, n_arms)
        self.all_combos = list(itertools.combinations(range(n_arms), config.COMBO_SIZE))
        self.combo_counts = {combo: 0 for combo in self.all_combos}
        self.combo_rewards = {combo: 0.0 for combo in self.all_combos}
        self.t = 0

    def select_combo(self):
        self.t += 1
        best_combo = None
        best_score = -float('inf')
        for combo in self.all_combos:
            if self.combo_counts[combo] == 0:
                score = float('inf')
            else:
                mean = self.combo_rewards[combo] / self.combo_counts[combo]
                bonus = self.config.BETA * np.sqrt(2 * np.log(self.t) / self.combo_counts[combo])
                score = mean + bonus
            if score > best_score:
                best_score = score
                best_combo = combo
        return list(best_combo)

    def update(self, combo, reward):
        combo_key = tuple(sorted(combo))
        self.combo_counts[combo_key] += 1
        self.combo_rewards[combo_key] += reward


class CombinatorialThompsonSampling(BaseBandit):
    """Combinatorial Thompson Sampling (CTS)"""
    def __init__(self, config, n_arms):
        super().__init__(config, n_arms)
        self.all_combos = list(itertools.combinations(range(n_arms), config.COMBO_SIZE))
        self.alpha = {combo: config.TS_PRIOR_ALPHA for combo in self.all_combos}
        self.beta = {combo: config.TS_PRIOR_BETA for combo in self.all_combos}

    def select_combo(self):
        # Sample from each combo's Beta distribution, take maximum
        sampled = {combo: np.random.beta(self.alpha[combo], self.beta[combo])
                   for combo in self.all_combos}
        best_combo = max(sampled, key=sampled.get)
        return list(best_combo)

    def update(self, combo, reward):
        combo_key = tuple(sorted(combo))
        if reward > 0.5:
            self.alpha[combo_key] += 1
        else:
            self.beta[combo_key] += 1


class CombinatorialLinUCB(BaseBandit):
    """Combinatorial Linear UCB (CLinUCB)"""

    def __init__(self, config, n_arms, context_dim):
        super().__init__(config, n_arms)
        self.context_dim = context_dim
        self.all_combos = list(itertools.combinations(range(n_arms), config.COMBO_SIZE))
        self.n_combos = len(self.all_combos)
        self.feature_dim = context_dim + n_arms

        # Maintain inverse matrix and b vector for each combo
        self.A_inv = [np.eye(self.feature_dim) for _ in range(self.n_combos)]
        self.b = [np.zeros(self.feature_dim) for _ in range(self.n_combos)]
        self.alpha = 1.0

    def _combo_to_feature(self, combo, context):
        """Combo feature: context + combo one-hot"""
        combo_onehot = np.zeros(self.n_arms)
        combo_onehot[list(combo)] = 1
        return np.concatenate([context, combo_onehot])

    def select_combo(self, context):
        best_combo = None
        best_score = -float('inf')
        for idx, combo in enumerate(self.all_combos):
            x = self._combo_to_feature(combo, context)
            # theta = A_inv @ b
            theta = self.A_inv[idx] @ self.b[idx]
            pred = np.dot(theta, x)
            # uncertainty = sqrt(x^T A_inv x)
            uncertainty = np.sqrt(np.dot(x, self.A_inv[idx] @ x))
            score = pred + self.alpha * uncertainty
            if score > best_score:
                best_score = score
                best_combo = combo
        return list(best_combo)

    def update(self, combo, reward, context):
        combo_key = tuple(sorted(combo))
        idx = self.all_combos.index(combo_key)
        x = self._combo_to_feature(combo, context)

        # Update b
        self.b[idx] += reward * x

        # Sherman-Morrison update of A_inv
        v = self.A_inv[idx] @ x
        denom = 1 + np.dot(x, v)
        if abs(denom) > 1e-12:  # avoid division by zero
            self.A_inv[idx] -= np.outer(v, v) / denom


# Per-arm Bandit Baselines
class ClassicalTS(BaseBandit):
    """Classical Thompson Sampling"""
    def __init__(self, config, n_arms):
        super().__init__(config, n_arms)
        self.alpha = np.ones(n_arms) * config.TS_PRIOR_ALPHA
        self.beta = np.ones(n_arms) * config.TS_PRIOR_BETA

    def select_combo(self):
        theta = np.random.beta(self.alpha, self.beta)
        selected = np.argsort(theta)[-self.combo_size:]
        return list(selected)

    def update(self, combo, reward):
        for arm in combo:
            self.arm_counts[arm] += 1
            self.total_reward[arm] += reward
            if reward > 0.5:
                self.alpha[arm] += 1
            else:
                self.beta[arm] += 1


class ClassicalUCB(BaseBandit):
    """Classical UCB"""
    def __init__(self, config, n_arms):
        super().__init__(config, n_arms)
        self.t = 0

    def select_combo(self):
        self.t += 1
        ucb = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            if self.arm_counts[arm] == 0:
                ucb[arm] = float('inf')
            else:
                mean = self.total_reward[arm] / self.arm_counts[arm]
                bonus = self.config.BETA * np.sqrt(2 * np.log(self.t) / self.arm_counts[arm])
                ucb[arm] = mean + bonus
        selected = np.argsort(ucb)[-self.combo_size:]
        return list(selected)

    def update(self, combo, reward):
        for arm in combo:
            self.arm_counts[arm] += 1
            self.total_reward[arm] += reward


# Combinatorial Algorithms
class C2UCB(BaseBandit):
    """C2UCB"""
    def __init__(self, config, n_arms):
        super().__init__(config, n_arms)
        self.all_combos = list(itertools.combinations(range(n_arms), config.COMBO_SIZE))
        self.combo_counts = {combo: 0 for combo in self.all_combos}
        self.combo_rewards = {combo: 0 for combo in self.all_combos}
        self.t = 0

    def select_combo(self):
        self.t += 1
        best_combo = None
        best_score = -float('inf')
        for combo, count in self.combo_counts.items():
            if count == 0:
                score = float('inf')
            else:
                mean = self.combo_rewards[combo] / count
                bonus = np.sqrt(2 * np.log(self.t) / count)
                score = mean + bonus
            if score > best_score:
                best_score = score
                best_combo = combo
        return list(best_combo)

    def update(self, combo, reward):
        combo_key = tuple(sorted(combo))
        self.combo_counts[combo_key] += 1
        self.combo_rewards[combo_key] += reward
        for arm in combo:
            self.arm_counts[arm] += 1
            self.total_reward[arm] += reward


class EXP3(BaseBandit):
    """EXP3"""
    def __init__(self, config, n_arms):
        super().__init__(config, n_arms)
        self.all_combos = list(itertools.combinations(range(n_arms), config.COMBO_SIZE))
        self.n_combos = len(self.all_combos)
        self.weights = np.ones(self.n_combos)
        self.gamma = 0.1
        self.eta = np.sqrt(np.log(self.n_combos) / (self.n_combos * config.N_ROUNDS))

    def select_combo(self):
        probs = (1 - self.gamma) * self.weights / np.sum(self.weights) + self.gamma / self.n_combos
        idx = np.random.choice(self.n_combos, p=probs)
        return list(self.all_combos[idx])

    def update(self, combo, reward):
        combo_key = tuple(sorted(combo))
        idx = self.all_combos.index(combo_key)
        probs = (1 - self.gamma) * self.weights / np.sum(self.weights) + self.gamma / self.n_combos
        loss = (1 - reward) / probs[idx]
        self.weights[idx] *= np.exp(-self.eta * loss)


class Hedge(BaseBandit):
    """Hedge"""
    def __init__(self, config, n_arms):
        super().__init__(config, n_arms)
        self.all_combos = list(itertools.combinations(range(n_arms), config.COMBO_SIZE))
        self.n_combos = len(self.all_combos)
        self.weights = np.ones(self.n_combos)
        self.eta = np.sqrt(8 * np.log(self.n_combos) / config.N_ROUNDS)

    def select_combo(self):
        probs = self.weights / np.sum(self.weights)
        idx = np.random.choice(self.n_combos, p=probs)
        return list(self.all_combos[idx])

    def update(self, combo, reward):
        combo_key = tuple(sorted(combo))
        idx = self.all_combos.index(combo_key)
        for i in range(self.n_combos):
            if i == idx:
                self.weights[i] *= np.exp(self.eta * reward)
            else:
                self.weights[i] *= np.exp(self.eta * 0)


class GPUCB(BaseBandit):
    """GP-UCB"""
    def __init__(self, config, n_arms):
        super().__init__(config, n_arms)
        kernel = RBF(length_scale=1.0) + DotProduct(sigma_0=1.0)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=5)
        self.X_history = []
        self.y_history = []
        self.beta_t = 2.0

    def _combo_to_feature(self, combo):
        feature = np.zeros(self.n_arms)
        feature[list(combo)] = 1
        return feature

    def select_combo(self):
        if len(self.X_history) < 10:
            return list(np.random.choice(self.n_arms, self.combo_size, replace=False))
        X = np.array(self.X_history)
        y = np.array(self.y_history)
        self.gp.fit(X, y)
        all_combos = itertools.combinations(range(self.n_arms), self.combo_size)
        best_combo = None
        best_score = -float('inf')
        for combo in all_combos:
            x = self._combo_to_feature(combo).reshape(1, -1)
            mean, std = self.gp.predict(x, return_std=True)
            score = mean[0] + self.beta_t * std[0]
            if score > best_score:
                best_score = score
                best_combo = combo
        return list(best_combo)

    def update(self, combo, reward):
        feature = self._combo_to_feature(combo)
        self.X_history.append(feature)
        self.y_history.append(reward)
        t = len(self.X_history)
        self.beta_t = 2 * np.log(t ** 2 * np.pi ** 2 / (6 * 0.1))

class UCBS(BaseBandit):
    """
    UCB-S algorithm for structured bandits (Lattimore & Munos, 2014).
    Requires:
        - reward_funcs: list of callable, each mapping theta (scalar/vector) to mean reward.
        - theta_space: list of possible theta values (discretized).
    """
    def __init__(self, config, n_arms, reward_funcs, theta_space):
        super().__init__(config, n_arms)
        self.reward_funcs = reward_funcs          # list of functions
        self.theta_space = theta_space            # list of theta candidates
        self.t = 0
        self.empirical_means = np.zeros(n_arms)   # empirical average reward
        self.n_pulls = np.zeros(n_arms)

    def select_combo(self):
        self.t += 1
        # Build confidence set Theta_hat
        theta_hat = self._build_confidence_set()
        if not theta_hat:
            # if confidence set empty, fallback to full space
            theta_hat = self.theta_space

        # Compute supremum over confidence set for each arm
        scores = []
        for k in range(self.n_arms):
            sup_val = max(self.reward_funcs[k](theta) for theta in theta_hat)
            scores.append(sup_val)

        # Select top m arms
        selected = np.argsort(scores)[-self.combo_size:]
        return list(selected)

    def _build_confidence_set(self):
        """Return all theta satisfying: empirical mean of each arm is within its confidence interval"""
        theta_hat = []
        for theta in self.theta_space:
            in_set = True
            for k in range(self.n_arms):
                if self.n_pulls[k] > 0:
                    # confidence radius (using UCB1 radius)
                    radius = np.sqrt(2 * np.log(self.t) / self.n_pulls[k])
                    if abs(self.reward_funcs[k](theta) - self.empirical_means[k]) > radius:
                        in_set = False
                        break
            if in_set:
                theta_hat.append(theta)
        return theta_hat

    def update(self, combo, reward):
        for k in combo:
            self.n_pulls[k] += 1
            # incremental mean update
            self.empirical_means[k] += (reward - self.empirical_means[k]) / self.n_pulls[k]


class GLMUCB(BaseBandit):
    """
    Generalized Linear Model UCB (Filippi et al., 2010).
    Assumes linear predictor (identity link) for simplicity.
    Requires: arm_features (n_arms x d) array.
    """
    def __init__(self, config, n_arms, context_dim, arm_features):
        super().__init__(config, n_arms)
        self.d = context_dim
        self.arm_features = arm_features            # shape (n_arms, d)
        self.lambda_ = 1.0                          # regularization
        self.A = [np.eye(self.d) * self.lambda_ for _ in range(n_arms)]
        self.b = [np.zeros(self.d) for _ in range(n_arms)]
        self.theta_hat = [np.zeros(self.d) for _ in range(n_arms)]
        self.t = 0

    def select_combo(self, context):
        self.t += 1
        scores = []
        for k in range(self.n_arms):
            x = self.arm_features[k]
            mean = np.dot(self.theta_hat[k], x)
            std = np.sqrt(np.dot(x, np.linalg.solve(self.A[k], x)))
            scores.append(mean + self.config.BETA * std)
        selected = np.argsort(scores)[-self.combo_size:]
        return list(selected)

    def update(self, combo, reward, context):
        for k in combo:
            x = self.arm_features[k]
            self.A[k] += np.outer(x, x)
            self.b[k] += reward * x
            self.theta_hat[k] = np.linalg.solve(self.A[k], self.b[k])
