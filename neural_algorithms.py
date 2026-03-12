import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations
from typing import List, Optional, Tuple
from config import Config
from torch.cuda.amp import autocast, GradScaler

class NeuralNetworkBase(nn.Module):
    """Base neural network: input → hidden layers → scalar output"""
    def __init__(self, config: Config, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.NEURAL_DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.NEURAL_DROPOUT),
            nn.Linear(hidden_dim // 2, 1)
        ).to(config.DEVICE)

    def forward(self, x):
        return self.network(x)


class NeuralTS:
    def __init__(self, config, n_arms, context_dim):
        self.config = config
        self.n_arms = n_arms
        self.combo_size = config.COMBO_SIZE
        self.context_dim = context_dim
        self.ensemble_size = 5

        # Input: context, output: score for each arm
        self.input_dim = context_dim
        self.models = []
        self.optimizers = []
        for _ in range(self.ensemble_size):
            model = nn.Sequential(
                nn.Linear(self.input_dim, 128),
                nn.ReLU(),
                nn.Dropout(config.NEURAL_DROPOUT),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, n_arms)
            ).to(config.DEVICE)
            self.models.append(model)
            self.optimizers.append(torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE))

        self.replay_buffer = []
        self.buffer_size = 500
        self.batch_size = 32
        self.train_step = 0
        self.update_freq = 1

    def select_combo(self, context):
        model_idx = np.random.randint(self.ensemble_size)
        model = self.models[model_idx]
        model.eval()
        ctx_t = torch.tensor(context, dtype=torch.float32, device=self.config.DEVICE).unsqueeze(0)
        with torch.no_grad():
            scores = model(ctx_t).squeeze().cpu().numpy()
        selected = np.argsort(scores)[-self.combo_size:].tolist()
        return selected

    def update(self, combo, reward, context):
        self.replay_buffer.append((context.copy(), combo.copy(), reward))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
        self.train_step += 1
        if self.train_step % self.update_freq == 0 and len(self.replay_buffer) >= self.batch_size:
            self._train_batch()

    def _train_batch(self):
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch_ctx = [self.replay_buffer[i][0] for i in indices]
        batch_combo = [self.replay_buffer[i][1] for i in indices]
        batch_reward = [self.replay_buffer[i][2] for i in indices]

        ctx_t = torch.tensor(np.array(batch_ctx), dtype=torch.float32, device=self.config.DEVICE)
        targets = torch.zeros((self.batch_size, self.n_arms), device=self.config.DEVICE)
        for i, (combo, reward) in enumerate(zip(batch_combo, batch_reward)):
            targets[i, combo] = reward

        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()
            preds = model(ctx_t)
            loss = F.mse_loss(preds, targets)
            loss.backward()
            optimizer.step()


class DeepUCB:
    """
    Deep UCB - supports two modes (switched by config.NEURAL_USE_ENUMERATION):
    - Mode A (enumeration, high precision): input one-hot+context, output scalar, enumerates all combos to compute UCB.
    - Mode B (arm-level scoring, fast): input context, output n_arms scores, top‑k selection.
    """
    def __init__(self, config: Config, n_arms: int, context_dim: int):
        self.config = config
        self.n_arms = n_arms
        self.combo_size = config.COMBO_SIZE
        self.context_dim = context_dim
        self.ensemble_size = 5
        self.beta = config.BETA
        self.use_enumeration = getattr(config, 'NEURAL_USE_ENUMERATION', False)  # 默认False（臂级得分）
        self.scaler = GradScaler() if config.DEVICE == 'cuda' else None

        # Choose network structure according to mode
        if self.use_enumeration:
            # Enumeration version: input = one-hot + context
            self.input_dim = n_arms + context_dim
            self.output_dim = 1
        else:
            # Arm-level scoring version: input = context, output = score per arm
            self.input_dim = context_dim
            self.output_dim = n_arms

        # Create ensemble networks and optimizers
        self.models = []
        self.optimizers = []
        for _ in range(self.ensemble_size):
            model = nn.Sequential(
                nn.Linear(self.input_dim, 128),
                nn.ReLU(),
                nn.Dropout(config.NEURAL_DROPOUT),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.output_dim)
            ).to(config.DEVICE)
            self.models.append(model)
            self.optimizers.append(torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE))

        # Experience replay
        self.replay_buffer = []
        self.buffer_size = 500
        self.batch_size = 32
        self.train_step = 0
        self.update_freq = 1

        if self.use_enumeration:
            from itertools import combinations
            self.all_combos = list(combinations(range(n_arms), config.COMBO_SIZE))
            self.n_combos = len(self.all_combos)
            self.t = 0

    def _combo_to_feature(self, combo, context):
        if self.use_enumeration:
            one_hot = np.zeros(self.n_arms)
            one_hot[list(combo)] = 1.0
            return np.concatenate([one_hot, context])
        else:
            raise NotImplementedError("Arm-level scoring version does not need this method")

    # Decision
    def select_combo(self, context):
        if self.use_enumeration:
            self.t += 1
            features = []
            for combo in self.all_combos:
                feat = self._combo_to_feature(combo, context)
                features.append(feat)
            features_t = torch.tensor(np.array(features), dtype=torch.float32, device=self.config.DEVICE)

            all_preds = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    preds = model(features_t).squeeze(-1).cpu().numpy()
                all_preds.append(preds)
            all_preds = np.array(all_preds)

            mean_pred = np.mean(all_preds, axis=0)
            std_pred = np.std(all_preds, axis=0, ddof=1)
            ucb = mean_pred + self.beta * std_pred
            best_idx = np.argmax(ucb)
            return list(self.all_combos[best_idx])

        else:
            ctx_t = torch.tensor(context, dtype=torch.float32, device=self.config.DEVICE).unsqueeze(0)

            all_scores = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    scores = model(ctx_t).squeeze().cpu().numpy()
                all_scores.append(scores)
            all_scores = np.array(all_scores)

            mean_scores = np.mean(all_scores, axis=0)
            std_scores = np.std(all_scores, axis=0, ddof=1)
            ucb_scores = mean_scores + self.beta * std_scores   # (n_arms,)
            selected = np.argsort(ucb_scores)[-self.combo_size:].tolist()
            return selected

    # Update
    def update(self, combo, reward, context):
        self.replay_buffer.append((context.copy(), combo.copy(), reward))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

        self.train_step += 1
        if self.train_step % self.update_freq == 0 and len(self.replay_buffer) >= self.batch_size:
            self._train_batch()

    # Batch training
    def _train_batch(self):
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch_context = [self.replay_buffer[i][0] for i in indices]
        batch_combo = [self.replay_buffer[i][1] for i in indices]
        batch_reward = [self.replay_buffer[i][2] for i in indices]

        if self.use_enumeration:
            features = []
            for ctx, com in zip(batch_context, batch_combo):
                feat = self._combo_to_feature(com, ctx)
                features.append(feat)
            features_t = torch.tensor(np.array(features), dtype=torch.float32, device=self.config.DEVICE)
            targets_t = torch.tensor(batch_reward, dtype=torch.float32, device=self.config.DEVICE).unsqueeze(1)
        else:
            features_t = torch.tensor(np.array(batch_context), dtype=torch.float32, device=self.config.DEVICE)
            targets = torch.zeros((self.batch_size, self.n_arms), device=self.config.DEVICE)
            for i, (com, rew) in enumerate(zip(batch_combo, batch_reward)):
                targets[i, com] = rew
            targets_t = targets

        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

            if self.scaler is not None:
                # Mixed precision forward pass
                with autocast():
                    preds = model(features_t)  # features_t already on GPU
                    loss = F.mse_loss(preds, targets_t)

                # Backward pass (scaler automatically scales gradients)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                preds = model(features_t)
                loss = F.mse_loss(preds, targets_t)
                loss.backward()
                optimizer.step()


class AttentionBandit:
    """
    Attention-based combinatorial bandit
    - Feeds a sequence of combo features into an attention layer, aggregates, then predicts reward
    - Input features: combo one‑hot + context
    """
    def __init__(self, config: Config, n_arms: int, feature_dim: int):
        self.config = config
        self.n_arms = n_arms
        self.combo_size = config.COMBO_SIZE
        self.feature_dim = feature_dim
        self.all_combos = list(combinations(range(n_arms), config.COMBO_SIZE))
        self.n_combos = len(self.all_combos)

        self.input_dim = n_arms + feature_dim

        self.attention = nn.MultiheadAttention(
            embed_dim=self.input_dim, num_heads=4,
            dropout=config.NEURAL_DROPOUT, batch_first=True
        ).to(config.DEVICE)

        self.predictor = nn.Sequential(
            nn.Linear(self.input_dim, config.NEURAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.NEURAL_DROPOUT),
            nn.Linear(config.NEURAL_HIDDEN_DIM, 1),
            nn.Sigmoid()
        ).to(config.DEVICE)

        self.optimizer = torch.optim.Adam(
            list(self.attention.parameters()) + list(self.predictor.parameters()),
            lr=config.LEARNING_RATE
        )

        self.replay_buffer = []
        self.buffer_size = 500
        self.batch_size = 16
        self.train_step = 0
        self.update_freq = 1

    def _combo_to_feature(self, combo, context):
        one_hot = np.zeros(self.n_arms)
        one_hot[list(combo)] = 1.0
        return np.concatenate([one_hot, context])

    def select_combo(self, context):
        """Construct feature sequence for all combos, apply attention, predict, take maximum"""
        features = []
        for combo in self.all_combos:
            feat = self._combo_to_feature(combo, context)
            features.append(feat)
        features_t = torch.tensor(np.array(features), dtype=torch.float32, device=self.config.DEVICE)
        features_t = features_t.unsqueeze(0)

        with torch.no_grad():
            attn_out, _ = self.attention(features_t, features_t, features_t)
            attn_out = attn_out.squeeze(0)
            scores = self.predictor(attn_out).squeeze(-1).cpu().numpy()

        best_idx = np.argmax(scores)
        return list(self.all_combos[best_idx])

    def update(self, combo, reward, context):
        self.replay_buffer.append((context.copy(), combo.copy(), reward))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

        self.train_step += 1
        if self.train_step % self.update_freq == 0 and len(self.replay_buffer) >= self.batch_size:
            self._train_batch()

    def _train_batch(self):
        """During training: for each sample, only feed the feature of the chosen combo to predict reward"""
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch_context = [self.replay_buffer[i][0] for i in indices]
        batch_combo   = [self.replay_buffer[i][1] for i in indices]
        batch_reward  = [self.replay_buffer[i][2] for i in indices]

        features = []
        for ctx, com in zip(batch_context, batch_combo):
            feat = self._combo_to_feature(com, ctx)
            features.append(feat)
        features_t = torch.tensor(np.array(features), dtype=torch.float32, device=self.config.DEVICE).unsqueeze(1)
        targets_t = torch.tensor(batch_reward, dtype=torch.float32, device=self.config.DEVICE).unsqueeze(1)

        self.optimizer.zero_grad()
        attn_out, _ = self.attention(features_t, features_t, features_t)
        attn_out = attn_out.squeeze(1)
        preds = self.predictor(attn_out)
        loss = F.mse_loss(preds, targets_t)
        loss.backward()
        self.optimizer.step()

class NeuralUCB:
    """
    NeuralUCB algorithm (Zhou et al., 2020) with gradient features.
    For each context x, the network outputs f(x, a) for each arm a.
    The gradient of f(x, a) w.r.t. the last layer parameters is used as feature.
    A shared covariance matrix is maintained for all arms.
    """
    def __init__(self, config: Config, n_arms: int, context_dim: int,
                 hidden_dim: int = 100, epochs: int = 1):
        self.config = config
        self.n_arms = n_arms
        self.combo_size = config.COMBO_SIZE
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.device = config.DEVICE
        self.beta = config.BETA
        self.lambda_ = 1.0  # regularization parameter

        # Define neural network: input context, output n_arms scores
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_arms)
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.LEARNING_RATE)

        # Covariance matrix and vector (for ridge regression)
        # Number of parameters = parameters of the last layer (including bias)
        last_layer = self.net[-1]  # Linear layer
        self.d = last_layer.in_features * last_layer.out_features + last_layer.out_features
        self.A = self.lambda_ * torch.eye(self.d, device=self.device)
        self.b = torch.zeros(self.d, device=self.device)

        # Store historical data for network training
        self.replay_buffer: List[Tuple[np.ndarray, List[int], float]] = []
        self.buffer_size = 500
        self.batch_size = 32
        self.train_step = 0
        self.update_freq = 10

        # Cache last gradient features
        self.last_features = None

    def _get_features(self, context: np.ndarray, arm: int) -> np.ndarray:
        """
        Compute the gradient of the network output for a specific arm w.r.t.
        the last layer parameters. Returns a feature vector of length d as a numpy array.
        """
        ctx_t = torch.tensor(context, dtype=torch.float32, device=self.device).unsqueeze(0)
        ctx_t.requires_grad_(True)  # need gradient

        # Forward pass
        output = self.net(ctx_t)
        # Take output for the specific arm
        arm_output = output[0, arm]

        # Compute gradient
        self.net.zero_grad()
        arm_output.backward(retain_graph=True)

        # Extract gradients of the last layer
        last_layer = self.net[-1]
        # Weight gradient
        weight_grad = last_layer.weight.grad[arm, :].detach().cpu().numpy()
        # Bias gradient
        bias_grad = last_layer.bias.grad[arm].detach().cpu().numpy() if last_layer.bias is not None else np.array([])

        # Concatenate into feature vector
        feature = np.concatenate([weight_grad, bias_grad])
        return feature

    def select_combo(self, context: np.ndarray) -> List[int]:
        ctx_t = torch.tensor(context, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            scores = self.net(ctx_t).squeeze().cpu().numpy()  # (n_arms,)

        # Compute UCB for each arm
        ucb_scores = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            phi = self._get_features(context, arm)
            phi_t = torch.tensor(phi, device=self.device).float()
            # Mean = network output (which is scores[arm])
            mean = scores[arm]
            # Uncertainty = beta * sqrt( phi^T A^{-1} phi )
            try:
                A_inv_phi = torch.linalg.solve(self.A, phi_t)  # A^{-1} * phi
                uncertainty = self.beta * torch.sqrt(phi_t @ A_inv_phi).item()
            except:
                uncertainty = 0.0
            ucb_scores[arm] = mean + uncertainty

        # Select top-k arms by UCB score
        selected = np.argsort(ucb_scores)[-self.combo_size:].tolist()
        return selected

    def update(self, combo: List[int], reward: float, context: np.ndarray):
        # Store experience
        self.replay_buffer.append((context.copy(), combo.copy(), reward))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

        # For each arm in the combo, update covariance matrix
        for arm in combo:
            phi = self._get_features(context, arm)
            phi_t = torch.tensor(phi, device=self.device).float()
            self.A += torch.outer(phi_t, phi_t)
            self.b += reward * phi_t

        # Periodically train the neural network
        self.train_step += 1
        if self.train_step % self.update_freq == 0 and len(self.replay_buffer) >= self.batch_size:
            self._train_network()

    def _train_network(self):
        """Train the neural network using experience replay, minimizing MSE"""
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch_ctx = [self.replay_buffer[i][0] for i in indices]
        batch_combo = [self.replay_buffer[i][1] for i in indices]
        batch_reward = [self.replay_buffer[i][2] for i in indices]

        ctx_t = torch.tensor(np.array(batch_ctx), dtype=torch.float32, device=self.device)
        targets = torch.zeros((self.batch_size, self.n_arms), device=self.device)
        for i, (combo, rew) in enumerate(zip(batch_combo, batch_reward)):
            targets[i, combo] = rew

        self.net.train()
        self.optimizer.zero_grad()
        preds = self.net(ctx_t)
        loss = F.mse_loss(preds, targets)
        loss.backward()
        self.optimizer.step()
