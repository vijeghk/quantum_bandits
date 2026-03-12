"""
Quantum-enhanced Combinatorial Bandit Algorithms
=======================================
1. QuantumTS      : Quantum Thompson Sampling (Full Bayesian)
2. QuantumCUCB    : Quantum Combinatorial UCB (Fisher Information Uncertainty)
3. QuantumNeural  : Quantum-Neural Hybrid Architecture (End-to-End Training)
"""

import gc

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
import warnings
from itertools import combinations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import Estimator, Sampler
from qiskit_algorithms.optimizers import SPSA
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator as AerEstimator, Sampler as AerSampler
from qiskit_aer import Aer

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

class QuantumComboBanditBase:
    """Base class for quantum combinatorial bandits"""

    def __init__(self, config: Config, n_arms: int,
                 similarity_matrix: Optional[np.ndarray] = None):
        self.config = config
        self.n_arms = n_arms
        self.n_qubits = config.N_QUBITS
        self.combo_size = config.COMBO_SIZE
        self.similarity_matrix = similarity_matrix
        self._ent_param_map = {}   # Store entanglement pairs -> parameter index list

        # ---------- Combinatorial space ----------
        self.all_combos = list(combinations(range(self.n_arms), self.combo_size))
        self.n_combos = len(self.all_combos)

        # ---------- Combo-level statistics (for exploration/history) ----------
        self.combo_counts = {combo: 0 for combo in self.all_combos}
        self.combo_rewards = {combo: 0.0 for combo in self.all_combos}

        # ---------- Build parameterized circuits ----------
        self._build_circuits()

        # ---------- Parameter initialization (uniform random) ----------
        self._param_values = np.random.uniform(0, 2 * np.pi,
                                               size=len(self._param_vector))

        N = 1 << self.n_qubits
        # Precompute mask matrix shape (n_arms, N)
        indices = np.arange(N, dtype=np.int32)
        self.bit_masks = np.array([(indices >> i) & 1 for i in range(self.n_arms)], dtype=bool)
        self.all_states = np.arange(1 << self.n_qubits, dtype=np.uint32)  # 长度为 2^n 的数组

        # ---------- SPSA optimizer (delayed calibration) ----------
        self.optimizer: Optional[SPSA] = None
        self._optimization_step = 0
        self._loss_history = []

        # SPSA hyperparameters (tunable)
        self.spsa_a = 0.2
        self.spsa_c = 0.1
        self.spsa_alpha = 0.602
        self.spsa_gamma = 0.101
        self.spsa_A = 0

        self.spsa_c = getattr(config, 'SPSA_C', 0.1)
        self.fisher_update_freq = getattr(config, 'FISHER_UPDATE_FREQ', 200)

        # Create statevector backend
        self.backend = Aer.get_backend('statevector_simulator')
        # Enable GPU
        try:
            self.backend.set_options(device='GPU')
        except Exception as e:
            print(f"Warning: Failed to set GPU for statevector_simulator: {e}")
            self.backend.set_options(device='CPU')

        print(f"Using device: {self.backend.options.device}")

        # Select simulator
        method = getattr(config, 'SIMULATION_METHOD', 'statevector')
        device = getattr(config, 'SIMULATION_DEVICE', 'GPU')

        if method == 'statevector':
            self.simulator = AerSimulator(**{'method': 'statevector', 'device': device})
            self.estimator = AerEstimator(backend_options={'device': device})
            self.sampler = AerSampler(backend_options={'device': device})
        elif method == 'matrix_product_state':
            self.simulator = AerSimulator(**{'method': 'matrix_product_state', 'device': 'CPU'})
            self.estimator = AerEstimator()  # Estimator may not support MPS, needs adjustment
            self.sampler = AerSampler()
        else:
            raise ValueError(f"Unknown simulation method: {method}")

        self._optimizer_calibrated = False


    # Circuit construction
    def _build_circuits(self):
        """Construct parameterized circuit without measurement + sampling circuit with measurement"""
        qr = QuantumRegister(self.n_qubits, 'q')
        param_circuit = QuantumCircuit(qr)

        # ----- 1. Determine entanglement pairs (only act on first n_arms qubits) -----
        if not hasattr(self, '_entangling_pairs') or self._entangling_pairs is None:
            if self.config.ENTANGLEMENT_STRUCTURE == 'full':
                pairs = [(i, j) for i in range(self.n_arms)
                         for j in range(i + 1, self.n_arms)]
            elif self.config.ENTANGLEMENT_STRUCTURE == 'similarity':
                if self.similarity_matrix is None:
                    raise ValueError("Similarity matrix must be provided")
                pairs = []
                for i in range(self.n_arms):
                    for j in range(i + 1, self.n_arms):
                        if self.similarity_matrix[i, j] > self.config.ENTANGLEMENT_THRESHOLD:
                            pairs.append((i, j))
            else:
                raise ValueError(f"Unknown entanglement structure: {self.config.ENTANGLEMENT_STRUCTURE}")
            self._entangling_pairs = pairs
        else:
            pairs = self._entangling_pairs

        # ----- 2. Calculate total number of parameters (rotations + entanglement + extra layers) -----
        n_rot = self.n_arms
        n_ent = len(pairs)
        extra_layers = max(0, self.config.DEPTH - 1)
        extra_rot = extra_layers * self.n_arms
        extra_ent = extra_layers * n_ent
        total_params = n_rot + n_ent + extra_rot + extra_ent
        self._param_vector = ParameterVector('θ', total_params)

        # ----- 3. Build circuit (use parameters sequentially, no out-of-bounds check needed) -----
        param_idx = 0
        self._ent_param_map = {}

        # First layer: rotations
        for i in range(self.n_arms):
            param_circuit.ry(self._param_vector[param_idx], qr[i])
            param_idx += 1

        # First layer: entanglement
        for i, j in pairs:
            # Record current entanglement parameter index
            self._ent_param_map.setdefault((i, j), []).append(param_idx)
            param_circuit.cp(self._param_vector[param_idx], qr[i], qr[j])
            param_idx += 1

        # Extra layers
        for _ in range(extra_layers):
            for i in range(self.n_arms):
                param_circuit.ry(self._param_vector[param_idx], qr[i])
                param_idx += 1
            for i, j in pairs:
                # Record entanglement parameter index for extra layers
                self._ent_param_map.setdefault((i, j), []).append(param_idx)
                param_circuit.cp(self._param_vector[param_idx], qr[i], qr[j])
                param_idx += 1

        self.param_circuit = param_circuit

        # ----- Sampling circuit (for Q-TS) -----
        sampling_circuit = param_circuit.copy()
        cr = ClassicalRegister(self.n_qubits, 'c')
        sampling_circuit.add_register(cr)
        sampling_circuit.measure(qr, cr)
        self.sampling_circuit = sampling_circuit


    # Quantum state and probability computation
    def _get_statevector(self, param_values: Optional[np.ndarray] = None) -> Statevector:
        import time
        t0 = time.time()
        if param_values is None:
            param_values = self._param_values
        bound_circuit = self.param_circuit.assign_parameters(
            {self._param_vector[i]: param_values[i] for i in range(len(param_values))}
        )
        t1 = time.time()
        job = self.backend.run(bound_circuit, shots=None)
        result = job.result()
        t2 = time.time()
        elapsed_prep = t1 - t0
        elapsed_sim = t2 - t1
        total = t2 - t0
        # Log to file to avoid interfering with progress bar
        with open("timing_log.txt", "a") as f:
            f.write(f"{self.t if hasattr(self, 't') else 0},{elapsed_prep:.4f},{elapsed_sim:.4f},{total:.4f}\n")

        if result.success:
            return result.get_statevector()
        else:
            err_msg = result.results[0].error_message if result.results else "Unknown error"
            raise RuntimeError(f"Statevector simulation failed: {err_msg}")

    def _get_marginal_probabilities(self, param_values=None, statevector=None):
        if statevector is None:
            sv = self._get_statevector(param_values)
        else:
            sv = statevector
        probs = np.abs(sv.data) ** 2


        return np.dot(self.bit_masks, probs)

    def _get_combo_probability(self, combo, param_values=None, statevector=None):
        if statevector is None:
            sv = self._get_statevector(param_values)
        else:
            sv = statevector

        probs = np.abs(sv.data) ** 2
        target_bits = 0
        for arm in combo:
            target_bits |= 1 << arm  # compute bit mask for the combo

        # Vectorized mask: select all basis states where the bits corresponding to the combo are all 1
        mask = (self.all_states & target_bits) == target_bits
        return np.sum(probs[mask])

    def _get_all_combo_probabilities(self, param_values: Optional[np.ndarray] = None) -> Dict:
        sv = self._get_statevector(param_values)
        probs = {}
        for combo in self.all_combos:
            target_bits = 0
            for arm in combo:
                target_bits |= 1 << arm
            prob = 0.0
            for i, amp in enumerate(sv.data):
                if (i & target_bits) == target_bits:
                    prob += np.abs(amp) ** 2
            probs[combo] = prob
        return probs

    # SPSA optimizer core (single step update)
    def _calibrate_optimizer(self, loss_fn: Callable, initial_point: np.ndarray):
        """Single calibration to obtain suitable learning rate and perturbation sequences"""
        target_magnitude = 0.02 * np.pi  # desired parameter change magnitude
        learning_rate, perturbation = SPSA.calibrate(
            loss=loss_fn,
            initial_point=initial_point,
            c=self.spsa_c,
            stability_constant=self.spsa_A,
            target_magnitude=target_magnitude,
            alpha=self.spsa_alpha,
            gamma=self.spsa_gamma,
            modelspace=True
        )
        self.optimizer = SPSA(
            maxiter=self.config.N_ROUNDS * 5,
            learning_rate=learning_rate,
            perturbation=perturbation,
            callback=self._optimization_callback
        )
        self.learning_rate = learning_rate
        self.perturbation = perturbation

    def _optimization_callback(self, nfev, x, fx, stepsize, accept):
        self._loss_history.append(fx)
        self._optimization_step += 1

    def _initialize_optimizer(self, loss_fn: Callable):
        if self.optimizer is None:
            self._calibrate_optimizer(loss_fn, self._param_values.copy())

    def _spsa_step(self, loss_fn: Callable, current_params: np.ndarray) -> np.ndarray:
        if self.optimizer is None:
            if not self._optimizer_calibrated:
                self._calibrate_optimizer(loss_fn, current_params.copy())
                self._optimizer_calibrated = True
            # Regardless of calibration, create optimizer instance (using calibrated learning_rate/perturbation)
            self.optimizer = SPSA(
                maxiter=1,
                learning_rate=self.learning_rate,  # set after calibration
                perturbation=self.perturbation,  # set after calibration
                callback=self._optimization_callback
            )
        self.optimizer.maxiter = 1
        result = self.optimizer.minimize(loss_fn, current_params)
        return result.x

    def get_association_matrix(self) -> np.ndarray:
        """Obtain association matrix by summing parameters from all entanglement layers"""
        theta = np.zeros((self.n_arms, self.n_arms))
        for (i, j), idx_list in self._ent_param_map.items():
            # Sum the parameter values (phases) of this entanglement pair across all layers
            total_phase = sum(self._param_values[idx] for idx in idx_list)
            theta[i, j] = total_phase
            theta[j, i] = total_phase   # symmetric matrix
        return theta

    def _compute_fisher_information_diag(self) -> np.ndarray:
        """Compute diagonal of quantum Fisher information matrix using parameter shift rule"""
        n_params = len(self._param_values)
        fisher = np.zeros(n_params)
        eps = np.pi / 2  # parameter shift step

        for i in range(n_params):
            # positive and negative shifts
            p_plus = self._param_values.copy()
            p_plus[i] += eps
            p_minus = self._param_values.copy()
            p_minus[i] -= eps

            # compute corresponding quantum states
            sv_plus = self._get_statevector(p_plus)
            sv_minus = self._get_statevector(p_minus)

            # fidelity F = |<ψ+|ψ->|^2
            fid = np.abs(np.vdot(sv_plus.data, sv_minus.data)) ** 2
            fisher[i] = 8 * (1 - np.sqrt(fid))  # second order approximation

            del sv_plus, sv_minus
            gc.collect()  # force garbage collection

        return fisher

    # Serialization (multi-process safe, adaptive parameter length)
    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ['param_circuit', 'sampling_circuit', '_param_vector',
                    'estimator', 'sampler', 'optimizer', 'simulator']:
            state.pop(key, None)
        state['_rebuild_info'] = {
            'n_arms': self.n_arms,
            'n_qubits': self.n_qubits,
            'combo_size': self.combo_size,
            'depth': self.config.DEPTH,
            'entanglement_structure': self.config.ENTANGLEMENT_STRUCTURE,
            'entanglement_threshold': self.config.ENTANGLEMENT_THRESHOLD,
            'config': self.config,
            'similarity_matrix': self.similarity_matrix,
            '_param_values': self._param_values.copy(),     # parameter values
            '_ent_param_map': self._ent_param_map,  # save parameter index mapping
            '_entangling_pairs': self._entangling_pairs,
            'combo_counts': self.combo_counts,
            'combo_rewards': self.combo_rewards,
            '_optimization_step': self._optimization_step,
            '_loss_history': self._loss_history,
            'spsa_a': self.spsa_a,
            'spsa_c': self.spsa_c,
            'spsa_alpha': self.spsa_alpha,
            'spsa_gamma': self.spsa_gamma,
            'spsa_A': self.spsa_A,
        }
        return state

    def __setstate__(self, state):
        rebuild_info = state.pop('_rebuild_info')
        self.__dict__.update(state)

        # Restore core attributes
        self.n_arms = rebuild_info['n_arms']
        self.n_qubits = rebuild_info['n_qubits']
        self.combo_size = rebuild_info['combo_size']
        self.config = rebuild_info['config']
        self.similarity_matrix = rebuild_info['similarity_matrix']
        self._entangling_pairs = rebuild_info['_entangling_pairs']
        self._ent_param_map = rebuild_info['_ent_param_map']
        self._param_values = rebuild_info['_param_values']
        self.combo_counts = rebuild_info['combo_counts']
        self.combo_rewards = rebuild_info['combo_rewards']
        self._optimization_step = rebuild_info['_optimization_step']
        self._loss_history = rebuild_info['_loss_history']
        self.spsa_a = rebuild_info['spsa_a']
        self.spsa_c = rebuild_info['spsa_c']
        self.spsa_alpha = rebuild_info['spsa_alpha']
        self.spsa_gamma = rebuild_info['spsa_gamma']
        self.spsa_A = rebuild_info['spsa_A']
        self.simulator = AerSimulator(**{'method': 'statevector', 'device': 'GPU'})
        self.estimator = AerEstimator(backend_options={'device': 'GPU'})
        self.sampler = AerSampler(backend_options={'device': 'GPU'})
        self.backend = Aer.get_backend('statevector_simulator')
        try:
            self.backend.set_options(device='GPU')
        except Exception as e:
            print(f"Warning: Failed to set GPU: {e}")
            self.backend.set_options(device='CPU')
        self.optimizer = None

        # Rebuild circuits
        self._build_circuits()

        # Check if parameter length matches; if not, adjust
        if len(self._param_values) != len(self._param_vector):
            warnings.warn(f"Parameter length mismatch: saved values {len(self._param_values)} "
                          f"vs new circuit {len(self._param_vector)}，truncating/padding")
            if len(self._param_values) > len(self._param_vector):
                self._param_values = self._param_values[:len(self._param_vector)]
            else:
                self._param_values = np.pad(self._param_values,
                                            (0, len(self._param_vector) - len(self._param_values)),
                                            'wrap')
        self.estimator = Estimator()
        self.sampler = Sampler()
        self.optimizer = None


# Quantum Thompson Sampling (Q-TS)
class QuantumTS(QuantumComboBanditBase):
    """Quantum Thompson Sampling (diagonal posterior covariance update)"""

    def __init__(self, config: Config, n_arms: int,
                 similarity_matrix: Optional[np.ndarray] = None):
        super().__init__(config, n_arms, similarity_matrix)
        n_params = len(self._param_vector)

        self.prior_mean = np.zeros(n_params)
        self.prior_cov = 0.1 * np.ones(n_params)          # diagonal prior variance
        self.posterior_mean = self._param_values.copy()
        self.posterior_cov = 1.0 / self.prior_cov      # diagonal posterior variance

    # Decision
    def select_combo(self):
        # Sample parameters from posterior
        sampled_params = np.random.normal(self.posterior_mean, np.sqrt(1.0 / self.posterior_cov))

        # 1. Get statevector (only one simulation)
        sv = self._get_statevector(sampled_params)

        # 2. Sample a bitstring from the statevector (according to probability distribution)
        probs_amplitudes = np.abs(sv.data) ** 2  # probabilities of all basis states
        bitstring_int = np.random.choice(len(probs_amplitudes), p=probs_amplitudes)

        # 3. Extract included arms (little‑endian)
        combo = [i for i in range(self.n_arms) if (bitstring_int >> i) & 1]

        # 4. Handle combo size
        if len(combo) == self.combo_size:
            return combo
        elif len(combo) > self.combo_size:
            probs = self._get_marginal_probabilities(statevector=sv)  # reuse sv
            combo.sort(key=lambda x: probs[x], reverse=True)
            return combo[:self.combo_size]
        else:
            probs = self._get_marginal_probabilities(statevector=sv)  # reuse sv
            all_arms = set(range(self.n_arms))
            unused = list(all_arms - set(combo))
            unused.sort(key=lambda x: probs[x], reverse=True)
            needed = self.combo_size - len(combo)
            combo.extend(unused[:needed])
            return combo

    # Loss function and update
    def _loss_fn(self, params: np.ndarray, combo: List[int], reward: float) -> float:
        """Negative log-likelihood (Bernoulli observation model)"""
        prob = self._get_combo_probability(combo, params)
        # prevent log(0)
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        return - (reward * np.log(prob) + (1 - reward) * np.log(1 - prob))

    def update(self, combo: List[int], reward: float):
        combo_key = tuple(sorted(combo))
        self.combo_counts[combo_key] += 1
        self.combo_rewards[combo_key] += reward

        # Bind loss function for current observation
        def loss_fn(params):
            return self._loss_fn(params, combo, reward)

        # Single SPSA step to update parameters
        self._param_values = self._spsa_step(loss_fn, self._param_values)

        # Posterior mean = current MAP estimate
        self.posterior_mean = self._param_values.copy()


# Quantum Combinatorial UCB (Q-CUCB) —— Fisher Information Uncertainty
class QuantumCUCB(QuantumComboBanditBase):

    def __init__(self, config: Config, n_arms: int,
                 similarity_matrix: Optional[np.ndarray] = None):
        super().__init__(config, n_arms, similarity_matrix)
        self.fisher_diag_cache = np.ones(len(self._param_vector))
        self.t = 0

    # Decision
    def select_combo(self):
        self.t += 1
        selected = []
        # Compute Fisher information (diagonal) every `fisher_update_freq` rounds
        if self.t % self.fisher_update_freq == 1:
            self.fisher_diag_cache = self._compute_fisher_information_diag()
        fisher_diag = self.fisher_diag_cache

        sv = self._get_statevector()

        # Mapping: rotation parameter indices for each arm (first layer + extra layers)
        arm_param_indices = [[] for _ in range(self.n_arms)]
        idx = 0
        for layer in range(self.config.DEPTH):
            for i in range(self.n_arms):
                arm_param_indices[i].append(idx)
                idx += 1
            idx += len(self._entangling_pairs)

        selected = []
        # Greedy construction
        for _ in range(self.combo_size):
            best_arm = -1
            best_score = -float('inf')
            for i in range(self.n_arms):
                if i in selected:
                    continue
                # Current candidate combo
                candidate = selected + [i]
                # Combo probability
                prob = self._get_combo_probability(candidate, statevector=sv)
                # Arm uncertainty: sum of Fisher info of all rotation parameters for that arm
                fisher_sum = sum(fisher_diag[idx] for idx in arm_param_indices[i])
                # Combo count (still maintained; if zero, use infinite exploration)
                combo_key = tuple(sorted(candidate))
                count = self.combo_counts.get(combo_key, 0)
                if count == 0:
                    uncertainty = self.config.BETA * np.sqrt(np.log(self.t + 1))
                else:
                    uncertainty = self.config.BETA * np.sqrt(np.log(self.t + 1) / (count * (fisher_sum + 1e-8)))
                score = prob + uncertainty
                if score > best_score:
                    best_score = score
                    best_arm = i
            selected.append(best_arm)
        return selected


    # Loss function and update
    def _loss_fn(self, params: np.ndarray, combo: List[int], reward: float) -> float:
        prob = self._get_combo_probability(combo, params)
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        return - (reward * np.log(prob) + (1 - reward) * np.log(1 - prob))

    def update(self, combo: List[int], reward: float):
        combo_key = tuple(sorted(combo))
        self.combo_counts[combo_key] += 1
        self.combo_rewards[combo_key] += reward

        def loss_fn(params):
            return self._loss_fn(params, combo, reward)

        self._param_values = self._spsa_step(loss_fn, self._param_values)


# ----------------------------------------------------------------------
# Quantum Neural Bandit —— Quantum-Neural Hybrid Architecture
# ----------------------------------------------------------------------
class QuantumNeuralBandit(QuantumComboBanditBase):
    """
    Quantum Neural Bandit (context-aware version)
    - Quantum features: <Z> expectation for each qubit (n_qubits dimensions)
    - Context features: protein fragment fingerprint (context_dim dimensions)
    - Combined features: one‑hot encoding (n_arms dimensions)
    - Input features = quantum features + context features + one‑hot encoding
    - Classical network outputs combo reward prediction (scalar)
    - End-to-end training: quantum parameters updated by SPSA, classical network by Adam
    """

    def __init__(self, config, n_arms, context_dim, similarity_matrix=None):
        super().__init__(config, n_arms, similarity_matrix)
        self.context_dim = context_dim
        # Input: quantum features (n_qubits) + context (context_dim)
        input_dim = self.n_qubits + self.context_dim
        self.classical_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.NEURAL_DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_arms),   # output score for each arm
        ).to(config.DEVICE)
        self.optimizer_cl = torch.optim.Adam(self.classical_net.parameters(), lr=config.LEARNING_RATE)

        # ---------- Experience replay (store: quantum features, context, combo, reward) ----------
        self.qf_history = []      # quantum features
        self.context_history = [] # context features
        self.combo_history = []   # combo
        self.reward_history = []  # reward
        self.max_history = 500
        self.batch_size = 32
        self.train_step = 0
        self.update_freq = 1

    # Quantum feature extraction
    def _get_quantum_features(self, param_values: Optional[np.ndarray] = None, statevector=None) -> np.ndarray:
        if statevector is None:
            sv = self._get_statevector(param_values)
        else:
            sv = statevector
        probs = np.abs(sv.data) ** 2
        marginal = np.dot(self.bit_masks, probs)
        return 2 * marginal - 1  # <Z>

    # Decision: enumerate all combos, predict reward, take maximum
    def select_combo(self, context):
        sv = self._get_statevector()
        qf = self._get_quantum_features(statevector=sv)
        self.qf_cache = qf
        qf_t = torch.tensor(qf, dtype=torch.float32, device=self.config.DEVICE)
        ctx_t = torch.tensor(context, dtype=torch.float32, device=self.config.DEVICE)
        combined = torch.cat([qf_t, ctx_t]).unsqueeze(0)
        with torch.no_grad():
            scores = self.classical_net(combined).squeeze().cpu().numpy()
        # select arms with highest scores
        selected = np.argsort(scores)[-self.combo_size:].tolist()
        return selected

    # Update: store experience + train classical network + SPSA update quantum parameters
    def update(self, combo: List[int], reward: float, context: np.ndarray):
        combo_key = tuple(sorted(combo))
        self.combo_counts[combo_key] += 1
        self.combo_rewards[combo_key] += reward

        # 1. Get quantum features (prefer cache to avoid recomputation)
        qf = getattr(self, 'qf_cache', None)
        if qf is None:
            qf = self._get_quantum_features()
        self.qf_cache = qf

        # 2. Store experience
        self.qf_history.append(qf)
        self.context_history.append(context)
        self.combo_history.append(combo)
        self.reward_history.append(reward)

        # Limit queue length
        if len(self.qf_history) > self.max_history:
            self.qf_history.pop(0)
            self.context_history.pop(0)
            self.combo_history.pop(0)
            self.reward_history.pop(0)

        # 3. Train classical network (every round or periodically)
        self.train_step += 1
        if self.train_step % self.update_freq == 0 and len(self.reward_history) >= self.batch_size:
            self._train_classical_network()

        # 4. Update quantum parameters (SPSA) —— maximize probability of observed combo
        def loss_fn(params):
            prob = self._get_combo_probability(combo, params)
            prob = np.clip(prob, 1e-10, 1 - 1e-10)
            return - (reward * np.log(prob) + (1 - reward) * np.log(1 - prob))

        self._param_values = self._spsa_step(loss_fn, self._param_values)

    # Batch training of classical network
    def _train_classical_network(self):
        indices = np.random.choice(len(self.reward_history), self.batch_size, replace=False)
        # Construct quantum features + context input, target scores for each arm
        batch_qf = [self.qf_history[i] for i in indices]
        batch_ctx = [self.context_history[i] for i in indices]
        batch_combo = [self.combo_history[i] for i in indices]
        batch_reward = [self.reward_history[i] for i in indices]
        qf_t = torch.tensor(np.array(batch_qf), dtype=torch.float32, device=self.config.DEVICE)
        ctx_t = torch.tensor(np.array(batch_ctx), dtype=torch.float32, device=self.config.DEVICE)
        combined = torch.cat([qf_t, ctx_t], dim=1)
        # Target scores: assign reward to arms in combo, 0 to others
        targets = torch.zeros((self.batch_size, self.n_arms), device=self.config.DEVICE)
        for i, (combo, reward) in enumerate(zip(batch_combo, batch_reward)):
            targets[i, combo] = reward
        self.classical_net.train()
        self.optimizer_cl.zero_grad()
        predictions = self.classical_net(combined)
        loss = F.mse_loss(predictions, targets)
        loss.backward()
        self.optimizer_cl.step()

    # Enhanced serialization
    def __getstate__(self):
        state = super().__getstate__()
        state.pop('classical_net', None)
        state.pop('optimizer_cl', None)
        if hasattr(self, 'classical_net'):
            state['classical_net_state_dict'] = self.classical_net.state_dict()
            state['optimizer_cl_state_dict'] = self.optimizer_cl.state_dict()
        # Save context dimension
        state['context_dim'] = self.context_dim
        return state

    def __setstate__(self, state):
        self.context_dim = state.pop('context_dim', 0)
        net_state = state.pop('classical_net_state_dict', None)
        opt_state = state.pop('optimizer_cl_state_dict', None)
        super().__setstate__(state)

        # Rebuild network
        input_dim = self.n_qubits + self.context_dim
        self.classical_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.config.NEURAL_DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.config.NEURAL_DROPOUT),
            nn.Linear(64, self.n_arms)  # output score for each arm, no Sigmoid
        ).to(self.config.DEVICE)
        if net_state is not None:
            self.classical_net.load_state_dict(net_state)

        self.optimizer_cl = torch.optim.Adam(
            self.classical_net.parameters(),
            lr=self.config.LEARNING_RATE
        )
        if opt_state is not None:
            self.optimizer_cl.load_state_dict(opt_state)

        # Clear experience buffers
        self.qf_history = []
        self.context_history = []
        self.combo_history = []
        self.reward_history = []
        self.train_step = 0
