import torch
import warnings


class Config:
    # Experiment parameters
    N_FRAGMENTS = 5000 # # Number of fragments selected from ChEMBL
    N_ACTIVE_FRAGMENTS = 22  # Number of active fragments used in the bandit
    COMBO_SIZE = 5  # Number of fragments selected per round
    N_ROUNDS = 1200  # Number of rounds per experiment
    N_REPEATS = 10  # Number of repeated experiments
    RANDOM_SEED = 42

    # Quantum parameters
    N_QUBITS = 22  # Must be >= N_ACTIVE_FRAGMENTS
    DEPTH = 6  # Quantum circuit depth
    SHOTS = 512  # Number of quantum measurement shots
    ENTANGLEMENT_STRUCTURE = 'full'  # Entanglement topology: 'full' or 'similarity'
    ENTANGLEMENT_THRESHOLD = 0.15     # Similarity threshold

    # Quantum simulation method: 'statevector' 或 'matrix_product_state'
    SIMULATION_METHOD = 'statevector'   # default: statevector
    # Device specification: 'GPU' or 'CPU'
    SIMULATION_DEVICE = 'GPU'             # for MPS, it's recommended to use 'CPU'

    # Learning parameters
    LEARNING_RATE = 0.01
    BETA = 0.5  # UCB exploration coefficient
    TS_PRIOR_ALPHA = 1.0  # Thompson sampling prior parameter
    TS_PRIOR_BETA = 1.0

    # Neural network parameters
    NEURAL_HIDDEN_DIM = 512
    NEURAL_DROPOUT = 0.2

    FISHER_UPDATE_FREQ = 200
    SPSA_C = 0.1

    # ---------- DeepUCB neural algorithm mode ----------
    NEURAL_USE_ENUMERATION = False  # False = arm‑level scoring (fast), True = enumeration (high precision, slower)

    # Device configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4  # number of data loading workers
    N_PARALLEL = 2  # quantum algorithms use single process, classical algorithms can use multiple

    # 路径配置
    DATA_PATH = "./data/chembl_36.fa"
    OUTPUT_DIR = "./results"

    # MySQL
    MYSQL_HOST = 'localhost'
    MYSQL_PORT = 3306
    MYSQL_USER = 'root'
    MYSQL_PASSWORD = ''  # replace with your MySQL password
    MYSQL_DATABASE = 'chembl_36'

    ALGORITHMS = [
        # ---------- Quantum combinatorial algorithms ----------
        'Q-TS',  # Quantum Thompson Sampling
        'Q-CUCB',  # Quantum Combinatorial UCB
        'Q-Neural',  # Quantum-Neural hybrid

        # ---------- Classical combinatorial algorithms ----------
        'CUCB',  # Combinatorial UCB
        'CTS',  # Combinatorial Thompson Sampling
        'CLinUCB',  # Combinatorial Linear UCB
        'C2UCB',  # C2UCB

        # ---------- Combinatorial baselines ----------
        'NeuralTS',  # Neural Thompson Sampling
        'DeepUCB',  # Deep UCB
        #'AttentionBandit',  # Attention-based combinatorial algorithm — computationally heavy
        'EXP3',  # Adversarial combinatorial algorithm
        'Hedge',  # Exponential weighting combinatorial algorithm

        # ---------- Per-arm bandit baselines ----------
        'Classical-TS',  # Per‑arm independent Beta, top‑k selection
        'Classical-UCB',  # Per‑arm independent UCB, top‑k selection
        #'GP-UCB'             # Gaussian process UCB — computationally heavy
    ]

if Config.N_QUBITS < Config.N_ACTIVE_FRAGMENTS:
    warnings.warn(
        f"N_QUBITS ({Config.N_QUBITS}) < N_ACTIVE_FRAGMENTS ({Config.N_ACTIVE_FRAGMENTS}). "
        f"Automatically set N_QUBITS = N_ACTIVE_FRAGMENTS"
    )
    Config.N_QUBITS = Config.N_ACTIVE_FRAGMENTS