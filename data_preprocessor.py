import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from Bio import SeqIO
import re
from tqdm import tqdm
from config import Config
import warnings
import time
from typing import List, Tuple, Dict, Optional
import mysql.connector
from mysql.connector import Error
try:
    from cuml import KMeans as cuKMeans
    from cuml import PCA as cuPCA
    _USE_CUML = True
    import cupy as cp
    _USE_CUPY = True
except ImportError:
    from sklearn.cluster import KMeans as cuKMeans
    from sklearn.decomposition import PCA as cuPCA
    _USE_CUML = False
    _USE_CUPY = False
    print("cuml not available, using sklearn (CPU)")

# ---------- MySQL ----------
def get_mysql_connection(config):
    """Return a MySQL database connection"""
    try:
        conn = mysql.connector.connect(
            host=config.MYSQL_HOST,
            port=config.MYSQL_PORT,
            user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD,
            database=config.MYSQL_DATABASE,
            charset='utf8',
            use_unicode=True
        )
        return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def fetch_target_activities_from_db(config, target_id):
    """Fetch list of pChEMBL values for a given target from MySQL database"""
    conn = get_mysql_connection(config)
    if conn is None:
        return []
    cursor = conn.cursor()
    query = """
    SELECT act.pchembl_value
    FROM activities act
    JOIN assays ass ON act.assay_id = ass.assay_id
    JOIN target_dictionary td ON ass.tid = td.tid
    WHERE td.chembl_id = %s
      AND act.pchembl_value IS NOT NULL
      AND act.standard_type IN ('IC50', 'Ki', 'EC50', 'Kd', 'Potency')
    """
    try:
        cursor.execute(query, (target_id,))
        results = cursor.fetchall()
    except Error as e:
        print(f"MySQL query error for {target_id}: {e}")
        results = []
    finally:
        cursor.close()
        conn.close()
    values = [float(row[0]) for row in results if row[0] is not None]
    return values

def fetch_target_sequence_from_db(config, target_id):
    """Fetch protein sequence of a target from database (via component_sequences table)"""
    conn = get_mysql_connection(config)
    if conn is None:
        return ''
    cursor = conn.cursor()

    query = """
    SELECT cs.sequence
    FROM target_components tc
    JOIN component_sequences cs ON tc.component_id = cs.component_id
    JOIN target_dictionary td ON tc.tid = td.tid
    WHERE td.chembl_id = %s
    LIMIT 1
    """
    try:
        cursor.execute(query, (target_id,))
        result = cursor.fetchone()
    except Error as e:
        print(f"MySQL query error for {target_id}: {e}")
        result = None
    finally:
        cursor.close()
        conn.close()
    if result:
        return result[0]
    else:
        return ''

# ---------- ChemblDataset class ----------
class ChemblDataset(Dataset):
    def __init__(self, config: Config, use_real_data: bool = True):
        self.config = config
        self.use_real_data = use_real_data
        self.sequences: List[str] = []          # all sequences loaded from FASTA
        self.sequence_features: Optional[np.ndarray] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.advanced_features: Optional[np.ndarray] = None
        self.active_indices: List[int] = []
        self.active_sequences: List[str] = []
        self.active_features: Optional[np.ndarray] = None
        self.active_fingerprints: Optional[np.ndarray] = None
        self._load_data()

    def _filter_sequences(self, sequences: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Filter sequences: keep only those with valid amino acids and length ≥5"""
        valid = []
        amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        for seq_id, seq in sequences:
            seq_clean = re.sub(r'\d+', '', seq).replace(' ', '').replace('\n', '').replace('\r', '')
            if all(aa in amino_acids for aa in seq_clean) and len(seq_clean) >= 5:
                valid.append((seq_id, seq_clean))
        # limit number
        max_seqs = min(len(valid), self.config.N_FRAGMENTS)
        return valid[:max_seqs]

    def _compute_sequence_features(self, sequences: List[str]) -> np.ndarray:
        """Compute 24-dimensional features for sequences (amino acid composition + 4 global properties)"""
        if not sequences:
            return np.empty((0, 24), dtype=np.float32)
        features = []
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        for seq in tqdm(sequences, desc="Computing features"):
            try:
                # Amino acid composition
                aa_counts = {aa: 0 for aa in amino_acids}
                for aa in seq:
                    if aa in aa_counts:
                        aa_counts[aa] += 1
                seq_len = len(seq)
                comp = [aa_counts[aa] / seq_len if seq_len > 0 else 0 for aa in amino_acids]

                # Normalized length
                length_norm = min(seq_len / 200.0, 1.0)

                # Hydrophobicity
                hydrophobic_aas = set('AILMFWV')
                hydrophobic_count = sum(1 for aa in seq if aa in hydrophobic_aas)
                hydrophobicity = hydrophobic_count / seq_len if seq_len > 0 else 0

                # Charge
                positive_aas = set('KRH')
                negative_aas = set('DE')
                positive_count = sum(1 for aa in seq if aa in positive_aas)
                negative_count = sum(1 for aa in seq if aa in negative_aas)
                net_charge = (positive_count - negative_count) / seq_len if seq_len > 0 else 0

                # Aromaticity
                aromatic_aas = set('FHWY')
                aromatic_count = sum(1 for aa in seq if aa in aromatic_aas)
                aromaticity = aromatic_count / seq_len if seq_len > 0 else 0

                feature = comp + [length_norm, hydrophobicity, net_charge, aromaticity]
                features.append(np.array(feature, dtype=np.float32))
            except Exception as e:
                # If error, return zero vector
                print(f"Warning: feature computation failed, using zeros: {e}")
                features.append(np.zeros(24, dtype=np.float32))
        return np.array(features)

    def _compute_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix of feature matrix"""
        n = features.shape[0]
        if n == 0:
            return np.empty((0, 0), dtype=np.float32)

        if _USE_CUPY:
            features_gpu = cp.asarray(features)
            norms = cp.linalg.norm(features_gpu, axis=1, keepdims=True)
            norms[norms == 0] = 1
            sim_gpu = (features_gpu @ features_gpu.T) / (norms @ norms.T)
            sim_gpu = cp.clip(sim_gpu, -1.0, 1.0)
            similarity = cp.asnumpy(sim_gpu)
        else:
            similarity = np.zeros((n, n), dtype=np.float32)
            for i in range(n):
                for j in range(i, n):
                    norm_i = np.linalg.norm(features[i])
                    norm_j = np.linalg.norm(features[j])
                    if norm_i > 0 and norm_j > 0:
                        cos = np.dot(features[i], features[j]) / (norm_i * norm_j)
                        cos = max(-1.0, min(1.0, cos))
                        similarity[i, j] = cos
                        similarity[j, i] = cos
                    else:
                        similarity[i, j] = 0
                        similarity[j, i] = 0
        np.fill_diagonal(similarity, 1.0)
        return similarity

    def _extract_advanced_features(self, features: np.ndarray) -> np.ndarray:
        """Apply PCA to features, concatenate, then standardize. If sample size <2, just standardize."""
        if features.shape[0] == 0:
            return features
        if features.shape[0] < 2:
            # too few samples for PCA, only standardize
            scaler = StandardScaler()
            return scaler.fit_transform(features)

        n_components = min(10, features.shape[1], features.shape[0])
        if _USE_CUML:
            pca = cuPCA(n_components=n_components, random_state=self.config.RANDOM_SEED)
        else:
            pca = PCA(n_components=n_components, random_state=self.config.RANDOM_SEED)

        try:
            pca_features = pca.fit_transform(features)
        except Exception as e:
            print(f"PCA failed, falling back to original features. Error: {e}")
            # If PCA fails, return standardized original features
            scaler = StandardScaler()
            return scaler.fit_transform(features)

        combined = np.hstack([features, pca_features])
        scaler = StandardScaler()
        combined = scaler.fit_transform(combined)
        return combined

    def _load_real_target_data(self):
        """Load real target data from local MySQL database"""
        if hasattr(self, '_real_data_loaded') and self._real_data_loaded:
            print("Real target data already loaded, skipping.")
            return
        self._real_data_loaded = True

        # Base target list
        base_targets = [
            'CHEMBL203', 'CHEMBL1827', 'CHEMBL1907603', 'CHEMBL228', 'CHEMBL1836',
            'CHEMBL1862', 'CHEMBL4282', 'CHEMBL221', 'CHEMBL240', 'CHEMBL325',
            'CHEMBL237', 'CHEMBL218', 'CHEMBL222', 'CHEMBL234', 'CHEMBL255',
            'CHEMBL261', 'CHEMBL271', 'CHEMBL279', 'CHEMBL283', 'CHEMBL296',
            'CHEMBL1947', 'CHEMBL299', 'CHEMBL224', 'CHEMBL256', 'CHEMBL262',
            'CHEMBL214', 'CHEMBL245', 'CHEMBL288', 'CHEMBL310', 'CHEMBL339',
        ]
        needed = self.config.N_ACTIVE_FRAGMENTS
        if needed > len(base_targets):
            repeat = (needed // len(base_targets)) + 1
            default_targets = (base_targets * repeat)[:needed]
        else:
            default_targets = base_targets[:needed]

        print("Fetching real target data from local MySQL database...")
        active_seqs = []
        active_activities = []
        active_ids = []
        real_count = 0

        for tid in default_targets:
            acts = fetch_target_activities_from_db(self.config, tid)
            seq = fetch_target_sequence_from_db(self.config, tid)

            if seq and acts:
                mu = np.mean(acts)
                source = "REAL"
                real_count += 1
                print(f"  [DB] {tid}: REAL (seq_len={len(seq)}, acts={len(acts)})")
            else:
                seq = 'A' * 50  # placeholder
                mu = np.random.uniform(0.3, 0.7)
                source = "PLACEHOLDER"
                print(f"  [DB] {tid}: {source} (seq_len={len(seq) if seq else 0}, acts={len(acts)})")

            active_seqs.append(seq)
            active_activities.append(mu)
            active_ids.append(tid)

        print(f"  [DB] Total real targets: {real_count}/{len(default_targets)}")

        self.active_sequences = active_seqs
        self.active_activities = np.array(active_activities)
        self.active_target_ids = active_ids

        print("Computing features for active sequences...")
        self.sequence_features = self._compute_sequence_features(self.active_sequences)
        self.similarity_matrix = self._compute_similarity_matrix(self.sequence_features)
        self.advanced_features = self._extract_advanced_features(self.sequence_features)
        self.active_features = self.advanced_features
        self.active_fingerprints = self.active_features
        self.active_indices = list(range(len(self.active_sequences)))

        print(f"Loaded {len(self.active_indices)} real target sequences.")

        if len(self.active_sequences) == 0:
            print("No real sequences fetched, falling back to clustering mode.")
            self._load_clustered_data()

    def _load_clustered_data(self):
        """Load data from FASTA and select active sequences via clustering """
        print("Loading FASTA data and clustering...")
        # Read FASTA file
        records = []
        try:
            with open(self.config.DATA_PATH, "r") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    records.append((record.id, str(record.seq)))
        except Exception as e:
            print(f"Error reading FASTA file: {e}")
            raise

        # Filter sequences
        valid = self._filter_sequences(records)
        if not valid:
            raise ValueError("No valid protein sequences found in FASTA!")

        self.all_ids = [item[0] for item in valid]
        self.sequences = [item[1] for item in valid]

        # Compute features for all sequences
        print("Computing features for all sequences...")
        self.sequence_features = self._compute_sequence_features(self.sequences)
        self.similarity_matrix = self._compute_similarity_matrix(self.sequence_features)
        self.advanced_features = self._extract_advanced_features(self.sequence_features)

        # Use K-Means clustering to select active sequences
        n_clusters = self.config.N_ACTIVE_FRAGMENTS
        if n_clusters > len(self.advanced_features):
            n_clusters = len(self.advanced_features)
            print(f"Warning: Reducing n_clusters to {n_clusters}")

        if n_clusters >= len(self.advanced_features):
            self.active_indices = list(range(len(self.advanced_features)))
        else:
            if _USE_CUML:
                kmeans = cuKMeans(n_clusters=n_clusters, random_state=self.config.RANDOM_SEED, n_init=10)
            else:
                kmeans = cuKMeans(n_clusters=n_clusters, random_state=self.config.RANDOM_SEED, n_init=10)
            kmeans.fit(self.advanced_features)
            self.active_indices = []
            for i in range(n_clusters):
                cluster_idx = np.where(kmeans.labels_ == i)[0]
                if len(cluster_idx) > 0:
                    # Find point closest to cluster center
                    center = kmeans.cluster_centers_[i]
                    dist = np.linalg.norm(self.advanced_features[cluster_idx] - center, axis=1)
                    best = cluster_idx[np.argmin(dist)]
                    self.active_indices.append(int(best))
                else:
                    self.active_indices.append(np.random.choice(len(self.sequences)))

        # Deduplicate and pad if necessary
        self.active_indices = list(set(self.active_indices))
        if len(self.active_indices) < n_clusters:
            all_idx = list(range(len(self.sequences)))
            remaining = [i for i in all_idx if i not in self.active_indices]
            needed = n_clusters - len(self.active_indices)
            if remaining and needed > 0:
                additional = np.random.choice(remaining, min(needed, len(remaining)), replace=False)
                self.active_indices.extend(additional)
        self.active_indices = self.active_indices[:n_clusters]

        self.active_sequences = [self.sequences[i] for i in self.active_indices]
        self.active_features = self.advanced_features[self.active_indices]
        self.active_fingerprints = self.active_features
        print(f"Selected {len(self.active_indices)} active sequences via clustering.")

    def _load_data(self):
        """Main loading function"""
        if self.use_real_data:
            print("Loading FASTA sequences for potential matching...")
            # Load sequences from FASTA file
            records = []
            try:
                with open(self.config.DATA_PATH, "r") as handle:
                    for record in SeqIO.parse(handle, "fasta"):
                        records.append((record.id, str(record.seq)))
            except Exception as e:
                print(f"Error reading FASTA file: {e}")
                self.sequences = []
            else:
                valid = self._filter_sequences(records)
                self.sequences = [item[1] for item in valid]
                self.all_ids = [item[0] for item in valid]
                print(f"Loaded {len(self.sequences)} sequences from FASTA.")

            # Attempt to load real target data from database
            self._load_real_target_data()
        else:
            # Traditional clustering mode
            self._load_clustered_data()

    def get_ground_truth_parameters(self):
        """Return ground truth parameters μ and θ"""
        if self.use_real_data and hasattr(self, 'active_activities'):
            μ = self.active_activities.copy()
            # Normalize to [0.1, 0.9]
            if μ.max() > μ.min():
                μ = (μ - μ.min()) / (μ.max() - μ.min()) * 0.8 + 0.1
            else:
                μ = np.ones_like(μ) * 0.5
            # θ based on similarity matrix (already computed)
            θ = self.similarity_matrix.copy()
            # Map similarity to [-0.8, 0.8]
            θ = θ * 0.8
            np.fill_diagonal(θ, 0)
            return μ, θ
        else:
            # Use clustering-selected active sequences, generate random μ and similarity-based θ
            n_active = len(self.active_indices)
            μ = np.zeros(n_active)
            for i, idx in enumerate(self.active_indices):
                seq = self.sequences[idx]
                length = len(seq)
                hydrophobic_aas = 'AILMFWV'
                hydrophobicity = sum(1 for aa in seq if aa in hydrophobic_aas) / length if length > 0 else 0
                μ[i] = 0.3 * (min(length, 200) / 200) + 0.7 * hydrophobicity + np.random.uniform(-0.1, 0.1)
            if μ.max() > μ.min():
                μ = (μ - μ.min()) / (μ.max() - μ.min()) * 0.8 + 0.1
            else:
                μ = np.ones(n_active) * 0.5

            θ = np.zeros((n_active, n_active))
            for i in range(n_active):
                for j in range(i+1, n_active):
                    idx_i = self.active_indices[i]
                    idx_j = self.active_indices[j]
                    sim = self.similarity_matrix[idx_i, idx_j]
                    if sim > 0.6:
                        θ[i, j] = np.random.uniform(0.4, 0.8)
                    elif sim > 0.3:
                        θ[i, j] = np.random.uniform(0.1, 0.4)
                    elif sim < -0.3:
                        θ[i, j] = np.random.uniform(-0.8, -0.2)
                    else:
                        θ[i, j] = np.random.uniform(-0.1, 0.1)
                    θ[j, i] = θ[i, j]
            np.fill_diagonal(θ, 0)
            noise = np.random.randn(n_active, n_active) * 0.05
            noise = (noise + noise.T) / 2
            θ += noise
            np.fill_diagonal(θ, 0)
            return μ, θ

    def visualize_sequences(self):
        import matplotlib.pyplot as plt
        import numpy as np

        μ, θ = self.get_ground_truth_parameters()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # ---------- Left: sequence length distribution ----------
        lengths = [len(self.active_sequences[i]) for i in range(len(self.active_sequences))]
        axes[0].hist(lengths, bins='auto', alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(np.mean(lengths), color='red', linestyle='--', linewidth=1.5,
                        label=f'Mean: {np.mean(lengths):.1f}')
        axes[0].axvline(np.median(lengths), color='orange', linestyle='--', linewidth=1.5,
                        label=f'Median: {np.median(lengths):.1f}')
        axes[0].set_xlabel('Sequence Length (amino acids)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('(a) Sequence Length Distribution', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(axis='y', alpha=0.3)
        # statistics text box
        stats_text = f'n = {len(lengths)}\nμ = {np.mean(lengths):.2f}\nσ = {np.std(lengths):.2f}'
        axes[0].text(0.95, 0.95, stats_text, transform=axes[0].transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # ---------- Middle: base activities μ distribution ----------
        axes[1].hist(μ, bins='auto', alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1].axvline(np.mean(μ), color='red', linestyle='--', linewidth=1.5,
                        label=f'Mean: {np.mean(μ):.3f}')
        axes[1].axvline(np.median(μ), color='orange', linestyle='--', linewidth=1.5,
                        label=f'Median: {np.median(μ):.3f}')
        axes[1].set_xlabel('Base Activity μ', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('(b) Base Activities μ Distribution', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)
        stats_text = f'n = {len(μ)}\nμ = {np.mean(μ):.3f}\nσ = {np.std(μ):.3f}'
        axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # ---------- Right: association matrix θ heatmap ----------
        im = axes[2].imshow(θ, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        axes[2].set_xlabel('Fragment Index', fontsize=12)
        axes[2].set_ylabel('Fragment Index', fontsize=12)
        axes[2].set_title('(c) Association Matrix θ', fontsize=14, fontweight='bold')
        # set ticks (show all indices)
        axes[2].set_xticks(np.arange(θ.shape[1]))
        axes[2].set_yticks(np.arange(θ.shape[0]))
        axes[2].set_xticklabels(np.arange(θ.shape[1]), fontsize=8)
        axes[2].set_yticklabels(np.arange(θ.shape[0]), fontsize=8)

        if θ.shape[0] <= 15:
            for i in range(θ.shape[0]):
                for j in range(θ.shape[1]):
                    if i != j:  # 对角线为0，不显示
                        text = axes[2].text(j, i, f'{θ[i, j]:.2f}',
                                            ha='center', va='center',
                                            color='white' if abs(θ[i, j]) > 0.5 else 'black',
                                            fontsize=6)
        else:
            # For larger matrices, show only values with absolute value above threshold to reduce crowding
            threshold = 0.3
            for i in range(θ.shape[0]):
                for j in range(θ.shape[1]):
                    if i != j and abs(θ[i, j]) > threshold:
                        axes[2].text(j, i, f'{θ[i, j]:.2f}',
                                     ha='center', va='center',
                                     color='white' if abs(θ[i, j]) > 0.5 else 'black',
                                     fontsize=5)
        cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label('Interaction Strength', fontsize=10)

        plt.tight_layout()

        png_path = f"{self.config.OUTPUT_DIR}/sequence_visualization.png"
        pdf_path = f"{self.config.OUTPUT_DIR}/sequence_visualization.pdf"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Visualization saved to {png_path} and {pdf_path}")
        plt.show()
        return fig

    def __len__(self):
        return len(self.sequences) if hasattr(self, 'sequences') else 0

    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'features': self.advanced_features[idx] if self.advanced_features is not None else self.sequence_features[idx],
            'index': idx
        }
