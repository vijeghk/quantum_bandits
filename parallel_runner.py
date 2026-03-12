import warnings
from multiprocessing import Pool

import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use('Agg')  # use non-interactive backend to avoid thread issues

warnings.filterwarnings('ignore')

from data_preprocessor import ChemblDataset
from environment import CombinatorialBanditEnv
from classical_algorithms import (
    CombinatorialUCB, CombinatorialThompsonSampling, CombinatorialLinUCB,
    ClassicalTS, ClassicalUCB, C2UCB, EXP3, Hedge, GPUCB, UCBS, GLMUCB
)
from quantum_algorithms import QuantumTS, QuantumCUCB, QuantumNeuralBandit
from neural_algorithms import NeuralTS, DeepUCB, AttentionBandit, NeuralUCB
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
import threading
import queue

import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

class RichProgressManager:
    def __init__(self):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("• {task.fields[info]}"),
        )
        self.task_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._process_queue)
        self._thread.start()

    def _process_queue(self):
        while not self._stop_event.is_set():
            try:
                task_id, advance, info, description = self.task_queue.get(timeout=0.1)
                if description is not None:
                    self.progress.update(task_id, description=description)
                if advance:
                    self.progress.update(task_id, advance=advance, info=info)
            except queue.Empty:
                continue

    def add_task(self, description, total, **fields):
        task_id = self.progress.add_task(description, total=total, info="", **fields)
        return task_id

    def update_task(self, task_id, advance=0, info="", description=None):
        """Update a task; can simultaneously update description and progress."""
        self.task_queue.put((task_id, advance, info, description))

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        self.progress.stop()


class ParallelExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.dataset = ChemblDataset(config)
        self.true_μ, self.true_θ = self.dataset.get_ground_truth_parameters()

        # Extract similarity submatrix for active fragments
        if self.dataset.similarity_matrix is not None:
            active_idx = self.dataset.active_indices
            self.active_similarity = self.dataset.similarity_matrix[np.ix_(active_idx, active_idx)]
        else:
            self.active_similarity = None
        self.results = []

    def run_single_experiment(self, algorithm_name, seed, progress_position=None):
        import gc
        import time
        gc.disable()

        np.random.seed(seed)
        env = CombinatorialBanditEnv(self.config, self.true_μ, self.true_θ)
        n_arms = len(self.true_μ)

        # Get actual feature dimension
        context_dim = self.dataset.active_features.shape[1]

        # Algorithm initialization
        if algorithm_name == 'Q-TS':
            algo = QuantumTS(self.config, n_arms, self.active_similarity)
        elif algorithm_name == 'Q-CUCB':
            algo = QuantumCUCB(self.config, n_arms, self.active_similarity)
        elif algorithm_name == 'Q-Neural':
            algo = QuantumNeuralBandit(self.config, n_arms, context_dim, self.active_similarity)

        elif algorithm_name == 'CUCB':
            algo = CombinatorialUCB(self.config, n_arms)
        elif algorithm_name == 'CTS':
            algo = CombinatorialThompsonSampling(self.config, n_arms)
        elif algorithm_name == 'CLinUCB':
            algo = CombinatorialLinUCB(self.config, n_arms, context_dim)
        elif algorithm_name == 'C2UCB':
            algo = C2UCB(self.config, n_arms)

        elif algorithm_name == 'Classical-TS':
            algo = ClassicalTS(self.config, n_arms)
        elif algorithm_name == 'Classical-UCB':
            algo = ClassicalUCB(self.config, n_arms)
        elif algorithm_name == 'NeuralTS':
            algo = NeuralTS(self.config, n_arms, context_dim)
        elif algorithm_name == 'DeepUCB':
            algo = DeepUCB(self.config, n_arms, context_dim)
        elif algorithm_name == 'AttentionBandit':
            algo = AttentionBandit(self.config, n_arms, context_dim)
        elif algorithm_name == 'GP-UCB':
            algo = GPUCB(self.config, n_arms)
        elif algorithm_name == 'EXP3':
            algo = EXP3(self.config, n_arms)
        elif algorithm_name == 'Hedge':
            algo = Hedge(self.config, n_arms)

        elif algorithm_name == 'UCB-S':
            # Build reward_funcs and theta_space
            # Assume true_μ is a linear function of theta: mu = true_μ * theta
            theta_space = np.linspace(0, 1, 100)
            # Note: using closure captures k, but lambda's late binding would make all functions use the same k,
            # so use default argument trick.
            reward_funcs = [lambda theta, k=k: self.true_μ[k] * theta for k in range(n_arms)]
            algo = UCBS(self.config, n_arms, reward_funcs, theta_space)

        elif algorithm_name == 'GLM-UCB':
            # Need arm feature vectors; use active_features
            arm_features = self.dataset.active_features  # shape (n_arms, feature_dim)
            algo = GLMUCB(self.config, n_arms, context_dim=arm_features.shape[1], arm_features=arm_features)

        elif algorithm_name == 'NeuralUCB':
            algo = NeuralUCB(self.config, n_arms, context_dim)

        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        # Experiment loop
        cumulative_regret = []
        cumulative_reward = []
        per_step_rewards = []
        optimal_found = False
        optimal_round = None

        rounds_iter = range(self.config.N_ROUNDS)
        if progress_position is not None:
            rounds_iter = tqdm(
                rounds_iter,
                desc=f"    {algorithm_name} seed={seed}",
                position=progress_position,
                leave=False
            )

        # Determine if context is needed
        needs_context = algorithm_name in ['CLinUCB', 'NeuralTS', 'DeepUCB', 'AttentionBandit',
                                               'Q-Neural', 'GLM-UCB', 'NeuralUCB']

        log_file = open(f"{algorithm_name}_seed{seed}_timing.txt", "w")
        log_file.write("round,select_time,step_time,update_time\n")  # 表头
        print(f"[INFO] Optimal combo: {env.optimal_combo}, reward: {env.optimal_reward:.4f}")

        for t in range(self.config.N_ROUNDS):
            t_start = time.time()

            if needs_context:
                context = self.dataset.active_fingerprints[t % len(self.dataset.active_fingerprints)]
                combo = algo.select_combo(context)
            else:
                combo = algo.select_combo()

            t_select = time.time()

            reward, is_optimal = env.step(combo)
            per_step_rewards.append(reward)
            t_step = time.time()

            if needs_context:
                algo.update(combo, reward, context)
            else:
                algo.update(combo, reward)

            t_update = time.time()

            regret = env.get_regret(combo)
            cumulative_regret.append(regret + (cumulative_regret[-1] if cumulative_regret else 0))
            cumulative_reward.append(reward + (cumulative_reward[-1] if cumulative_reward else 0))

            if is_optimal and not optimal_found:
                optimal_found = True
                optimal_round = t

            if t % 50 == 0:
                print(f"Round {t:4d}: selected {combo}, optimal {env.optimal_combo}, match = {is_optimal}")
            log_file.write(f"{t},{t_select - t_start:.4f},{t_step - t_select:.4f},{t_update - t_step:.4f}\n")
            if t % 100 == 0:
                log_file.flush()

            if progress_position is not None and t % 50 == 0 and cumulative_regret:
                rounds_iter.set_postfix(regret=f"{cumulative_regret[-1]:.2f}")

        log_file.close()
        gc.enable()

        final_regret = cumulative_regret[-1] if cumulative_regret else 0
        avg_reward = np.mean(per_step_rewards[-100:]) if len(per_step_rewards) >= 100 else np.mean(per_step_rewards)

        # # Association matrix recovery error
        learned_theta = None
        param_error = 0.0
        if hasattr(algo, 'get_association_matrix'):
            try:
                learned_theta = algo.get_association_matrix()
                if learned_theta.shape[0] != self.true_θ.shape[0]:
                    min_dim = min(learned_theta.shape[0], self.true_θ.shape[0])
                    learned = learned_theta[:min_dim, :min_dim]
                    true = self.true_θ[:min_dim, :min_dim]
                else:
                    learned = learned_theta
                    true = self.true_θ
                # Compute Pearson correlation
                flat_true = true.flatten()
                flat_learned = learned.flatten()
                # Handle possible NaN or constant matrices
                if np.std(flat_true) == 0 or np.std(flat_learned) == 0:
                    corr = 0.0
                else:
                    corr = np.corrcoef(flat_true, flat_learned)[0, 1]
                param_error = corr  # directly used as error metric
            except Exception as e:
                print(f"Warning: param error extraction failed for {algorithm_name}: {e}")
                param_error = 0.0
                learned_theta = None


        return {
            'algorithm': algorithm_name,
            'seed': seed,
            'cumulative_regret': final_regret,
            'average_reward': avg_reward,
            'optimal_round': optimal_round,
            'param_error': param_error,
            'regret_curve': cumulative_regret,
            'reward_curve': cumulative_reward,
            'learned_theta': learned_theta
        }

    def _run_single_wrapper(self, algorithm_name, seed, pos = None):
        return self.run_single_experiment(algorithm_name, seed, pos)

    def run_parallel_experiments(self):
        manager = RichProgressManager()
        manager.progress.start()

        total_algorithms = len(self.config.ALGORITHMS)
        # Add top-level algorithm task
        algo_task = manager.add_task("[cyan]Algorithms", total=total_algorithms)

        all_results = []
        for algo_idx, algorithm_name in enumerate(self.config.ALGORITHMS):
            # Update top-level task description with current algorithm and progress
            current = algo_idx + 1
            manager.update_task(
                algo_task,
                description=f"[bold cyan]Algorithm: {algorithm_name} ({current}/{total_algorithms})[/]"
            )

            seeds = range(self.config.RANDOM_SEED, self.config.RANDOM_SEED + self.config.N_REPEATS)
            # Add sub-task for seeds of this algorithm
            seed_task = manager.add_task(
                f"  [green]{algorithm_name} seeds",
                total=len(seeds)
            )

            with Pool(processes=self.config.N_PARALLEL) as pool:
                results = []
                for seed in seeds:
                    # Execute asynchronously, update sub-task via callback
                    result = pool.apply_async(
                        self.run_single_experiment,
                        args=(algorithm_name, seed, None),
                        callback=lambda _: manager.update_task(
                            seed_task,
                            advance=1,
                            info=f"seed {seed} completed"
                        )
                    )
                    results.append(result)
                # Wait for all seeds to complete
                for r in results:
                    r.wait()

            # After algorithm finishes, update top-level progress
            manager.update_task(algo_task, advance=1)
            # Collect results
            all_results.extend([r.get() for r in results])

        manager.stop()
        self.results = all_results
        self._save_results()
        return all_results

    def _save_results(self):
        import os, pickle
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        # Summary table
        summary_path = f"{self.config.OUTPUT_DIR}/experiment_summary.csv"
        for r in self.results:
            row = {
                'algorithm': r['algorithm'],
                'seed': r['seed'],
                'cumulative_regret': r['cumulative_regret'],
                'average_reward': r['average_reward'],
                'optimal_round': r['optimal_round'],
                'param_error': r['param_error']
            }
            df_row = pd.DataFrame([row])
            if not os.path.exists(summary_path):
                df_row.to_csv(summary_path, index=False)
            else:
                df_row.to_csv(summary_path, mode='a', header=False, index=False)

        for r in self.results:
            curve_df = pd.DataFrame({
                'regret': r['regret_curve'],
                'reward': r['reward_curve']
            })
            curve_df.to_csv(f"{self.config.OUTPUT_DIR}/curves_{r['algorithm']}_seed{r['seed']}.csv", index=False)

        theta_dict = {}
        for r in self.results:
            if r['learned_theta'] is not None:
                key = f"{r['algorithm']}_seed{r['seed']}"
                theta_dict[key] = r['learned_theta']
        if theta_dict:
            with open(f"{self.config.OUTPUT_DIR}/learned_thetas.pkl", 'wb') as f:
                pickle.dump(theta_dict, f)
            print(f"Saved {len(theta_dict)} learned association matrices")

    def get_summary_statistics(self):
        if not self.results:
            return None
        df = pd.DataFrame(self.results)
        summary = df.groupby('algorithm').agg(
            Mean_Regret=('cumulative_regret', 'mean'),
            Std_Regret=('cumulative_regret', 'std'),
            Mean_Reward=('average_reward', 'mean'),
            Std_Reward=('average_reward', 'std'),
            Optimal_Found_Ratio=('optimal_round', lambda x: x.notna().mean() * 100),
            Mean_Optimal_Round=('optimal_round', lambda x: x.dropna().mean() if x.dropna().size>0 else np.nan),
            Mean_Param_Error=('param_error', 'mean')
        ).reset_index()
        summary = summary.sort_values('Mean_Regret')
        summary.columns = ['Algorithm', 'Mean Regret', 'Std Regret', 'Mean Reward',
                           'Std Reward', 'Optimal Found %', 'Mean Optimal Round', 'Mean Param Error']
        return summary
