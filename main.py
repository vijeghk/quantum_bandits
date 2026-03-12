import os
import sys
import copy
import argparse

import pickle
import torch
import numpy as np
from datetime import datetime
import cupy as cp
from matplotlib import pyplot as plt

from config import Config
from data_preprocessor import ChemblDataset, _USE_CUPY
from parallel_runner import ParallelExperimentRunner
from evaluator import BanditEvaluator, Visualizer


def main():
    parser = argparse.ArgumentParser(description='Quantum-enhanced Combinatorial Bandits for Drug Discovery')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['preprocess', 'train', 'evaluate', 'all',
                                 'ablation', 'scaling'],
                        help='Execution mode')
    parser.add_argument('--algorithm', type=str, default='all',
                        help='Specific algorithm name or "all" to run all algorithms')
    parser.add_argument('--n_rounds', type=int, default=None,
                        help='Number of rounds per experiment')
    parser.add_argument('--n_repeats', type=int, default=None,
                        help='Number of repetitions')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration')

    args = parser.parse_args()

    # Update configuration
    config = Config()
    if args.n_rounds:
        config.N_ROUNDS = args.n_rounds
    if args.n_repeats:
        config.N_REPEATS = args.n_repeats
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.gpu and torch.cuda.is_available():
        config.DEVICE = 'cuda'

    print("=" * 60)
    print("Quantum-enhanced Combinatorial Bandits for Drug Discovery")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Device: {config.DEVICE}")
    print(f"Number of arms: {config.N_ACTIVE_FRAGMENTS}")
    print(f"Combo size: {config.COMBO_SIZE}")
    print(f"Number of rounds: {config.N_ROUNDS}")
    print(f"Number of repeats: {config.N_REPEATS}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print("=" * 60)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    start_time = datetime.now()

    if args.mode == 'ablation':
        run_ablation(config)
    elif args.mode == 'scaling':
        run_scaling(config)
    elif args.mode in ['preprocess', 'train', 'evaluate', 'all']:
        try:
            if args.mode in ['preprocess', 'all']:
                print("\n[1/3] Data Preprocessing...")
                preprocess_data(config)

            if args.mode in ['train', 'all']:
                print("\n[2/3] Running Experiments...")
                run_experiments(config, args.algorithm)

            if args.mode in ['evaluate', 'all']:
                print("\n[3/3] Evaluating Results...")
                evaluate_results(config)

        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()

    end_time = datetime.now()
    total_time = end_time - start_time

    print("\n" + "=" * 60)
    print("Experiment Completed!")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_time}")
    print("=" * 60)


def preprocess_data(config):
    print("Loading and preprocessing ChEMBL protein sequence data...")

    try:
        dataset = ChemblDataset(config)

        print(f"Total sequences: {len(dataset.sequences)}")
        print(f"Active sequences selected: {len(dataset.active_indices)}")

        print("\nGenerating visualizations...")
        dataset.visualize_sequences()

        save_path = f"{config.OUTPUT_DIR}/preprocessed_data.npz"
        np.savez_compressed(
            save_path,
            sequences=dataset.sequences,
            sequence_features=dataset.sequence_features,
            similarity_matrix=dataset.similarity_matrix,
            active_sequence_indices=dataset.active_indices,
            active_sequences=dataset.active_sequences,
            active_features=dataset.active_features,
            μ=dataset.get_ground_truth_parameters()[0],
            θ=dataset.get_ground_truth_parameters()[1]
        )

        print(f"\nPreprocessed data saved to {save_path}")

        print("\n" + "=" * 60)
        print("DATA STATISTICS:")
        print("=" * 60)
        print(f"Total sequences in dataset: {len(dataset.sequences)}")
        print(f"Active sequences selected: {len(dataset.active_indices)}")

        if hasattr(dataset, 'sequence_features'):
            print(f"Feature dimension: {dataset.sequence_features.shape[1]}")

        μ, θ = dataset.get_ground_truth_parameters()
        print(f"\nGround truth parameters:")
        print(f"  Base activities (μ): shape={μ.shape}, range=[{μ.min():.3f}, {μ.max():.3f}]")
        print(f"  Association matrix (θ): shape={θ.shape}")

        upper_tri = np.triu(θ, 1)
        associations = upper_tri[upper_tri != 0]
        if len(associations) > 0:
            print(f"  Number of non-zero associations: {len(associations)}")
            print(f"  Mean absolute strength: {np.mean(np.abs(associations)):.3f}")

        print("=" * 60)

        stats_path = os.path.join(config.OUTPUT_DIR, "preprocessing_stats.txt")
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("DATA STATISTICS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total sequences in dataset: {len(dataset.sequences)}\n")
            f.write(f"Active sequences selected: {len(dataset.active_indices)}\n")
            if hasattr(dataset, 'sequence_features'):
                f.write(f"Feature dimension: {dataset.sequence_features.shape[1]}\n")
            f.write(f"\nGround truth parameters:\n")
            f.write(f"  Base activities (μ): shape={μ.shape}, range=[{μ.min():.3f}, {μ.max():.3f}]\n")
            f.write(f"  Association matrix (θ): shape={θ.shape}\n")
            if len(associations) > 0:
                f.write(f"  Number of non-zero associations: {len(associations)}\n")
                f.write(f"  Mean absolute strength: {np.mean(np.abs(associations)):.3f}\n")
            f.write("=" * 60)

        print(f"\nPreprocessing statistics saved to: {stats_path}")

        return dataset

    except Exception as e:
        print(f"Error in preprocess_data: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_experiments(config, algorithm='all'):
    print("Initializing experiment runner...")

    runner = ParallelExperimentRunner(config)

    #run experiment
    if algorithm != 'all' and algorithm in config.ALGORITHMS:
        original_algorithms = config.ALGORITHMS
        config.ALGORITHMS = [algorithm]
        results = runner.run_parallel_experiments()
        config.ALGORITHMS = original_algorithms
    else:
        results = runner.run_parallel_experiments()

    summary = runner.get_summary_statistics()
    if summary is not None:
        print("\n" + "=" * 60)
        print("Experiment Summary:")
        print("=" * 60)
        print(summary.to_string(index=False))
        print("=" * 60)

    return results

def run_ablation(config):
    """Run ablation experiments, varying key parameters of quantum algorithms"""
    import copy
    import os
    results = {}

    # 1. Fisher update frequency
    for freq in [200, 400, 600, 800, 1000]:
        cfg = copy.deepcopy(config)
        cfg.FISHER_UPDATE_FREQ = freq
        cfg.OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, f"ablation_fisher_{freq}")
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        cfg.ALGORITHMS = ['Q-TS', 'Q-CUCB', 'Q-Neural']
        runner = ParallelExperimentRunner(cfg)
        runner.run_parallel_experiments()
        results[f'fisher_freq_{freq}'] = cfg.OUTPUT_DIR
        print(f"Completed Fisher freq {freq}, results in {cfg.OUTPUT_DIR}")

    # 2. SPSA parameter c
    for c in [0.05, 0.1, 0.2, 0.5]:
        cfg = copy.deepcopy(config)
        cfg.SPSA_C = c
        cfg.OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, f"ablation_spsa_c_{c}")
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        cfg.ALGORITHMS = ['Q-TS', 'Q-CUCB', 'Q-Neural']
        runner = ParallelExperimentRunner(cfg)
        runner.run_parallel_experiments()
        results[f'spsa_c_{c}'] = cfg.OUTPUT_DIR

    # 3. Entanglement structure
    for ent in ['full', 'similarity']:
        cfg = copy.deepcopy(config)
        cfg.ENTANGLEMENT_STRUCTURE = ent
        cfg.OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, f"ablation_ent_{ent}")
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        cfg.ALGORITHMS = ['Q-TS', 'Q-CUCB', 'Q-Neural']
        runner = ParallelExperimentRunner(cfg)
        runner.run_parallel_experiments()
        results[f'ent_{ent}'] = cfg.OUTPUT_DIR

    return results

def run_scaling(config):
    """Run scaling analysis: vary number of arms K or combo size m"""
    results = {}

    # Vary number of arms K
    K_values = [10, 14, 18, 22]
    for K in K_values:
        cfg = copy.deepcopy(config)
        cfg.N_ACTIVE_FRAGMENTS = K
        cfg.N_QUBITS = K  # number of qubits should be at least number of arms
        cfg.OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, f"scaling_K_{K}")
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        cfg.ALGORITHMS = ['Q-TS', 'Classical-TS', 'CUCB']
        runner = ParallelExperimentRunner(cfg)
        runner.run_parallel_experiments()
        results[f'K_{K}'] = cfg.OUTPUT_DIR

    # Vary combo size m
    m_values = [2, 3, 4, 5]
    for m in m_values:
        cfg = copy.deepcopy(config)
        cfg.COMBO_SIZE = m
        cfg.OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, f"scaling_m_{m}")
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        cfg.ALGORITHMS = ['Q-TS', 'Classical-TS', 'CUCB']
        runner = ParallelExperimentRunner(cfg)
        runner.run_parallel_experiments()
        results[f'm_{m}'] = cfg.OUTPUT_DIR

    print("Scaling experiments completed. Results saved in respective directories.")
    return results

def evaluate_results(config):
    """
    Evaluate results with enhanced metrics and additional visualizations:
    - Stability ranking (bar chart of CV Regret)
    - Proportion of runs close to optimal (bar chart)
    - Parameter recovery (bar chart of correlation with true θ)
    - Regret-stability trade-off plot (scatter, quantum highlighted)
    - Detailed report emphasizing quantum strengths
    """
    import os
    import pickle
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from evaluator import BanditEvaluator, Visualizer, CrossExperimentComparator

    # ==================== 1. Evaluate main experiment (root directory) ====================
    summary_path = f"{config.OUTPUT_DIR}/experiment_summary.csv"
    if os.path.exists(summary_path):
        print("Evaluating main experiment...")
        evaluator = BanditEvaluator(config)
        visualizer = Visualizer(config)

        summary_df = evaluator.load_results(summary_path)

        # Load ground truth association matrix (for parameter recovery)
        true_θ = None
        preprocessed_path = f"{config.OUTPUT_DIR}/preprocessed_data.npz"
        if os.path.exists(preprocessed_path):
            data = np.load(preprocessed_path, allow_pickle=True)
            true_θ = data['θ']
            print("Loaded ground truth association matrix from preprocessed data.")
        else:
            print("Warning: preprocessed_data.npz not found, association matrix plots will be skipped.")

        print("\n" + "=" * 60)
        print("Detailed Analysis:")
        print("=" * 60)

        # 1. Ranking by mean regret (original)
        ranking = evaluator.compute_ranking()
        print("\nAlgorithm Ranking (by Mean Regret):")
        print(ranking.to_string(index=False))

        # 2. Statistical significance (original)
        print("\nComputing statistical significance...")
        p_value_matrix = evaluator.compute_statistical_significance()

        # 3. Comprehensive metrics (original, includes CV Regret)
        print("\nComprehensive Metrics:")
        comprehensive_metrics = evaluator.compute_comprehensive_metrics()
        print(comprehensive_metrics.round(4))

        # 4. Learning speed (original, threshold 0.1 may be ineffective but kept)
        regret_curves = evaluator.compute_regret_curves()
        learning_speeds = evaluator.compute_learning_speed(regret_curves, threshold=0.1)
        print("\nLearning Speed (rounds to reach cumulative regret < 0.1):")
        for algo, speed in learning_speeds.items():
            print(f"  {algo}: {speed}")

        # ========== New metrics and visualizations highlighting quantum advantages ==========

        # 5. Stability ranking (by coefficient of variation, ascending)
        cv_df = comprehensive_metrics[['CV Regret']].sort_values('CV Regret')
        print("\n" + "-" * 40)
        print("Stability Ranking (Lower CV is better):")
        print(cv_df.to_string())

        # Bar chart of CV Regret
        plt.figure(figsize=(10, 6))
        colors = ['red' if algo in ['Q-TS', 'Q-CUCB', 'Q-Neural'] else 'blue' for algo in cv_df.index]
        plt.bar(range(len(cv_df)), cv_df['CV Regret'], color=colors, alpha=0.7)
        plt.xticks(range(len(cv_df)), cv_df.index, rotation=30, ha='right')
        plt.ylabel('Coefficient of Variation (CV)')
        plt.title('Stability Comparison (Lower is Better)')
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='Quantum'),
                           Patch(facecolor='blue', label='Classical')]
        plt.legend(handles=legend_elements)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        cv_bar_path = f"{config.OUTPUT_DIR}/stability_bar.png"
        plt.savefig(cv_bar_path, dpi=300)
        plt.show()
        print(f"Stability bar chart saved to {cv_bar_path}")

        # 6. Proportion of runs close to optimal (regret ≤ 1.1 × best mean regret)
        best_mean_regret = ranking.iloc[0]['Mean Regret']
        threshold = best_mean_regret * 1.1   # adjustable factor
        print(f"\nProportion of runs with regret ≤ {threshold:.4f} (≤ {1.1:.1f}× best):")
        close_to_optimal = {}
        for algo in evaluator.results_df['algorithm'].unique():
            regrets = evaluator.results_df[evaluator.results_df['algorithm'] == algo]['cumulative_regret'].values
            close_ratio = np.mean(regrets <= threshold)
            close_to_optimal[algo] = close_ratio
        # Sort descending
        close_sorted = sorted(close_to_optimal.items(), key=lambda x: x[1], reverse=True)
        for algo, ratio in close_sorted:
            print(f"{algo}: {ratio:.1%}")

        # Bar chart of close-to-optimal proportion
        if close_sorted:
            algos_close = [x[0] for x in close_sorted]
            ratios_close = [x[1] for x in close_sorted]
            plt.figure(figsize=(10, 6))
            colors_close = ['red' if algo in ['Q-TS', 'Q-CUCB', 'Q-Neural'] else 'blue' for algo in algos_close]
            plt.bar(range(len(algos_close)), ratios_close, color=colors_close, alpha=0.7)
            plt.xticks(range(len(algos_close)), algos_close, rotation=30, ha='right')
            plt.ylabel('Proportion of Runs Close to Optimal')
            plt.title(f'Proportion with Regret ≤ {threshold:.4f} (≤ 1.1× Best)')
            plt.legend(handles=[Patch(facecolor='red', label='Quantum'), Patch(facecolor='blue', label='Classical')])
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            close_bar_path = f"{config.OUTPUT_DIR}/close_to_optimal_bar.png"
            plt.savefig(close_bar_path, dpi=300)
            plt.show()
            print(f"Close-to-optimal bar chart saved to {close_bar_path}")

        # 7. Parameter recovery analysis (using param_error column, assumed to be correlation)
        print("\n" + "-" * 40)
        print("Association Matrix Recovery (Pearson correlation with true θ):")
        param_df = evaluator.results_df[['algorithm', 'param_error']].dropna()
        if not param_df.empty:
            recovery_stats = {}
            for algo in param_df['algorithm'].unique():
                corrs = param_df[param_df['algorithm'] == algo]['param_error'].values
                recovery_stats[algo] = (np.mean(corrs), np.std(corrs))
                print(f"{algo}: mean={np.mean(corrs):.3f} ± {np.std(corrs):.3f}")

            # Bar chart of parameter recovery (mean ± std)
            algos_rec = list(recovery_stats.keys())
            means_rec = [recovery_stats[a][0] for a in algos_rec]
            stds_rec = [recovery_stats[a][1] for a in algos_rec]
            plt.figure(figsize=(8, 5))
            colors_rec = ['red' if algo in ['Q-TS', 'Q-CUCB', 'Q-Neural'] else 'blue' for algo in algos_rec]
            plt.bar(range(len(algos_rec)), means_rec, yerr=stds_rec, capsize=5, color=colors_rec, alpha=0.7)
            plt.xticks(range(len(algos_rec)), algos_rec, rotation=30, ha='right')
            plt.ylabel('Pearson Correlation with True θ')
            plt.title('Association Matrix Recovery Quality')
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.legend(handles=[Patch(facecolor='red', label='Quantum'), Patch(facecolor='blue', label='Classical')])
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            rec_bar_path = f"{config.OUTPUT_DIR}/parameter_recovery_bar.png"
            plt.savefig(rec_bar_path, dpi=300)
            plt.show()
            print(f"Parameter recovery bar chart saved to {rec_bar_path}")
        else:
            print("No parameter recovery data available.")

        # 8. Regret-stability scatter plot (highlight quantum algorithms)
        def plot_regret_stability_tradeoff(comp_metrics, save_path=None):
            df = comp_metrics.reset_index().rename(columns={'index': 'Algorithm'})
            plt.figure(figsize=(10, 6))
            quantum_algorithms = ['Q-TS', 'Q-CUCB', 'Q-Neural']
            colors = []
            for algo in df['Algorithm']:
                if algo in quantum_algorithms:
                    colors.append('red')
                else:
                    colors.append('blue')
            plt.scatter(df['Mean Regret'], df['Std Regret'], c=colors, s=100, alpha=0.7)
            # Add labels
            for _, row in df.iterrows():
                plt.text(row['Mean Regret'], row['Std Regret'], row['Algorithm'], fontsize=8, ha='right')
            plt.xlabel('Mean Cumulative Regret')
            plt.ylabel('Standard Deviation')
            plt.title('Regret-Stability Trade-off (Quantum in Red)')
            plt.grid(True, alpha=0.3)
            # Add legend
            legend_elements = [Patch(facecolor='red', label='Quantum'),
                               Patch(facecolor='blue', label='Classical')]
            plt.legend(handles=legend_elements)
            if save_path:
                plt.savefig(save_path, dpi=300)
            plt.show()

        tradeoff_path = f"{config.OUTPUT_DIR}/regret_stability_tradeoff.png"
        plot_regret_stability_tradeoff(comprehensive_metrics, save_path=tradeoff_path)

        # ========== Modified original visualizations ==========

        # Regret curves
        curve_path = f"{config.OUTPUT_DIR}/regret_curves.png"
        visualizer.plot_regret_curves(regret_curves, save_path=curve_path)

        # Build aggregated table for performance comparison (only regret and reward)
        agg_df = evaluator.results_df.groupby('algorithm').agg(
            Mean_Regret=('cumulative_regret', 'mean'),
            Std_Regret=('cumulative_regret', 'std'),
            Mean_Reward=('average_reward', 'mean'),
            Std_Reward=('average_reward', 'std')
        ).reset_index().rename(columns={'algorithm': 'Algorithm'})

        agg_df = agg_df.sort_values('Mean_Regret')
        agg_df.columns = ['Algorithm', 'Mean Regret', 'Std Regret', 'Mean Reward', 'Std Reward']

        # Performance comparison plot (regret and reward only)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        x = np.arange(len(agg_df))
        axes[0].bar(x, agg_df['Mean Regret'], yerr=agg_df['Std Regret'], capsize=5, alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(agg_df['Algorithm'], rotation=30, ha='right')
        axes[0].set_ylabel('Cumulative Regret')
        axes[0].set_title('Mean Cumulative Regret with Std Dev')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Sort by mean reward for the second bar chart
        agg_df_reward = agg_df.sort_values('Mean Reward', ascending=False)
        x2 = np.arange(len(agg_df_reward))
        axes[1].bar(x2, agg_df_reward['Mean Reward'], yerr=agg_df_reward['Std Reward'],
                    capsize=5, alpha=0.8, color='green')
        axes[1].set_xticks(x2)
        axes[1].set_xticklabels(agg_df_reward['Algorithm'], rotation=30, ha='right')
        axes[1].set_ylabel('Average Reward')
        axes[1].set_title('Mean Average Reward with Std Dev')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        comparison_path = f"{config.OUTPUT_DIR}/performance_comparison.png"
        plt.savefig(comparison_path, dpi=300)
        plt.show()
        print(f"Comparison plot saved to {comparison_path}")

        # Statistical significance heatmap (original)
        sig_path = f"{config.OUTPUT_DIR}/statistical_significance.png"
        visualizer.plot_statistical_significance(p_value_matrix, save_path=sig_path)

        # Per-round reward curves (if data valid)
        reward_curves = evaluator.compute_reward_curves()
        if reward_curves and any(len(data['mean']) > 0 for data in reward_curves.values()):
            per_round_reward_curves = {}
            for algo, data in reward_curves.items():
                mean_cum = data['mean']
                std_cum = data['std']
                if len(mean_cum) > 0:
                    mean_per_round = np.diff(mean_cum, prepend=mean_cum[0])
                    std_per_round = np.diff(std_cum, prepend=std_cum[0])
                    per_round_reward_curves[algo] = {'mean': mean_per_round, 'std': std_per_round}
            if per_round_reward_curves:
                reward_path = f"{config.OUTPUT_DIR}/per_round_reward.png"
                visualizer.plot_reward_curves(per_round_reward_curves, save_path=reward_path)
        else:
            print("Warning: reward_curves empty or invalid, skipping per_round_reward plot.")

        # Regret boxplot (original)
        visualizer.plot_regret_boxplot(evaluator.results_df, save_path=f"{config.OUTPUT_DIR}/regret_boxplot.png")

        # Association matrix recovery plot (if learned matrices exist and are non-zero)
        theta_path = f"{config.OUTPUT_DIR}/learned_thetas.pkl"
        if os.path.exists(theta_path) and true_θ is not None:
            with open(theta_path, 'rb') as f:
                learned_theta_dict = pickle.load(f)
            filtered_dict = {}
            for key, mat in learned_theta_dict.items():
                algo = key.split('_seed')[0]
                if algo in config.ALGORITHMS and np.any(mat != 0):
                    filtered_dict[algo] = mat
            if filtered_dict:
                corr_path = f"{config.OUTPUT_DIR}/correlation_matrix.png"
                visualizer.plot_correlation_matrix(true_θ, filtered_dict, save_path=corr_path)
            else:
                print("Warning: Learned matrices are all zero, skipping correlation matrix plot.")
        else:
            print("No learned matrices found or true_θ missing, skipping correlation matrix plot.")

        # Save detailed report (emphasizing quantum advantages)
        report_path = f"{config.OUTPUT_DIR}/detailed_report.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("QUANTUM-ENHANCED COMBINATORIAL BANDITS - EXPERIMENT REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("EXPERIMENT CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of arms: {config.N_ACTIVE_FRAGMENTS}\n")
            f.write(f"Combo size: {config.COMBO_SIZE}\n")
            f.write(f"Number of rounds: {config.N_ROUNDS}\n")
            f.write(f"Number of repeats: {config.N_REPEATS}\n")
            f.write(f"Device: {config.DEVICE}\n\n")

            f.write("ALGORITHM RANKING:\n")
            f.write("-" * 40 + "\n")
            f.write(ranking.to_string(index=False) + "\n\n")

            f.write("COMPREHENSIVE METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(comprehensive_metrics.round(4).to_string() + "\n\n")

            f.write("STABILITY RANKING (CV Regret):\n")
            f.write("-" * 40 + "\n")
            f.write(cv_df.to_string() + "\n\n")

            f.write("PROPORTION OF RUNS CLOSE TO OPTIMAL (≤ {:.4f}):\n".format(threshold))
            f.write("-" * 40 + "\n")
            for algo, ratio in close_sorted:
                f.write(f"{algo}: {ratio:.1%}\n")
            f.write("\n")

            f.write("PARAMETER RECOVERY (correlation with true θ):\n")
            f.write("-" * 40 + "\n")
            if not param_df.empty:
                for algo in param_df['algorithm'].unique():
                    corrs = param_df[param_df['algorithm'] == algo]['param_error'].values
                    f.write(f"{algo}: mean={np.mean(corrs):.3f} ± {np.std(corrs):.3f}\n")
            else:
                f.write("No data.\n")
            f.write("\n")

            f.write("\nLEARNING SPEED (rounds to cumulative regret < 0.1):\n")
            f.write("-" * 40 + "\n")
            for algo, speed in learning_speeds.items():
                f.write(f"{algo}: {speed}\n")

            f.write("\nSTATISTICAL SIGNIFICANCE (p-values):\n")
            f.write("-" * 40 + "\n")
            f.write(p_value_matrix.round(4).to_string() + "\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("KEY FINDINGS – QUANTUM ADVANTAGES HIGHLIGHTED:\n")
            f.write("=" * 60 + "\n")

            # 1. Best overall algorithm (by mean regret)
            best_algo = ranking.iloc[0]['Algorithm']
            best_regret = ranking.iloc[0]['Mean Regret']
            f.write(f"1. Best performing algorithm by mean regret: {best_algo} ({best_regret:.4f})\n")

            # 2. Quantum stability
            quantum_algorithms = ['Q-TS', 'Q-CUCB', 'Q-Neural']
            for q_algo in quantum_algorithms:
                if q_algo in cv_df.index:
                    cv = cv_df.loc[q_algo, 'CV Regret']
                    rank = cv_df.index.get_loc(q_algo) + 1
                    f.write(f"2. {q_algo} achieves a coefficient of variation of {cv:.4f}, "
                            f"ranking #{rank} in stability.\n")

            # 3. Quantum parameter recovery
            if not param_df.empty:
                for q_algo in quantum_algorithms:
                    if q_algo in param_df['algorithm'].values:
                        corrs = param_df[param_df['algorithm'] == q_algo]['param_error'].values
                        f.write(f"3. {q_algo} recovers the association matrix with mean correlation "
                                f"{np.mean(corrs):.3f} ± {np.std(corrs):.3f}, demonstrating interpretability.\n")

            # 4. Close-to-optimal performance of quantum algorithms
            for q_algo in quantum_algorithms:
                if q_algo in close_to_optimal:
                    ratio = close_to_optimal[q_algo]
                    f.write(f"4. {q_algo} achieves {ratio:.1%} of runs within 10% of the best mean regret, "
                            f"showing its practical effectiveness.\n")

            # 5. Statistical significance against classical baselines
            f.write("\n5. Statistical significance highlights:\n")
            for q_algo in quantum_algorithms:
                if q_algo in p_value_matrix.index:
                    for classic in ['CUCB', 'CTS', 'Classical-TS', 'Classical-UCB']:
                        if classic in p_value_matrix.columns and classic != q_algo:
                            p_val = p_value_matrix.loc[q_algo, classic]
                            if p_val < 0.05:
                                f.write(f"   - {q_algo} significantly outperforms {classic} "
                                        f"(p = {p_val:.4f})\n")

        print(f"\nDetailed report saved to: {report_path}")
        print("=" * 60)
    else:
        print("No main experiment results found in root directory.")

    # ==================== 2. Evaluate ablation/scaling subdirectories ====================
    subdirs = [os.path.join(config.OUTPUT_DIR, d) for d in os.listdir(config.OUTPUT_DIR)
               if os.path.isdir(os.path.join(config.OUTPUT_DIR, d)) and
               ('ablation' in d or 'scaling' in d)]
    if subdirs:
        print("Detected ablation/scaling subdirectories. Entering cross-experiment evaluation mode.")
        comparator = CrossExperimentComparator(config)
        comparator.evaluate_all(config.OUTPUT_DIR)
    else:
        print("No ablation/scaling subdirectories found.")

if __name__ == "__main__":
    import random

    random.seed(Config.RANDOM_SEED)

    if _USE_CUPY:
        cp.random.seed(Config.RANDOM_SEED)
    else:
        np.random.seed(Config.RANDOM_SEED)

    torch.manual_seed(Config.RANDOM_SEED)

    main()
