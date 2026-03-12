import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from config import Config


class BanditEvaluator:
    """Algorithm evaluator"""

    def __init__(self, config):
        self.config = config

    def load_results(self, summary_path):
        """Load experimental results"""
        self.results_df = pd.read_csv(summary_path)
        return self.results_df

    def compute_statistical_significance(self):
        """Compute statistical significance (t-tests)"""
        algorithms = self.results_df['algorithm'].unique()
        n_algorithms = len(algorithms)

        # Create significance matrix
        p_value_matrix = np.ones((n_algorithms, n_algorithms))

        for i, algo1 in enumerate(algorithms):
            for j, algo2 in enumerate(algorithms):
                if i >= j:
                    continue

                # Get regret values for both algorithms
                regrets1 = self.results_df[self.results_df['algorithm'] == algo1]['cumulative_regret'].values
                regrets2 = self.results_df[self.results_df['algorithm'] == algo2]['cumulative_regret'].values

                # t-test
                t_stat, p_value = stats.ttest_ind(regrets1, regrets2, equal_var=False)
                p_value_matrix[i, j] = p_value
                p_value_matrix[j, i] = p_value

        return pd.DataFrame(p_value_matrix, index=algorithms, columns=algorithms)

    def compute_ranking(self):
        """Compute algorithm ranking by mean regret"""
        # Sort by mean regret
        avg_regret = self.results_df.groupby('algorithm')['cumulative_regret'].mean().sort_values()

        # Compute ranking
        ranking = pd.DataFrame({
            'Algorithm': avg_regret.index,
            'Mean Regret': avg_regret.values,
            'Rank': range(1, len(avg_regret) + 1)
        })

        return ranking

    def compute_regret_curves(self):
        """Compute average regret curves over all seeds using glob to find all seed files."""
        import glob
        regret_curves = {}
        for algorithm in self.config.ALGORITHMS:
            algo_curves = []
            pattern = f"{self.config.OUTPUT_DIR}/curves_{algorithm}_seed*.csv"
            for curve_path in glob.glob(pattern):
                try:
                    curve_df = pd.read_csv(curve_path)
                    regret_curve = curve_df['regret'].values
                    algo_curves.append(regret_curve)
                except:
                    continue
            if algo_curves:
                min_length = min(len(c) for c in algo_curves)
                aligned_curves = [c[:min_length] for c in algo_curves]
                regret_curves[algorithm] = {
                    'mean': np.mean(aligned_curves, axis=0),
                    'std': np.std(aligned_curves, axis=0),
                    'curves': aligned_curves
                }
        return regret_curves

    def compute_reward_curves(self):
        """Compute average reward curves over all seeds"""
        reward_curves = {}
        for algorithm in self.config.ALGORITHMS:
            algo_curves = []
            for seed in range(self.config.N_REPEATS):
                curve_path = f"{self.config.OUTPUT_DIR}/curves_{algorithm}_seed{seed}.csv"
                try:
                    curve_df = pd.read_csv(curve_path)
                    reward_curve = curve_df['reward'].values
                    algo_curves.append(reward_curve)
                except:
                    continue
            if algo_curves:
                min_length = min(len(c) for c in algo_curves)
                aligned_curves = [c[:min_length] for c in algo_curves]
                reward_curves[algorithm] = {
                    'mean': np.mean(aligned_curves, axis=0),
                    'std': np.std(aligned_curves, axis=0),
                    'curves': aligned_curves
                }
        return reward_curves

    def compute_learning_speed(self, regret_curves, threshold=0.1):
        """Compute learning speed"""
        learning_speeds = {}

        for algorithm, data in regret_curves.items():
            mean_curve = data['mean']

            # Find earliest round where regret falls below threshold
            for t in range(len(mean_curve)):
                if mean_curve[t] <= threshold:
                    learning_speeds[algorithm] = t
                    break
            else:
                learning_speeds[algorithm] = len(mean_curve)

        return learning_speeds

    def compute_sample_efficiency(self):
        """Compute sample efficiency: final performance / total rounds"""
        sample_efficiency = {}

        for algorithm in self.config.ALGORITHMS:
            algo_results = self.results_df[self.results_df['algorithm'] == algorithm]
            avg_reward = algo_results['average_reward'].mean()
            sample_efficiency[algorithm] = avg_reward / self.config.N_ROUNDS

        return sample_efficiency

    def compute_association_recovery(self, true_θ, learned_θ_dict):
        """Compute quality of association matrix recovery"""
        recovery_metrics = {}

        for algorithm, learned_θ in learned_θ_dict.items():
            if learned_θ is not None:
                # Compute various metrics
                mse = np.mean((learned_θ - true_θ) ** 2)
                mae = np.mean(np.abs(learned_θ - true_θ))

                # Compute correlation
                flat_true = true_θ.flatten()
                flat_learned = learned_θ.flatten()
                correlation = np.corrcoef(flat_true, flat_learned)[0, 1]

                recovery_metrics[algorithm] = {
                    'MSE': mse,
                    'MAE': mae,
                    'Correlation': correlation
                }

        return recovery_metrics

    def compute_optimal_combo_analysis(self):
        """Analyze optimal combination discovery"""
        analysis = {}

        for algorithm in self.config.ALGORITHMS:
            algo_results = self.results_df[self.results_df['algorithm'] == algorithm]

            # Ratio of runs that found the optimal combination
            optimal_found = algo_results['optimal_round'].notna().sum()
            optimal_ratio = optimal_found / len(algo_results)

            # Average round when optimal was first found
            avg_optimal_round = algo_results['optimal_round'].mean()

            analysis[algorithm] = {
                'Optimal Found Ratio': optimal_ratio,
                'Average Optimal Round': avg_optimal_round
            }

        return analysis

    def compute_comprehensive_metrics(self):
        """Compute comprehensive evaluation metrics"""
        metrics = {}

        # Load curve data
        regret_curves = self.compute_regret_curves()

        for algorithm in self.config.ALGORITHMS:
            algo_results = self.results_df[self.results_df['algorithm'] == algorithm]

            if len(algo_results) == 0:
                continue

            # Basic metrics
            mean_regret = algo_results['cumulative_regret'].mean()
            std_regret = algo_results['cumulative_regret'].std()
            mean_reward = algo_results['average_reward'].mean()

            # Stability (coefficient of variation)
            cv_regret = std_regret / mean_regret if mean_regret > 0 else float('inf')

            # Compute AUC (area under curve)
            if algorithm in regret_curves:
                regret_auc = np.trapz(regret_curves[algorithm]['mean'])
            else:
                regret_auc = None

            metrics[algorithm] = {
                'Mean Regret': mean_regret,
                'Std Regret': std_regret,
                'CV Regret': cv_regret,
                'Mean Reward': mean_reward,
                'Regret AUC': regret_auc
            }

        return pd.DataFrame(metrics).T


class Visualizer:
    """Visualization tools"""

    def __init__(self, config):
        self.config = config
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams.update({'font.size': 12})
        self.color_palette = None

    def _get_colors(self, n):
        if n <= 20:
            return plt.cm.tab20(np.linspace(0, 1, n))
        else:
            return plt.cm.gist_ncar(np.linspace(0, 1, n))

    def plot_regret_curves(self, regret_curves, save_path=None):
        n_curves = len(regret_curves)
        if n_curves == 0:
            print("No regret curves to plot.")
            return
        colors = self._get_colors(n_curves)

        plt.figure(figsize=(12, 8))
        for (algorithm, data), color in zip(regret_curves.items(), colors):
            mean_curve = data['mean']
            std_curve = data['std']
            x = np.arange(len(mean_curve))
            plt.plot(x, mean_curve, label=algorithm, color=color, linewidth=2)
            plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                             alpha=0.15, color=color)

        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Cumulative Regret', fontsize=14)
        plt.title('Cumulative Regret Curves', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 动态调整图例列数
        ncol = min(5, (n_curves + 2) // 3)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=ncol,
                   frameon=True, fancybox=True, shadow=True, fontsize=11)

        plt.tight_layout(rect=(0, 0, 0.85, 1))
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()

    def plot_performance_comparison(self, summary_df, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        summary_df = summary_df.sort_values('Mean Regret').copy()

        # 1. Bar chart of mean regret with error bars
        ax1 = axes[0, 0]
        x = np.arange(len(summary_df))
        ax1.bar(x, summary_df['Mean Regret'], yerr=summary_df['Std Regret'],
                capsize=5, alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(summary_df['Algorithm'], rotation=30, ha='right')
        ax1.set_ylabel('Cumulative Regret')
        ax1.set_title('Mean Cumulative Regret with Standard Deviation')
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Bar chart of mean reward (ascending order)
        ax2 = axes[0, 1]
        summary_df2 = summary_df.sort_values('Mean Reward', ascending=False)
        x2 = np.arange(len(summary_df2))
        ax2.bar(x2, summary_df2['Mean Reward'], yerr=summary_df2['Std Reward'],
                capsize=5, alpha=0.8, color='green')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(summary_df2['Algorithm'], rotation=30, ha='right')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Mean Average Reward with Standard Deviation')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Optimal combination discovery rate
        ax3 = axes[1, 0]
        algorithms = summary_df['Algorithm']
        optimal_rates = summary_df['Optimal Found %']

        colors = ['red' if r < 50 else 'orange' if r < 80 else 'green' for r in optimal_rates]
        bars = ax3.bar(x, optimal_rates, color=colors, alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(algorithms, rotation=30, ha='right')
        ax3.set_ylabel('Optimal Found (%)')
        ax3.set_title('Percentage of Runs Finding Optimal Combination')
        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax3.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3, axis='y')
        for bar, rate in zip(bars, optimal_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

        # 4. Learning speed
        ax4 = axes[1, 1]
        if 'Mean Optimal Round' in summary_df.columns:
            optimal_rounds = summary_df['Mean Optimal Round'].fillna(self.config.N_ROUNDS)
            # Handle case where all are equal, avoid division by zero
            if optimal_rounds.max() == optimal_rounds.min():
                norm_rounds = np.ones_like(optimal_rounds)
            else:
                norm_rounds = optimal_rounds / optimal_rounds.max()
            cmap = plt.cm.RdYlGn_r
            colors = cmap(norm_rounds)
            bars = ax4.bar(x, optimal_rounds, color=colors, alpha=0.8)
            ax4.set_xticks(x)
            ax4.set_xticklabels(algorithms, rotation=30, ha='right')
            ax4.set_ylabel('Mean Round to Find Optimal')
            ax4.set_title('Learning Speed (Lower is Better)')
            ax4.grid(True, alpha=0.3, axis='y')
            for bar, rounds in zip(bars, optimal_rounds):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2., height + 1,
                         f'{rounds:.0f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        plt.show()

    def plot_correlation_matrix(self, true_θ, learned_θ_dict, save_path=None):
        """Plot comparison of association matrices"""
        n_algorithms = len(learned_θ_dict)
        fig, axes = plt.subplots(2, n_algorithms + 1, figsize=(5 * (n_algorithms + 1), 10))

        # True association matrix
        im0 = axes[0, 0].imshow(true_θ, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, 0].set_title('True Associations', fontweight='bold')
        axes[0, 0].set_xlabel('Fragment Index')
        axes[0, 0].set_ylabel('Fragment Index')
        plt.colorbar(im0, ax=axes[0, 0])

        # Learned association matrices
        for idx, (algorithm, learned_θ) in enumerate(learned_θ_dict.items(), 1):
            if learned_θ is not None:
                im = axes[0, idx].imshow(learned_θ, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[0, idx].set_title(f'{algorithm}', fontweight='bold')
                axes[0, idx].set_xlabel('Fragment Index')
                plt.colorbar(im, ax=axes[0, idx])

                # Error matrix
                error = learned_θ - true_θ
                im_error = axes[1, idx].imshow(error, cmap='RdBu_r',
                                               vmin=-np.max(np.abs(error)),
                                               vmax=np.max(np.abs(error)))
                axes[1, idx].set_title(f'{algorithm} Error', fontweight='bold')
                axes[1, idx].set_xlabel('Fragment Index')
                axes[1, idx].set_ylabel('Fragment Index')
                plt.colorbar(im_error, ax=axes[1, idx])

        # 隐藏多余的子图
        #for i in range(n_algorithms + 1, 2 * (n_algorithms + 1)):
            #axes[1, i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix plot saved to {save_path}")

        plt.show()

    def plot_statistical_significance(self, p_value_matrix, save_path=None):
        """
        Plot statistical significance heatmap (regular heatmap + clustered heatmap)
        - If save_path is provided, the regular heatmap is saved to save_path,
          and the clustered heatmap is saved to save_path with "_clustermap" suffix.
        - If save_path is not provided, only display without saving.
        """
        # 1. Regular heatmap (upper triangle masked)
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(p_value_matrix, dtype=bool), k=1)  # mask upper triangle (excluding diagonal)

        # Create annotations: p-value + asterisks
        annot_text = p_value_matrix.round(3).astype(str)
        for i in range(p_value_matrix.shape[0]):
            for j in range(p_value_matrix.shape[1]):
                if i < j and p_value_matrix.iloc[i, j] < 0.05:
                    if p_value_matrix.iloc[i, j] < 0.01:
                        annot_text.iloc[i, j] = annot_text.iloc[i, j] + '**'
                    else:
                        annot_text.iloc[i, j] = annot_text.iloc[i, j] + '*'

        sns.heatmap(
            p_value_matrix,
            mask=mask,
            annot=annot_text,
            fmt='',
            cmap='RdYlBu_r',
            center=0.05,
            square=True,
            cbar_kws={'label': 'p-value'},
            annot_kws={'size': 10}
        )
        plt.title('Statistical Significance (t-test p-values)', fontsize=14, fontweight='bold')
        plt.xlabel('Algorithm')
        plt.ylabel('Algorithm')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Significance heatmap saved to {save_path}")
        plt.show()

        # 2. Clustered heatmap (no mask, show all cells)
        g = sns.clustermap(
            p_value_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            center=0.05,
            linewidths=0.5,
            figsize=(12, 10),
            cbar_kws={'label': 'p-value'}
        )
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=30, ha='right')
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

        if save_path:
            # Generate clustered map path by adding "_clustermap" before extension
            base, ext = os.path.splitext(save_path)
            clustermap_path = base + "_clustermap" + ext
            g.savefig(clustermap_path, dpi=300, bbox_inches='tight')
            print(f"Significance clustermap saved to {clustermap_path}")
        plt.show()


    def plot_reward_curves(self, reward_curves, save_path=None):
        """reward_curves: {algorithm: {'mean': list, 'std': list}}"""
        plt.figure(figsize=(12, 8))
        colors = self._get_colors(len(reward_curves))
        for (algo, data), color in zip(reward_curves.items(), colors):
            mean = data['mean']
            std = data['std']
            x = np.arange(len(mean))
            plt.plot(x, mean, label=algo, color=color, linewidth=2)
            plt.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)
        plt.xlabel('Round')
        plt.ylabel('Average Reward (smoothed)')
        plt.title('Reward Learning Curves')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout(rect=(0, 0, 0.85, 1))
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_optimal_discovery_curves(self, discovery_data, save_path=None):
        """discovery_data: {algorithm: list of rounds when optimal first found (or NaN)}"""
        plt.figure(figsize=(10, 6))
        for algo, rounds in discovery_data.items():
            sorted_rounds = np.sort(rounds[~np.isnan(rounds)])
            cumulative = np.arange(1, len(sorted_rounds) + 1) / len(rounds) * 100
            plt.step(sorted_rounds, cumulative, where='post', label=algo, linewidth=2)
        plt.xlabel('Round')
        plt.ylabel('Cumulative % of Runs Found Optimal')
        plt.title('Optimal Combination Discovery Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_regret_boxplot(self, results_df, save_path=None):
        algorithms = results_df['algorithm'].unique()
        data = [results_df[results_df['algorithm'] == algo]['cumulative_regret'].values for algo in algorithms]
        plt.figure(figsize=(12, 6))
        bp = plt.boxplot(data, labels=algorithms, patch_artist=True)
        plt.xticks(rotation=30, ha='right')
        plt.ylabel('Cumulative Regret')
        plt.title('Distribution of Final Cumulative Regret')
        plt.grid(True, alpha=0.3, axis='y')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()



class CrossExperimentComparator:
    """Batch evaluation and plotting for ablation and scaling experiments"""

    def __init__(self, config):
        self.config = config
        self.visualizer = Visualizer(config)

    def parse_dirname(self, dirname):
        patterns = {
            'fisher': r'ablation_fisher_(\d+)',
            'spsa_c': r'ablation_spsa_c_([\d.]+)',
            'ent': r'ablation_ent_(\w+)',
            'K': r'scaling_K_(\d+)',
            'm': r'scaling_m_(\d+)',
        }
        for key, pat in patterns.items():
            m = re.search(pat, dirname)
            if m:
                return key, m.group(1)
        return None, None

    def load_summaries(self, dir_list):
        summaries = {}
        for d in dir_list:
            path = os.path.join(d, 'experiment_summary.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                summaries[os.path.basename(d)] = df
        return summaries

    def load_curves(self, dir_path, algorithm):
        curve_files = glob.glob(os.path.join(dir_path, f'curves_{algorithm}_seed*.csv'))
        if not curve_files:
            return None, None
        curves = []
        for f in curve_files:
            df = pd.read_csv(f)
            curves.append(df['regret'].values)
        min_len = min(len(c) for c in curves)
        aligned = [c[:min_len] for c in curves]
        mean_curve = np.mean(aligned, axis=0)
        std_curve = np.std(aligned, axis=0, ddof=1)
        return mean_curve, std_curve

    # Ablation experiment plotting
    def plot_ablation_fisher(self, dir_list, algorithms=['Q-TS', 'Q-CUCB', 'Q-Neural']):
        """Plot final regret vs. Fisher update frequency (with error bars)"""
        data = {algo: {'x': [], 'y': [], 'std': []} for algo in algorithms}
        summaries = self.load_summaries(dir_list)
        for dirname, df in summaries.items():
            key, val = self.parse_dirname(dirname)
            if key != 'fisher':
                continue
            for algo in algorithms:
                sub = df[df['algorithm'] == algo]
                if not sub.empty:
                    data[algo]['x'].append(float(val))
                    data[algo]['y'].append(sub['cumulative_regret'].mean())
                    data[algo]['std'].append(sub['cumulative_regret'].std())
        if not any(data[algo]['x'] for algo in algorithms):
            print("No Fisher ablation data found.")
            return
        plt.figure(figsize=(10, 6))
        for algo in algorithms:
            if data[algo]['x']:
                xy = sorted(zip(data[algo]['x'], data[algo]['y'], data[algo]['std']))
                x_vals, y_vals, std_vals = zip(*xy)
                plt.errorbar(x_vals, y_vals, yerr=std_vals, marker='o', capsize=5, label=algo)
        plt.xlabel('Fisher Update Frequency')
        plt.ylabel('Mean Cumulative Regret')
        plt.xscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(self.config.OUTPUT_DIR, 'ablation_fisher_comparison.png')
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"Saved: {save_path}")

    def plot_ablation_fisher_curves(self, dir_list, algorithm='Q-CUCB'):
        """Plot full regret curves under different Fisher frequencies"""
        plt.figure(figsize=(12, 8))
        for d in dir_list:
            dirname = os.path.basename(d)
            key, val = self.parse_dirname(dirname)
            if key != 'fisher':
                continue
            mean_curve, std_curve = self.load_curves(d, algorithm)
            if mean_curve is not None:
                x = np.arange(len(mean_curve))
                plt.plot(x, mean_curve, label=f'freq={val}', linewidth=2)
                plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
        plt.xlabel('Round')
        plt.ylabel('Cumulative Regret')
        plt.title(f'{algorithm} Regret Curves under Different Fisher Frequencies')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(self.config.OUTPUT_DIR, 'ablation_fisher_curves.png')
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"Saved: {save_path}")

    def plot_ablation_spsa(self, dir_list, algorithms=['Q-TS', 'Q-CUCB', 'Q-Neural']):
        """Plot final regret vs. SPSA c parameter"""
        data = {algo: {'x': [], 'y': [], 'std': []} for algo in algorithms}
        summaries = self.load_summaries(dir_list)
        for dirname, df in summaries.items():
            key, val = self.parse_dirname(dirname)
            if key != 'spsa_c':
                continue
            for algo in algorithms:
                sub = df[df['algorithm'] == algo]
                if not sub.empty:
                    data[algo]['x'].append(float(val))
                    data[algo]['y'].append(sub['cumulative_regret'].mean())
                    data[algo]['std'].append(sub['cumulative_regret'].std())
        if not any(data[algo]['x'] for algo in algorithms):
            print("No SPSA c ablation data found.")
            return
        plt.figure(figsize=(10, 6))
        for algo in algorithms:
            if data[algo]['x']:
                xy = sorted(zip(data[algo]['x'], data[algo]['y'], data[algo]['std']))
                x_vals, y_vals, std_vals = zip(*xy)
                plt.errorbar(x_vals, y_vals, yerr=std_vals, marker='o', capsize=5, label=algo)
        plt.xlabel('SPSA c')
        plt.ylabel('Mean Cumulative Regret')
        plt.xscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(self.config.OUTPUT_DIR, 'ablation_spsa_comparison.png')
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"Saved: {save_path}")

    def plot_ablation_ent(self, dir_list, algorithms=['Q-TS', 'Q-CUCB', 'Q-Neural']):
        """Plot final regret bar chart for different entanglement structures"""
        data = {algo: {'labels': [], 'means': [], 'stds': []} for algo in algorithms}
        summaries = self.load_summaries(dir_list)
        for dirname, df in summaries.items():
            key, val = self.parse_dirname(dirname)
            if key != 'ent':
                continue
            for algo in algorithms:
                sub = df[df['algorithm'] == algo]
                if not sub.empty:
                    data[algo]['labels'].append(val)
                    data[algo]['means'].append(sub['cumulative_regret'].mean())
                    data[algo]['stds'].append(sub['cumulative_regret'].std())
        if not any(data[algo]['labels'] for algo in algorithms):
            print("No entanglement structure ablation data found.")
            return
        x = np.arange(len(data[algorithms[0]]['labels']))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, algo in enumerate(algorithms):
            if data[algo]['labels']:
                ax.bar(x + i * width, data[algo]['means'], width, yerr=data[algo]['stds'],
                       capsize=5, label=algo)
        ax.set_xlabel('Entanglement Structure')
        ax.set_ylabel('Mean Cumulative Regret')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(data[algorithms[0]]['labels'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        save_path = os.path.join(self.config.OUTPUT_DIR, 'ablation_ent_comparison.png')
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"Saved: {save_path}")

    # Scaling experiment plotting
    def plot_scaling_K(self, dir_list, algorithms=['Q-TS', 'Classical-TS', 'CUCB']):
        """Plot final regret vs. number of arms K"""
        data = {algo: {'x': [], 'y': [], 'std': []} for algo in algorithms}
        summaries = self.load_summaries(dir_list)
        for dirname, df in summaries.items():
            key, val = self.parse_dirname(dirname)
            if key != 'K':
                continue
            for algo in algorithms:
                sub = df[df['algorithm'] == algo]
                if not sub.empty:
                    data[algo]['x'].append(int(val))
                    data[algo]['y'].append(sub['cumulative_regret'].mean())
                    data[algo]['std'].append(sub['cumulative_regret'].std())
        if not any(data[algo]['x'] for algo in algorithms):
            print("No scaling K data found.")
            return
        plt.figure(figsize=(10, 6))
        for algo in algorithms:
            if data[algo]['x']:
                xy = sorted(zip(data[algo]['x'], data[algo]['y'], data[algo]['std']))
                x_vals, y_vals, std_vals = zip(*xy)
                plt.errorbar(x_vals, y_vals, yerr=std_vals, marker='o', capsize=5, label=algo)
        plt.xlabel('Number of Arms (K)')
        plt.ylabel('Mean Cumulative Regret')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(self.config.OUTPUT_DIR, 'scaling_K_comparison.png')
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"Saved: {save_path}")

    def plot_scaling_m(self, dir_list, algorithms=['Q-TS', 'Classical-TS', 'CUCB']):
        """Plot final regret vs. combo size m"""
        data = {algo: {'x': [], 'y': [], 'std': []} for algo in algorithms}
        summaries = self.load_summaries(dir_list)
        for dirname, df in summaries.items():
            key, val = self.parse_dirname(dirname)
            if key != 'm':
                continue
            for algo in algorithms:
                sub = df[df['algorithm'] == algo]
                if not sub.empty:
                    data[algo]['x'].append(int(val))
                    data[algo]['y'].append(sub['cumulative_regret'].mean())
                    data[algo]['std'].append(sub['cumulative_regret'].std())
        if not any(data[algo]['x'] for algo in algorithms):
            print("No scaling m data found.")
            return
        plt.figure(figsize=(10, 6))
        for algo in algorithms:
            if data[algo]['x']:
                xy = sorted(zip(data[algo]['x'], data[algo]['y'], data[algo]['std']))
                x_vals, y_vals, std_vals = zip(*xy)
                plt.errorbar(x_vals, y_vals, yerr=std_vals, marker='o', capsize=5, label=algo)
        plt.xlabel('Combo Size (m)')
        plt.ylabel('Mean Cumulative Regret')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(self.config.OUTPUT_DIR, 'scaling_m_comparison.png')
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"Saved: {save_path}")

    def evaluate_all(self, root_dir):
        subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, d))]
        fisher_dirs = [d for d in subdirs if 'ablation_fisher' in d]
        spsa_dirs = [d for d in subdirs if 'ablation_spsa' in d]
        ent_dirs = [d for d in subdirs if 'ablation_ent' in d]
        K_dirs = [d for d in subdirs if 'scaling_K' in d]
        m_dirs = [d for d in subdirs if 'scaling_m' in d]

        if fisher_dirs:
            self.plot_ablation_fisher(fisher_dirs)
            self.plot_ablation_fisher_curves(fisher_dirs, algorithm='Q-CUCB')
        if spsa_dirs:
            self.plot_ablation_spsa(spsa_dirs)
        if ent_dirs:
            self.plot_ablation_ent(ent_dirs)
        if K_dirs:
            self.plot_scaling_K(K_dirs)
        if m_dirs:
            self.plot_scaling_m(m_dirs)
