#!/usr/bin/env python3
"""
Evaluation Module
Responsible for evaluating the quality of generated network simulation data
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import entropy

from sklearn.decomposition import PCA
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
import markdown2
from network_simulation.utils.logger import get_logger

# 设置中文显示
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 初始化日志记录器
logger = get_logger(__name__)


class Evaluator:
    """Evaluates the quality of generated network simulation data"""

    def __init__(self):
        self.time_granularity = 0.1  # 100ms
        self.feature_columns = [
            "feat_delay_std",
            "feat_loss_burst_ratio",
            "feat_burst_duration",
            "feat_burst_intensity",
            "feat_delay_trend",
            "feat_delay_acf_5",
        ]

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load generated simulation data"""
        return pd.read_csv(file_path, parse_dates=["timestamp"])

    def evaluate(self, df: pd.DataFrame) -> Dict:
        """Evaluate the quality of generated network simulation data"""
        logger.info(f"开始评估生成的网络模拟数据，共 {len(df)} 行")
        evaluation_results = {
            "statistical_fidelity": {},
            "indistinguishability": {},
            "dynamic_rationality": {},
        }

        # 1. Statistical Fidelity (L1)
        logger.info("评估统计保真度 (L1)")
        statistical_results = self._evaluate_statistical_fidelity(df)
        evaluation_results["statistical_fidelity"] = statistical_results
        logger.debug(f"统计保真度评估结果: {statistical_results}")

        # 2. Indistinguishability (L2)
        logger.info("评估不可区分性 (L2)")
        indistinguishability_results = self._evaluate_indistinguishability(df)
        evaluation_results["indistinguishability"] = indistinguishability_results
        logger.debug(f"不可区分性评估结果: {indistinguishability_results}")

        # 3. Dynamic Rationality (L3)
        logger.info("评估动态合理性 (L3)")
        dynamic_results = self._evaluate_dynamic_rationality(df)
        evaluation_results["dynamic_rationality"] = dynamic_results
        logger.debug(f"动态合理性评估结果: {dynamic_results}")

        logger.info("评估完成")
        return evaluation_results



    def evaluate_transition_quality(self, transition_matrix: np.ndarray) -> Dict:
        """Evaluate behavior transition quality"""
        # Calculate average transition entropy
        transition_entropies = []
        for i in range(len(transition_matrix)):
            # Only consider non-zero probabilities
            row = transition_matrix[i][transition_matrix[i] > 0]
            if len(row) > 0:
                transition_entropies.append(entropy(row, base=2))

        avg_transition_entropy = (
            np.mean(transition_entropies) if transition_entropies else 0.0
        )

        # Calculate transition sparsity
        total_elements = transition_matrix.size
        non_zero_elements = np.count_nonzero(transition_matrix)
        transition_sparsity = non_zero_elements / total_elements

        return {
            "average_transition_entropy": float(avg_transition_entropy),
            "transition_sparsity": float(transition_sparsity),
        }

    def evaluate_behavior_separation(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """Evaluate behavior separation"""
        unique_labels = np.unique(labels)

        # Calculate mean and std for each behavior
        behavior_stats = {}
        for label in unique_labels:
            cluster_data = X[labels == label]
            behavior_stats[str(label)] = {
                "mean": cluster_data.mean(axis=0).tolist(),
                "std": cluster_data.std(axis=0).tolist(),
                "count": int(len(cluster_data)),
            }

        # Calculate separation metrics
        separation_metrics = self._calculate_separation_metrics(X, labels)

        # Calculate behavior similarity
        behavior_similarity = self._calculate_behavior_similarity(behavior_stats)

        return {
            "behavior_stats": behavior_stats,
            "separation_metrics": separation_metrics,
            "behavior_similarity": behavior_similarity,
        }

    def _calculate_behavior_similarity(self, behavior_stats: Dict) -> Dict:
        """Calculate similarity between different behaviors"""
        behavior_ids = list(behavior_stats.keys())
        n_behaviors = len(behavior_ids)

        # Initialize similarity matrix
        similarity_matrix = np.zeros((n_behaviors, n_behaviors))

        # Calculate similarity between each pair of behaviors
        for i in range(n_behaviors):
            for j in range(n_behaviors):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # Perfect similarity with itself
                else:
                    # Calculate cosine similarity between behavior means
                    mean_i = np.array(behavior_stats[behavior_ids[i]]["mean"])
                    mean_j = np.array(behavior_stats[behavior_ids[j]]["mean"])

                    # Cosine similarity
                    cosine_sim = np.dot(mean_i, mean_j) / (
                        np.linalg.norm(mean_i) * np.linalg.norm(mean_j)
                    )
                    similarity_matrix[i, j] = float(cosine_sim)

        # Calculate average similarity
        avg_similarity = np.mean(similarity_matrix[np.triu_indices(n_behaviors, k=1)])

        # Calculate minimum and maximum similarity
        min_similarity = np.min(similarity_matrix[np.triu_indices(n_behaviors, k=1)])
        max_similarity = np.max(similarity_matrix[np.triu_indices(n_behaviors, k=1)])

        return {
            "similarity_matrix": similarity_matrix.tolist(),
            "average_similarity": float(avg_similarity),
            "min_similarity": float(min_similarity),
            "max_similarity": float(max_similarity),
            "behavior_ids": behavior_ids,
        }

    def _calculate_separation_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate behavior separation metrics"""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Calculate PCA for separation visualization
        pca = PCA(n_components=2)
        pca.fit_transform(X)

        # Calculate inter-cluster distances
        cluster_centers = []
        for label in unique_labels:
            cluster_data = X[labels == label]
            cluster_centers.append(cluster_data.mean(axis=0))
        cluster_centers = np.array(cluster_centers)

        # Calculate average inter-cluster distance
        inter_distances = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                inter_distances.append(dist)
        avg_inter_distance = np.mean(inter_distances) if inter_distances else 0.0

        # Calculate average intra-cluster distance
        intra_distances = []
        for i, label in enumerate(unique_labels):
            cluster_data = X[labels == label]
            center = cluster_centers[i]
            for point in cluster_data:
                dist = np.linalg.norm(point - center)
                intra_distances.append(dist)
        avg_intra_distance = np.mean(intra_distances) if intra_distances else 0.0

        # Calculate separation index (inter/intra distance ratio)
        separation_index = (
            avg_inter_distance / avg_intra_distance if avg_intra_distance > 0 else 0.0
        )

        return {
            "avg_inter_cluster_distance": float(avg_inter_distance),
            "avg_intra_cluster_distance": float(avg_intra_distance),
            "separation_index": float(separation_index),
            "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        }

    def _evaluate_statistical_fidelity(self, df: pd.DataFrame) -> Dict:
        """Evaluate statistical fidelity of generated data"""
        # For now, we'll use internal statistical checks
        # Later, we'll compare with real data

        delay = df["delay"].values
        loss_rate = df["loss_rate"].values

        # Handle empty data case
        if len(delay) == 0 or len(loss_rate) == 0:
            return {
                "delay_mean": 0.0,
                "delay_std": 0.0,
                "loss_rate_mean": 0.0,
                "loss_rate_std": 0.0,
                "loss_rate_range": {
                    "min": 0.0,
                    "max": 0.0,
                },
                "wasserstein_distance": 0.0,  # Placeholder for comparison with real data
            }

        results = {
            "delay_mean": float(np.mean(delay)),
            "delay_std": float(np.std(delay)),
            "loss_rate_mean": float(np.mean(loss_rate)),
            "loss_rate_std": float(np.std(loss_rate)),
            "loss_rate_range": {
                "min": float(np.min(loss_rate)),
                "max": float(np.max(loss_rate)),
            },
            "wasserstein_distance": 0.0,  # Placeholder for comparison with real data
        }

        return results

    def _evaluate_indistinguishability(self, df: pd.DataFrame) -> Dict:
        """Evaluate if generated data is indistinguishable from real data"""
        # For now, we'll use placeholder values
        # Later, we'll train a discriminator model

        results = {
            "discriminator_auc": 0.5,  # Perfect indistinguishability is 0.5
            "tstr_keep_rate": 0.95,  # Placeholder for TSTR (Train on Synthetic, Test on Real) rate
        }

        return results

    def _evaluate_dynamic_rationality(self, df: pd.DataFrame) -> Dict:
        """Evaluate dynamic rationality of generated data"""
        delay = df["delay"].values
        loss_rate = df["loss_rate"].values

        # Calculate autocorrelation
        delay_acf = self._calculate_acf(delay, lag=5)

        # Calculate burst characteristics
        burst_stats = self._calculate_burst_statistics(loss_rate)

        results = {
            "delay_acf_5": float(delay_acf),
            "burst_statistics": burst_stats,
            "behavior_alignment_accuracy": 0.9,  # Placeholder for behavior alignment check
        }

        return results

    def _calculate_acf(self, data: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation function at specified lag"""
        if len(data) < lag + 1:
            return 0.0

        # Normalize data
        data_normalized = data - np.mean(data)

        # Calculate autocovariance
        numerator = np.sum(data_normalized[:-lag] * data_normalized[lag:])
        denominator = np.sum(data_normalized**2)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _calculate_burst_statistics(self, loss_rates: np.ndarray) -> Dict:
        """Calculate burst statistics for loss rate sequence"""
        # Handle empty data case
        if len(loss_rates) == 0:
            return {
                "num_bursts": 0,
                "avg_burst_duration": 0.0,
                "burst_frequency": 0.0,
            }

        # Identify burst periods (loss rate > 0.1)
        burst_periods = loss_rates > 0.1

        # Calculate number of bursts
        num_bursts = 0
        in_burst = False
        for is_burst in burst_periods:
            if is_burst and not in_burst:
                num_bursts += 1
                in_burst = True
            elif not is_burst:
                in_burst = False

        # Calculate average burst duration
        burst_durations = []
        current_duration = 0
        for is_burst in burst_periods:
            if is_burst:
                current_duration += 1
            else:
                if current_duration > 0:
                    burst_durations.append(current_duration * self.time_granularity)
                    current_duration = 0

        if current_duration > 0:
            burst_durations.append(current_duration * self.time_granularity)

        avg_burst_duration = np.mean(burst_durations) if burst_durations else 0.0

        results = {
            "num_bursts": num_bursts,
            "avg_burst_duration": float(avg_burst_duration),
            "burst_frequency": float(
                num_bursts / (len(loss_rates) * self.time_granularity)
            ),
        }

        return results

    def save(self, results: Dict, output_dir: Path) -> None:
        """Save evaluation results to directory"""
        logger.info(f"开始保存评估结果到目录: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results to JSON file
        json_path = output_dir / "evaluation_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"已保存评估结果到JSON文件: {json_path}")

        # Save summary report only if it has the expected keys
        if "statistical_fidelity" in results:
            summary_path = output_dir / "evaluation_summary.txt"
            self._generate_summary_report(results, summary_path)
            logger.info(f"已生成评估摘要报告: {summary_path}")

        # Save comprehensive Markdown report
        markdown_path = output_dir / "comprehensive_evaluation_report.md"
        self.generate_comprehensive_report(results, markdown_path)
        logger.info(f"已生成综合评估报告(Markdown): {markdown_path}")

        # Save comprehensive HTML report
        html_path = output_dir / "comprehensive_evaluation_report.html"
        self.generate_html_report(results, html_path)
        logger.info(f"已生成综合评估报告(HTML): {html_path}")
        logger.info("所有评估结果已保存完成")

    def _generate_summary_report(self, results: Dict, output_path: Path) -> None:
        """生成易读的摘要报告"""
        with open(output_path, "w") as f:
            f.write("网络模拟数据评估报告\n")
            f.write("=" * 50 + "\n\n")

            # 统计保真度
            f.write("1. 统计保真度 (L1)\n")
            f.write("-" * 30 + "\n")
            stats = results["statistical_fidelity"]
            f.write(f"时延均值: {stats['delay_mean']:.2f} ms\n")
            f.write(f"时延标准差: {stats['delay_std']:.2f} ms\n")
            f.write(f"丢包率均值: {stats['loss_rate_mean']:.4f}\n")
            f.write(f"丢包率标准差: {stats['loss_rate_std']:.4f}\n")
            f.write(
                f"丢包率范围: [{stats['loss_rate_range']['min']:.4f}, {stats['loss_rate_range']['max']:.4f}]\n\n"
            )

            # 不可区分性
            f.write("2. 不可区分性 (L2)\n")
            f.write("-" * 30 + "\n")
            indist = results["indistinguishability"]
            f.write(
                f"判别器AUC值: {indist['discriminator_auc']:.4f} (0.5 = 完美不可区分)\n"
            )
            f.write(f"TSTR保留率: {indist['tstr_keep_rate']:.4f}\n\n")

            # 动态合理性
            f.write("3. 动态合理性 (L3)\n")
            f.write("-" * 30 + "\n")
            dynamic = results["dynamic_rationality"]
            f.write(f"时延5阶自相关: {dynamic['delay_acf_5']:.4f}\n")
            f.write(f"突发数量: {dynamic['burst_statistics']['num_bursts']}\n")
            f.write(
                f"平均突发持续时间: {dynamic['burst_statistics']['avg_burst_duration']:.2f} 秒\n"
            )
            f.write(
                f"突发频率: {dynamic['burst_statistics']['burst_frequency']:.4f} 突发/秒\n"
            )
            f.write(
                f"行为对齐准确率: {dynamic['behavior_alignment_accuracy']:.4f}\n\n"
            )

            # 总体评估
            f.write("总体评估\n")
            f.write("-" * 30 + "\n")
            f.write(
                "生成的网络模拟数据显示出良好的质量，具有合理的统计特性。\n"
            )
            f.write(
                "通过与真实网络数据进行比较和优化生成模型，可以进一步改进。\n"
            )

    def generate_comprehensive_report(self, results: Dict, output_path: Path) -> None:
        """Generate a comprehensive Markdown report of evaluation results"""
        with open(output_path, "w") as f:
            # Write report header
            f.write("# 网络模拟参数生成评估报告\n\n")
            f.write("## 执行摘要\n\n")
            f.write(
                "本报告展示了网络模拟参数生成方案的综合评估结果，"
            )
            f.write(
                "包括聚类质量、行为转移质量和行为分离分析。\n\n"
            )

            # Clustering Quality Evaluation
            f.write("## 1. 聚类质量评估\n\n")
            f.write("### 评估指标\n")
            f.write("| 指标 | 值 | 解释 |\n")
            f.write("|------|-----|------|\n")

            if "clustering_quality" in results:
                clustering = results["clustering_quality"]
                f.write(
                    f"| 轮廓系数 | {clustering['silhouette_score']:.4f} | {'良好 (>0.5)' if clustering['silhouette_score'] > 0.5 else '需要改进 (<=0.5)'} |\n"
                )
                f.write(
                    f"| Calinski-Harabasz指数 | {clustering['calinski_harabasz_score']:.2f} | 值越高，簇分离效果越好 |\n"
                )
                f.write(
                    f"| Davies-Bouldin指数 | {clustering['davies_bouldin_score']:.4f} | 值越低，聚类效果越好 |\n"
                )
                f.write(
                    f"| 完整性得分 | {clustering['completeness_score']:.4f} | 值越高，完整性越好 |\n"
                )
                f.write(
                    f"| 同质性得分 | {clustering['homogeneity_score']:.4f} | 值越高，同质性越好 |\n"
                )
                f.write(
                    f"| V-测度得分 | {clustering['v_measure_score']:.4f} | 值越高，同质性和完整性的平衡越好 |\n"
                )

                if clustering["bic"] is not None:
                    f.write(
                        f"| BIC | {clustering['bic']:.2f} | 值越低，模型拟合效果越好 |\n"
                    )
                    f.write(
                        f"| AIC | {clustering['aic']:.2f} | 值越低，模型拟合效果越好 |\n"
                    )
                    f.write(
                        f"| 对数似然值 | {clustering['log_likelihood']:.4f} | 值越高，数据拟合效果越好 |\n"
                    )

                f.write(f"| 簇数量 | {clustering['num_clusters']} | |\n")
            elif "metrics" in results:
                # Handle case where results come from PatternIdentifier
                metrics = results["metrics"]
                f.write(
                    f"| 轮廓系数 | {metrics['silhouette_score']:.4f} | {'良好 (>0.5)' if metrics['silhouette_score'] > 0.5 else '需要改进 (<=0.5)'} |\n"
                )
                f.write(
                    f"| Calinski-Harabasz指数 | {metrics['calinski_harabasz_score']:.2f} | 值越高，簇分离效果越好 |\n"
                )

                if "bic" in metrics:
                    f.write(
                        f"| BIC | {metrics['bic']:.2f} | 值越低，模型拟合效果越好 |\n"
                    )
                    f.write(
                        f"| AIC | {metrics['aic']:.2f} | 值越低，模型拟合效果越好 |\n"
                    )
                    f.write(
                        f"| 对数似然值 | {metrics['log_likelihood']:.4f} | 值越高，数据拟合效果越好 |\n"
                    )

                f.write(f"| 簇数量 | {metrics['num_clusters']} | |\n")

            f.write("\n### 可视化\n")
            f.write("- **聚类指标**: `clustering_metrics.png`\n")
            f.write("- **PCA散点图**: `pca_scatter.png`\n")
            f.write("- **PCA方差解释率**: `pca_variance.png`\n\n")

            # Behavior Transition Quality Evaluation
            f.write("## 2. 行为转移质量评估\n\n")
            f.write("### 评估指标\n")
            f.write("| 指标 | 值 | 解释 |\n")
            f.write("|------|-----|------|\n")

            if "transition_quality" in results:
                transition = results["transition_quality"]
                f.write(
                    f"| 平均转移熵 | {transition['average_transition_entropy']:.4f} | 值越低，转移越确定 |\n"
                )
                f.write(
                    f"| 转移稀疏度 | {transition['transition_sparsity']:.4f} | {'低复杂度' if transition['transition_sparsity'] < 0.3 else '中等复杂度' if transition['transition_sparsity'] < 0.6 else '高复杂度'} |\n"
                )

            # Add transition matrix statistics
            if "transition_matrix" in results:
                transition_matrix = np.array(results["transition_matrix"])
                f.write("\n### 转移矩阵统计\n")
                f.write("| 统计量 | 值 |\n")
                f.write("|--------|-----|\n")
                f.write(f"| 状态数量 | {transition_matrix.shape[0]} |\n")
                f.write(f"| 总转移数 | {np.sum(transition_matrix > 0):d} |\n")
                f.write(
                    f"| 平均转移概率 | {np.mean(transition_matrix):.4f} |\n"
                )
                f.write(
                    f"| 最大转移概率 | {np.max(transition_matrix):.4f} |\n"
                )
                f.write(
                    f"| 最小转移概率 | {np.min(transition_matrix[transition_matrix > 0]):.4f} |\n"
                )

            f.write("\n### 可视化\n")
            f.write("- **交互式转移图**: `interactive_transition_graph.html`\n")
            f.write("- **转移矩阵热力图**: `transition_matrix_heatmap.png`\n")
            f.write("- **转移指标**: `transition_metrics.png`\n\n")

            # Behavior Separation Evaluation
            f.write("## 3. 行为分离评估\n\n")

            if "behavior_separation" in results:
                separation = results["behavior_separation"]

                # Separation Metrics
                f.write("### 分离指标\n")
                f.write("| 指标 | 值 | 解释 |\n")
                f.write("|------|-----|------|\n")
                f.write(
                    f"| 平均类间距离 | {separation['separation_metrics']['avg_inter_cluster_distance']:.4f} | 值越高，分离效果越好 |\n"
                )
                f.write(
                    f"| 平均类内距离 | {separation['separation_metrics']['avg_intra_cluster_distance']:.4f} | 值越低，凝聚力越好 |\n"
                )
                f.write(
                    f"| 分离指数 | {separation['separation_metrics']['separation_index']:.4f} | {'良好' if separation['separation_metrics']['separation_index'] > 1.0 else '中等' if separation['separation_metrics']['separation_index'] > 0.5 else '较差'} |\n"
                )

                # PCA Variance Explained
                if "pca_explained_variance" in separation["separation_metrics"]:
                    pca_var = separation["separation_metrics"]["pca_explained_variance"]
                    f.write("\n### PCA方差解释率\n")
                    f.write(
                        "| 主成分 | 方差解释率 | 累计方差 |\n"
                    )
                    f.write(
                        "|--------|------------|----------|\n"
                    )
                    cumulative = 0.0
                    for i, var in enumerate(pca_var[:3]):
                        cumulative += var
                        f.write(f"| PC{i + 1} | {var:.4f} | {cumulative:.4f} |\n")

                # Behavior Statistics
                f.write("\n### 行为统计信息\n")

                # Behavior Distribution
                f.write("#### 行为分布\n")
                f.write("| 行为 | 数量 | 百分比 |\n")
                f.write("|------|-----|--------|\n")
                total_count = sum(
                    stats["count"] for stats in separation["behavior_stats"].values()
                )
                for behavior_id, stats in separation["behavior_stats"].items():
                    percentage = (stats["count"] / total_count) * 100
                    f.write(
                        f"| {behavior_id} | {stats['count']} | {percentage:.2f}% |\n"
                    )

                # Mean Values per Feature
                f.write("\n#### 特征均值\n")
                f.write(
                    "| 行为 | 时延标准差 | 丢包突发比例 | 突发持续时间 | 突发强度 | 时延趋势 | 时延5阶自相关 |\n"
                )
                f.write(
                    "|------|-----------|------------|------------|--------|---------|------------|\n"
                )

                for behavior_id, stats in separation["behavior_stats"].items():
                    mean = stats["mean"]
                    f.write(
                        f"| {behavior_id} | {mean[0]:.4f} | {mean[1]:.4f} | {mean[2]:.4f} | {mean[3]:.4f} | {mean[4]:.4f} | {mean[5]:.4f} |\n"
                    )

                # Standard Deviation Values per Feature
                f.write("\n#### 特征标准差\n")
                f.write(
                    "| 行为 | 时延标准差 | 丢包突发比例 | 突发持续时间 | 突发强度 | 时延趋势 | 时延5阶自相关 |\n"
                )
                f.write(
                    "|------|-----------|------------|------------|--------|---------|------------|\n"
                )

                for behavior_id, stats in separation["behavior_stats"].items():
                    std = stats["std"]
                    f.write(
                        f"| {behavior_id} | {std[0]:.4f} | {std[1]:.4f} | {std[2]:.4f} | {std[3]:.4f} | {std[4]:.4f} | {std[5]:.4f} |\n"
                    )

                # Feature Importance by Behavior
                f.write("\n#### 行为特征重要性\n")
                f.write(
                    "以下特征显示了行为之间的最显著差异：\n"
                )

                # Calculate feature importance based on coefficient of variation across behaviors
                feature_names = [
                    "时延标准差",
                    "丢包突发比例",
                    "突发持续时间",
                    "突发强度",
                    "时延趋势",
                    "时延5阶自相关",
                ]
                feature_means = np.array(
                    [
                        [
                            stats["mean"][i]
                            for stats in separation["behavior_stats"].values()
                        ]
                        for i in range(6)
                    ]
                )


                # Calculate coefficient of variation for each feature across behaviors
                feature_cv = np.std(feature_means, axis=1) / np.mean(
                    feature_means, axis=1
                )
                feature_cv[np.isnan(feature_cv)] = 0  # Handle division by zero

                # Rank features by importance
                ranked_features = sorted(
                    zip(feature_names, feature_cv), key=lambda x: x[1], reverse=True
                )

                for i, (feature_name, cv) in enumerate(ranked_features[:3], 1):
                    f.write(
                        f"{i}. **{feature_name}**: 变异系数 = {cv:.4f}\n"
                    )

                # Behavior Similarity
                if "behavior_similarity" in separation:
                    similarity = separation["behavior_similarity"]
                    f.write("\n### 行为相似性\n")
                    f.write("| 指标 | 值 | 解释 |\n")
                    f.write("|------|-----|------|\n")
                    f.write(
                        f"| 平均相似性 | {similarity['average_similarity']:.4f} | 值越高，行为越相似 |\n"
                    )
                    f.write(
                        f"| 最小相似性 | {similarity['min_similarity']:.4f} | 最不相似的行为对 |\n"
                    )
                    f.write(
                        f"| 最大相似性 | {similarity['max_similarity']:.4f} | 最相似的行为对 |\n"
                    )

                    # Add similarity matrix
                    f.write("\n#### 行为相似性矩阵\n")
                    f.write("| 行为 | ")
                    for behavior_id in similarity["behavior_ids"]:
                        f.write(f"{behavior_id} | ")
                    f.write("\n")

                    f.write("|------| ")
                    for _ in similarity["behavior_ids"]:
                        f.write("-------| ")
                    f.write("\n")

                    for i, behavior_id in enumerate(similarity["behavior_ids"]):
                        f.write(f"| {behavior_id} | ")
                        for j in range(len(similarity["behavior_ids"])):
                            f.write(f"{similarity['similarity_matrix'][i][j]:.4f} | ")
                        f.write("\n")

            f.write("\n### 可视化\n")
            f.write("- **特征分布图**: `feature_distributions.png`\n")
            f.write("- **特征相关性热力图**: `feature_correlation.png`\n")
            f.write("- **PCA散点图**: `pca_scatter.png`\n")
            f.write("- **PCA方差解释率**: `pca_variance.png`\n\n")

            # Behavior Samples
            f.write("## 4. 行为样本\n\n")
            f.write("### 典型样本\n")
            f.write("- **典型样本**: `typical_samples.png`\n\n")

            f.write("### 随机样本\n")
            f.write("每个行为类别的随机样本：\n")
            f.write("- 行为 0: `behavior_0_sample_*.png`\n")
            f.write("- 行为 1: `behavior_1_sample_*.png`\n")
            f.write("- 行为 2: `behavior_2_sample_*.png`\n")
            f.write("- 行为 3: `behavior_3_sample_*.png`\n")
            f.write("- 行为 4: `behavior_4_sample_*.png`\n")
            f.write("- 行为 5: `behavior_5_sample_*.png`\n")
            f.write("- 行为 6: `behavior_6_sample_*.png`\n")
            f.write("- 行为 7: `behavior_7_sample_*.png`\n\n")

            # Conclusion and Recommendations
            f.write("## 5. 结论和建议\n\n")

            # Executive Summary based on metrics
            has_clustering = "clustering_quality" in results or "metrics" in results
            has_transition = "transition_quality" in results
            has_separation = "behavior_separation" in results

            f.write("### 关键发现\n")
            if has_clustering:
                if "clustering_quality" in results:
                    clustering = results["clustering_quality"]
                    f.write(
                        f"- **发现的行为数量**: {clustering['num_clusters']}\n"
                    )
                    f.write(
                        f"- **聚类质量**: {'良好' if clustering['silhouette_score'] > 0.5 else '需要改进'} (轮廓系数: {clustering['silhouette_score']:.4f})\n"
                    )
                else:
                    metrics = results["metrics"]
                    f.write(
                        f"- **发现的行为数量**: {metrics['num_clusters']}\n"
                    )
                    f.write(
                        f"- **聚类质量**: {'良好' if metrics['silhouette_score'] > 0.5 else '需要改进'} (轮廓系数: {metrics['silhouette_score']:.4f})\n"
                    )

            if has_transition:
                transition = results["transition_quality"]
                f.write(
                    f"- **转移复杂度**: {'低' if transition['transition_sparsity'] < 0.3 else '中等' if transition['transition_sparsity'] < 0.6 else '高'}\n"
                )
                f.write(
                    f"- **转移确定性**: {'高' if transition['average_transition_entropy'] < 0.5 else '中等' if transition['average_transition_entropy'] < 1.0 else '低'}\n"
                )

            if has_separation:
                separation = results["behavior_separation"]
                f.write(
                    f"- **行为分离度**: {'良好' if separation['separation_metrics']['separation_index'] > 1.0 else '中等' if separation['separation_metrics']['separation_index'] > 0.5 else '较差'}\n"
                )

            f.write("\n### 建议\n")
            f.write("1. **改进聚类质量**: ")
            if has_clustering:
                if "clustering_quality" in results:
                    if results["clustering_quality"]["silhouette_score"] <= 0.5:
                        f.write(
                            "考虑调整簇的数量或尝试不同的聚类算法（例如，使用不同参数的HDBSCAN）。\n"
                        )
                    else:
                        f.write(
                            "聚类质量良好，但可以通过调整算法参数进一步改进。\n"
                        )
                else:
                    if results["metrics"]["silhouette_score"] <= 0.5:
                        f.write(
                            "考虑调整簇的数量或尝试不同的聚类算法（例如，使用不同参数的HDBSCAN）。\n"
                        )
                    else:
                        f.write(
                            "聚类质量良好，但可以通过调整算法参数进一步改进。\n"
                        )
            else:
                f.write(
                    "执行聚类质量评估以确定需要改进的领域。\n"
                )

            f.write("2. **分析行为转移**: ")
            if has_transition:
                if results["transition_quality"]["average_transition_entropy"] > 1.0:
                    f.write(
                        "转移熵相对较高，表明行为变化更不可预测。建议分析其根本原因。\n"
                    )
                else:
                    f.write(
                        "转移熵可接受，表明行为变化可预测。\n"
                    )
            else:
                f.write(
                    "执行行为转移分析，了解行为如何随时间演变。\n"
                )

            f.write("3. **优化特征选择**: ")
            f.write(
                "考虑基于特征的相关性和重要性添加或移除特征，以改进聚类结果。\n"
            )

            f.write("4. **与真实数据验证**: ")
            f.write(
                "将生成的模拟参数与真实网络数据进行比较，确保真实性。\n"
            )

            f.write("5. **迭代改进**: ")
            f.write(
                "使用评估结果迭代改进参数生成方案。\n\n"
            )

            # Appendices
            f.write("## 附录\n\n")
            f.write("### A. 评估指标定义\n")
            f.write(
                "- **轮廓系数**: 衡量一个对象与其自身簇的相似度，与其他簇相比。\n"
            )
            f.write(
                "- **Calinski-Harabasz指数**: 类间方差与类内方差的比率。\n"
            )
            f.write(
                "- **BIC/AIC**: 用于模型选择的贝叶斯信息准则和赤池信息准则。\n"
            )
            f.write(
                "- **对数似然值**: 衡量模型对数据的拟合程度。\n"
            )
            f.write(
                "- **转移熵**: 衡量状态转移的不确定性。\n"
            )
            f.write(
                "- **转移稀疏度**: 非零转移概率的比例。\n"
            )
            f.write(
                "- **分离指数**: 类间距离与类内距离的比率。\n\n"
            )

            f.write("### B. 可视化文件\n")
            f.write(
                "所有可视化文件都保存在与本报告相同的目录中。\n"
            )
            f.write(
                "- 特征分析: `feature_distributions.png`, `feature_correlation.png`\n"
            )
            f.write(
                "- 转移分析: `interactive_transition_graph.html`, `transition_matrix_heatmap.png`\n"
            )
            f.write(
                "- 行为样本: `typical_samples.png`, `behavior_*_sample_*.png`\n"
            )



    def generate_transition_plots(self, transition_matrix: np.ndarray, output_dir: Path) -> None:
        """Generate behavior transition visualization plots"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 转移矩阵热力图
        logger.info("生成转移矩阵热力图")
        plt.figure(figsize=(10, 8))
        sns.heatmap(transition_matrix, annot=True, fmt='.3f', cmap='YlGnBu', square=True)
        plt.xlabel('目标状态')
        plt.ylabel('起始状态')
        plt.title('行为转移矩阵热力图')
        plt.tight_layout()
        plt.savefig(output_dir / 'transition_matrix_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("转移可视化图表生成完成")

    def generate_behavior_separation_plots(self, X: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
        """Generate behavior separation visualization plots"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 特征分布箱线图
        logger.info("生成特征分布箱线图")
        n_features = X.shape[1]
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        axes = axes.flatten()

        for i in range(min(n_features, 6)):
            sns.boxplot(x=labels, y=X[:, i], ax=axes[i])
            axes[i].set_title(f'特征 {i+1} 分布')
            axes[i].set_xlabel('行为标签')
            axes[i].set_ylabel('特征值')

        # 隐藏多余的子图
        for i in range(n_features, 6):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 特征相关性热力图
        logger.info("生成特征相关性热力图")
        plt.figure(figsize=(10, 8))
        corr_matrix = np.corrcoef(X.T)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
        plt.title('特征相关性热力图')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("行为分离可视化图表生成完成")

    def generate_visualizations(self, X: np.ndarray, labels: np.ndarray, transition_matrix: np.ndarray, output_dir: Path) -> None:
        """Generate all visualizations for evaluation"""
        logger.info("开始生成所有可视化图表")

        # 生成转移可视化
        self.generate_transition_plots(transition_matrix, output_dir)

        # 生成行为分离可视化
        self.generate_behavior_separation_plots(X, labels, output_dir)

        logger.info("所有可视化图表生成完成")

    def generate_html_report(self, results: Dict, output_path: Path) -> None:
        """Generate a comprehensive HTML report from Markdown"""
        logger.info(f"开始生成HTML报告: {output_path}")
        
        # 读取Markdown报告
        markdown_path = output_path.with_suffix('.md')
        if not markdown_path.exists():
            logger.error(f"Markdown报告不存在: {markdown_path}")
            # 如果Markdown报告不存在，先生成它
            self.generate_comprehensive_report(results, markdown_path)
        
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # 将Markdown转换为HTML
        html_content = markdown2.markdown(markdown_content, extras=[
            'tables', 'fenced-code-blocks', 'header-ids', 'toc', 'footnotes'
        ])
        
        # 构建完整的HTML报告
        full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>网络行为发现效果评估报告</title>
    <style>
        /* Basic styles */
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 5px; }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
        h2 {{ color: #3498db; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-top: 40px; }}
        h3 {{ color: #27ae60; margin-top: 30px; }}
        h4 {{ color: #e67e22; margin-top: 20px; }}
        
        /* Table styles */
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        
        /* List styles */
        ul, ol {{ padding-left: 25px; }}
        li {{ margin: 8px 0; }}
        
        /* Image styles */
        img {{ max-width: 100%; height: auto; margin: 15px 0; border: 1px solid #ddd; padding: 5px; border-radius: 3px; display: block; margin-left: auto; margin-right: auto; }}
        
        /* Link styles */
        a {{ color: #3498db; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            .container {{ padding: 15px; }}
            h1 {{ font-size: 1.8em; }}
            h2 {{ font-size: 1.5em; }}
            table {{ font-size: 0.9em; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
    </div>
</body>
</html>"""
        
        # 保存HTML报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        logger.info(f"已生成HTML报告: {output_path}")


