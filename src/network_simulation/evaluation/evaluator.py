#!/usr/bin/env python3
"""
评估模块
负责评估生成的网络模拟数据的质量
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import entropy


from sklearn.decomposition import PCA
from typing import Dict
import matplotlib.pyplot as plt
from network_simulation.utils.logger import get_logger

# 设置中文显示
plt.rcParams["font.sans-serif"] = [
    "WenQuanYi Zen Hei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 初始化日志记录器
logger = get_logger(__name__)


class Evaluator:
    """评估器类

    该类用于评估生成的网络模拟数据的质量，包括统计特征、分布一致性、时序特性等多个维度。
    """

    def __init__(self):
        self.time_granularity = 0.1  # 100ms
        # 同时支持单通道和上下行特征
        self.feature_columns = {
            "single": [
                "feat_delay_std",
                "feat_loss_burst_ratio",
                "feat_burst_duration",
                "feat_burst_intensity",
                "feat_delay_trend",
                "feat_delay_acf_5",
            ],
            "up": [
                "feat_delay1_std",
                "feat_loss1_nonzero_ratio",
                "feat_max_consec_loss1",
                "feat_max_congestion_run1",
                "feat_delay1_trend",
                "feat_delay1_acf_5",
            ],
            "down": [
                "feat_delay2_std",
                "feat_loss2_nonzero_ratio",
                "feat_max_consec_loss2",
                "feat_max_congestion_run2",
                "feat_delay2_trend",
                "feat_delay2_acf_5",
            ]
        }

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """加载生成的模拟数据

        从CSV文件中加载生成的网络模拟数据，并解析时间戳字段。

        Args:
            file_path: 模拟数据CSV文件路径

        Returns:
            包含模拟数据的DataFrame，其中timestamp字段已解析为日期时间类型
        """
        return pd.read_csv(file_path, parse_dates=["timestamp"])

    def evaluate(self, df: pd.DataFrame) -> Dict:
        """评估生成的网络模拟数据质量

        该函数从多个维度评估生成的网络模拟数据质量，包括统计保真度、不可区分性和动态合理性。

        Args:
            df: 包含网络模拟数据的DataFrame，应包含时延、丢包率等字段

        Returns:
            包含各个维度评估结果的字典，包括：
            - statistical_fidelity: 统计保真度评估结果
            - indistinguishability: 不可区分性评估结果
            - dynamic_rationality: 动态合理性评估结果
        """
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
        """评估行为转移质量

        评估网络行为模式之间的转移质量，包括计算转移熵和转移稀疏度等指标。

        Args:
            transition_matrix: 行为转移矩阵

        Returns:
            包含转移质量评估结果的字典，包括平均转移熵、转移稀疏度等指标
        """
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
        """评估行为分离质量

        评估不同网络行为模式之间的分离质量，包括计算类内距离和类间距离等指标。

        Args:
            X: 特征向量数组
            labels: 行为标签数组

        Returns:
            包含行为分离质量评估结果的字典，包括类内距离、类间距离等指标
        """
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
        """计算行为相似性指标

        计算不同网络行为模式之间的相似性指标，包括统计特征的相关性等。

        Args:
            behavior_stats: 包含各行为统计特征的字典

        Returns:
            包含行为相似性评估结果的字典
        """
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
        """计算行为分离指标

        计算不同网络行为模式之间的分离指标，包括类内距离、类间距离、轮廓系数等。

        Args:
            X: 特征向量数组
            labels: 行为标签数组

        Returns:
            包含行为分离指标的字典
        """
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
        """评估生成数据的统计保真度

        评估生成数据与真实数据的统计特征一致性，包括均值、方差、分布等。

        Args:
            df: 包含生成数据的DataFrame

        Returns:
            包含统计保真度评估结果的字典
        """
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

    def calculate_stats(self, df: pd.DataFrame, prefix: str) -> Dict:
        """计算单个数据框的统计指标

        Args:
            df: 数据框
            prefix: 结果键名的前缀

        Returns:
            包含统计指标的字典
        """
        from scipy import stats

        stats_dict = {
            f"{prefix}_mean": df["delay"].mean(),
            f"{prefix}_std": df["delay"].std(),
            f"{prefix}_min": df["delay"].min(),
            f"{prefix}_max": df["delay"].max(),
            f"{prefix}_median": df["delay"].median(),
            f"{prefix}_skew": stats.skew(df["delay"]),
            f"{prefix}_kurtosis": stats.kurtosis(df["delay"]),
        }

        # 丢包率统计
        stats_dict[f"{prefix}_loss_mean"] = df["loss_rate"].mean()
        stats_dict[f"{prefix}_loss_std"] = df["loss_rate"].std()
        stats_dict[f"{prefix}_loss_unique"] = len(df["loss_rate"].unique())

        return stats_dict

    def evaluate_single_sample(self, original_file: Path, generated_file: Path, output_dir: Path) -> Dict:
        """评估单组生成样本的质量

        Args:
            original_file: 原始样本文件路径
            generated_file: 生成样本文件路径
            output_dir: 输出目录路径

        Returns:
            包含评估结果的字典
        """
        from scipy import stats
        import pandas as pd
        import os

        # 读取数据
        original_df = pd.read_csv(original_file)
        generated_df = pd.read_csv(generated_file)

        # 计算原始数据和生成数据的统计指标
        original_stats = self.calculate_stats(original_df, "original")
        generated_stats = self.calculate_stats(generated_df, "generated")

        # 合并统计结果
        stats_df = pd.DataFrame([original_stats, generated_stats])

        # 计算差异
        stats_df["delay_mean_diff"] = abs(
            stats_df["original_mean"] - stats_df["generated_mean"]
        )
        stats_df["delay_std_diff"] = abs(
            stats_df["original_std"] - stats_df["generated_std"]
        )
        stats_df["loss_mean_diff"] = abs(
            stats_df["original_loss_mean"] - stats_df["generated_loss_mean"]
        )

        # 生成评估报告
        report = []
        report.append("=" * 60)
        report.append(f"生成数据质量评估报告 - {os.path.basename(generated_file)}")
        report.append("=" * 60)
        report.append("")

        report.append("1. 时延统计对比:")
        report.append(
            f"   原始数据 - 平均值: {original_stats['original_mean']:.2f}, 标准差: {original_stats['original_std']:.2f}"
        )
        report.append(
            f"   生成数据 - 平均值: {generated_stats['generated_mean']:.2f}, 标准差: {generated_stats['generated_std']:.2f}"
        )
        report.append(
            f"   平均值差异: {abs(original_stats['original_mean'] - generated_stats['generated_mean']):.2f} ({abs(original_stats['original_mean'] - generated_stats['generated_mean']) / original_stats['original_mean'] * 100:.2f}%)"
        )
        report.append(
            f"   标准差差异: {abs(original_stats['original_std'] - generated_stats['generated_std']):.2f} ({abs(original_stats['original_std'] - generated_stats['generated_std']) / original_stats['original_std'] * 100:.2f}%)"
        )

        report.append("")
        report.append("2. 丢包率统计对比:")
        report.append(
            f"   原始数据 - 平均值: {original_stats['original_loss_mean']:.4f}, 标准差: {original_stats['original_loss_std']:.4f}"
        )
        report.append(
            f"   生成数据 - 平均值: {generated_stats['generated_loss_mean']:.4f}, 标准差: {generated_stats['generated_loss_std']:.4f}"
        )
        report.append(f"   原始数据丢包率唯一值: {original_stats['original_loss_unique']}")
        report.append(
            f"   生成数据丢包率唯一值: {generated_stats['generated_loss_unique']}"
        )

        report.append("")
        report.append("3. 分布相似性评估:")
        # KS检验
        delay_ks_stat, delay_ks_pvalue = stats.ks_2samp(
            original_df["delay"], generated_df["delay"]
        )
        report.append(
            f"   时延KS检验 - 统计量: {delay_ks_stat:.4f}, p值: {delay_ks_pvalue:.4f}"
        )
        if delay_ks_pvalue > 0.05:
            report.append("   结论: 时延分布相似 (p > 0.05)")
        else:
            report.append("   结论: 时延分布存在显著差异 (p ≤ 0.05)")

        # 生成数据有效性检查
        report.append("")
        report.append("4. 生成数据有效性检查:")
        invalid_delay = (generated_df["delay"] < 0).sum()
        invalid_loss = (
            (generated_df["loss_rate"] < 0) | (generated_df["loss_rate"] > 1)
        ).sum()
        report.append(f"   无效时延值数量: {invalid_delay}")
        report.append(f"   无效丢包率值数量: {invalid_loss}")
        report.append(
            f"   数据有效性: {'100%' if invalid_delay == 0 and invalid_loss == 0 else f'{(1 - (invalid_delay + invalid_loss) / len(generated_df)) * 100:.2f}%'}"
        )

        report.append("")
        report.append("5. 生成数据多样性评估:")
        # 计算生成数据的多样性指标
        delay_range = generated_df["delay"].max() - generated_df["delay"].min()
        loss_rate_values = sorted(generated_df["loss_rate"].unique())
        report.append(f"   时延范围: {delay_range:.2f}")
        report.append(f"   丢包率覆盖值: {loss_rate_values}")
        report.append(f"   丢包率覆盖类别数: {len(loss_rate_values)}")

        report.append("")
        report.append("=" * 60)
        report.append("评估完成!")
        report.append("=" * 60)

        # 记录评估报告
        report_text = "\n".join(report)
        logger.info(report_text)

        # 保存评估报告
        group_name = (
            os.path.basename(generated_file).split(".")[0].split("_")[-3:]
        )  # 获取group_1_behavior_1
        group_dir_name = "_" + "_".join(group_name)
        group_output_dir = output_dir / f"group{group_dir_name}"
        group_output_dir.mkdir(parents=True, exist_ok=True)

        report_file = group_output_dir / "evaluation_report.txt"
        with open(report_file, "w") as f:
            f.write(report_text)

        return {
            "group": group_name,
            "original_stats": original_stats,
            "generated_stats": generated_stats,
            "delay_ks_stat": delay_ks_stat,
            "delay_ks_pvalue": delay_ks_pvalue,
            "invalid_delay": invalid_delay,
            "invalid_loss": invalid_loss,
            "generated_file": generated_file,
        }

    def generate_summary_report(self, results: list, output_dir: Path) -> None:
        """生成所有样本的汇总评估报告

        Args:
            results: 评估结果列表
            output_dir: 输出目录路径
        """
        import numpy as np
        import pandas as pd
        import os

        report = []
        report.append("=" * 60)
        report.append("生成数据质量评估汇总报告")
        report.append("=" * 60)
        report.append("")

        # 计算平均指标
        if results:
            avg_delay_mean_diff = np.mean(
                [
                    abs(
                        r["generated_stats"]["generated_mean"]
                        - r["original_stats"]["original_mean"]
                    )
                    for r in results
                ]
            )
            avg_delay_std_diff = np.mean(
                [
                    abs(
                        r["generated_stats"]["generated_std"]
                        - r["original_stats"]["original_std"]
                    )
                    for r in results
                ]
            )
            avg_loss_mean_diff = np.mean(
                [
                    abs(
                        r["generated_stats"]["generated_loss_mean"]
                        - r["original_stats"]["original_loss_mean"]
                    )
                    for r in results
                ]
            )
            avg_ks_stat = np.mean([r["delay_ks_stat"] for r in results])
            avg_ks_pvalue = np.mean([r["delay_ks_pvalue"] for r in results])

            report.append("1. 平均指标")
            report.append(f"   平均时延平均值差异: {avg_delay_mean_diff:.2f}")
            report.append(f"   平均时延标准差差异: {avg_delay_std_diff:.2f}")
            report.append(f"   平均丢包率平均值差异: {avg_loss_mean_diff:.4f}")
            report.append(f"   平均KS检验统计量: {avg_ks_stat:.4f}")
            report.append(f"   平均KS检验p值: {avg_ks_pvalue:.4f}")
            report.append("")

        report.append("2. 样本评估详情:")
        for i, result in enumerate(results):
            report.append(f"   样本组 {i + 1}:")
            report.append(f"     文件名: {os.path.basename(result['generated_file'])}")
            report.append(f"     主要行为类别: {result['group'][-1]}")
            report.append(
                f"     时延平均值差异: {abs(result['original_stats']['original_mean'] - result['generated_stats']['generated_mean']):.2f}"
            )
            report.append(
                f"     丢包率平均值差异: {abs(result['original_stats']['original_loss_mean'] - result['generated_stats']['generated_loss_mean']):.4f}"
            )
            report.append(f"     KS检验p值: {result['delay_ks_pvalue']:.4f}")
            # 计算数据有效性
            generated_df = pd.read_csv(result["generated_file"])
            total_samples = len(generated_df)
            data_validity = (
                1 - (result["invalid_delay"] + result["invalid_loss"]) / total_samples
            ) * 100
            report.append(f"     数据有效性: {data_validity:.2f}%")

        report.append("")
        report.append("=" * 60)
        report.append("汇总评估完成!")
        report.append("=" * 60)

        # 保存汇总报告
        summary_report_file = output_dir / "summary_report.txt"
        with open(summary_report_file, "w") as f:
            f.write("\n".join(report))
        logger.info(f"汇总报告已保存到: {summary_report_file}")

        # 生成综合评估报告
        self._generate_comprehensive_report(results, output_dir)

    def _generate_comprehensive_report(self, results: list, output_dir: Path) -> None:
        """生成综合评估报告

        Args:
            results: 评估结果列表
            output_dir: 输出目录路径
        """
        import numpy as np
        import pandas as pd

        # 生成综合评估报告所需的结果字典
        comprehensive_results = {}

        # 如果有结果，使用第一个结果的统计信息作为代表
        if results:
            first_result = results[0]

            # 转换numpy类型为Python原生类型的辅助函数
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                else:
                    return obj

            # 转换原始统计信息和生成统计信息
            original_stats = convert_numpy_types(first_result["original_stats"])
            generated_stats = convert_numpy_types(first_result["generated_stats"])

            comprehensive_results["original_stats"] = original_stats
            comprehensive_results["generated_stats"] = generated_stats
            comprehensive_results["delay_ks_stat"] = float(first_result["delay_ks_stat"])
            comprehensive_results["delay_ks_pvalue"] = float(first_result["delay_ks_pvalue"])
            comprehensive_results["invalid_delay"] = int(first_result["invalid_delay"])
            comprehensive_results["invalid_loss"] = int(first_result["invalid_loss"])
            comprehensive_results["total_samples"] = int(len(pd.read_csv(first_result["generated_file"])))

        # 调用save()方法，生成综合评估报告
        self.save(comprehensive_results, output_dir)

    def evaluate_batch_samples(self, input_generation_dir: Path, output_evaluation_dir: Path) -> None:
        """批量评估生成样本的质量

        Args:
            input_generation_dir: 生成样本的目录路径
            output_evaluation_dir: 评估结果的输出目录路径
        """
        import glob

        # 确保输出目录存在
        output_evaluation_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"确保输出目录存在: {output_evaluation_dir}")

        # 查找所有原始样本和生成样本文件
        original_files = sorted(
            [Path(f) for f in glob.glob(str(input_generation_dir / "original_sample_6000*.csv"))]
        )
        generated_files = sorted(
            [Path(f) for f in glob.glob(str(input_generation_dir / "generated_sample_6000*.csv"))]
        )
        logger.info(f"找到原始样本文件数: {len(original_files)}, 生成样本文件数: {len(generated_files)}")

        if not original_files:
            logger.error(f"错误: 在 {input_generation_dir} 中未找到原始样本文件")
            return

        if not generated_files:
            logger.error(f"错误: 在 {input_generation_dir} 中未找到生成样本文件")
            return

        if len(original_files) != len(generated_files):
            logger.warning(
                f"警告: 原始样本文件数 ({len(original_files)}) 与生成样本文件数 ({len(generated_files)}) 不匹配"
            )

        # 评估所有样本对
        all_results = []
        for original_file, generated_file in zip(original_files, generated_files):
            logger.info(f"开始评估样本对: {original_file.name} 和 {generated_file.name}")
            result = self.evaluate_single_sample(
                original_file, generated_file, output_evaluation_dir
            )
            all_results.append(result)

        # 生成汇总报告
        self.generate_summary_report(all_results, output_evaluation_dir)

    def _evaluate_indistinguishability(self, _df: pd.DataFrame) -> Dict:
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
        logger.info("所有评估结果已保存完成")

    def save_with_visualization(self, results: Dict, output_dir: Path, X: np.ndarray = None, labels: np.ndarray = None, transition_matrix: np.ndarray = None) -> None:
        """Save evaluation results with visualization"""
        # Save evaluation results first
        self.save(results, output_dir)

        # Import visualization module here to avoid circular dependency
        from network_simulation.visualization.visualizer import Visualizer

        # Initialize visualizer
        visualizer = Visualizer(output_dir)

        # Generate all visualizations if required data is provided
        if X is not None and labels is not None and transition_matrix is not None:
            visualizer.generate_all_visualizations(X, labels, transition_matrix, output_dir / "plots")

        # Generate evaluation report
        visualizer.generate_evaluation_report(results, output_dir)

        logger.info("评估结果和可视化已保存完成")




