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
        """初始化评估器

        初始化评估器，设置时间粒度和支持的特征列。

        Examples:
            >>> from network_simulation.evaluation.evaluator import Evaluator
            >>> evaluator = Evaluator()
        """
        self.time_granularity = 0.1  # 100ms
        # 只支持上下行特征，不支持单通道
        self.feature_columns = [
            "feat_delay1_std",
            "feat_loss1_nonzero_ratio",
            "feat_max_consec_loss1",
            "feat_max_congestion_run1",
            "feat_delay1_trend",
            "feat_delay1_acf_5",
            "feat_delay2_std",
            "feat_loss2_nonzero_ratio",
            "feat_max_consec_loss2",
            "feat_max_congestion_run2",
            "feat_delay2_trend",
            "feat_delay2_acf_5",
        ]

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """加载生成的模拟数据

        从CSV文件中加载生成的网络模拟数据，并解析时间戳字段。

        Args:
            file_path (Path): 模拟数据CSV文件路径

        Returns:
            pd.DataFrame: 包含模拟数据的DataFrame，其中timestamp字段已解析为日期时间类型

        Examples:
            >>> from network_simulation.evaluation.evaluator import Evaluator
            >>> from pathlib import Path
            >>> evaluator = Evaluator()
            >>> # 假设存在一个生成的模拟数据文件
            >>> # df = evaluator.load_data(Path("generated_data.csv"))
            >>> # print(df.head())
        """
        return pd.read_csv(file_path, parse_dates=["timestamp"])

    def evaluate(self, df: pd.DataFrame) -> Dict:
        """评估生成的网络模拟数据质量

        该函数从多个维度评估生成的网络模拟数据质量，包括统计保真度、不可区分性和动态合理性。

        Args:
            df (pd.DataFrame): 包含网络模拟数据的DataFrame，应包含时延、丢包率等字段
                - 预期列名：delay1, delay2, loss_rate1, loss_rate2, timestamp

        Returns:
            Dict: 包含各个维度评估结果的字典
                - statistical_fidelity: 统计保真度评估结果，包括均值、方差等统计特征
                - indistinguishability: 不可区分性评估结果，包括判别器AUC等指标
                - dynamic_rationality: 动态合理性评估结果，包括自相关、突发特性等指标

        Examples:
            >>> from network_simulation.evaluation.evaluator import Evaluator
            >>> import pandas as pd
            >>> import numpy as np
            >>> evaluator = Evaluator()
            >>> # 生成示例数据
            >>> n_samples = 1000
            >>> df = pd.DataFrame({
            ...     'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='100ms'),
            ...     'delay1': np.random.normal(50, 10, n_samples),
            ...     'delay2': np.random.normal(60, 15, n_samples),
            ...     'loss_rate1': np.random.choice([0.0, 0.5, 1.0], n_samples),
            ...     'loss_rate2': np.random.choice([0.0, 0.5, 1.0], n_samples)
            ... })
            >>> # 评估数据质量
            >>> results = evaluator.evaluate(df)
            >>> print(f"评估结果包含的维度: {list(results.keys())}")
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
            transition_matrix (np.ndarray): 行为转移矩阵，shape (n_behaviors, n_behaviors)
                - 每个元素表示从行为i转移到行为j的概率

        Returns:
            Dict: 包含转移质量评估结果的字典
                - average_transition_entropy: 平均转移熵，衡量转移的不确定性
                - transition_sparsity: 转移稀疏度，衡量转移矩阵的稀疏程度

        Examples:
            >>> from network_simulation.evaluation.evaluator import Evaluator
            >>> import numpy as np
            >>> evaluator = Evaluator()
            >>> # 创建一个简单的转移矩阵
            >>> transition_matrix = np.array([
            ...     [0.7, 0.2, 0.1],
            ...     [0.3, 0.5, 0.2],
            ...     [0.2, 0.3, 0.5]
            ... ])
            >>> # 评估转移质量
            >>> results = evaluator.evaluate_transition_quality(transition_matrix)
            >>> print(f"平均转移熵: {results['average_transition_entropy']:.4f}")
            >>> print(f"转移稀疏度: {results['transition_sparsity']:.4f}")
        """
        # 计算平均转移熵
        transition_entropies = []
        for i in range(len(transition_matrix)):
            # 只考虑非零概率
            row = transition_matrix[i][transition_matrix[i] > 0]
            if len(row) > 0:
                transition_entropies.append(entropy(row, base=2))

        avg_transition_entropy = (
            np.mean(transition_entropies) if transition_entropies else 0.0
        )

        # 计算转移稀疏度
        total_elements = transition_matrix.size
        non_zero_elements = np.count_nonzero(transition_matrix)
        transition_sparsity = non_zero_elements / total_elements

        return {
            "average_transition_entropy": float(avg_transition_entropy),
            "transition_sparsity": float(transition_sparsity),
        }

    def evaluate_behavior_separation(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """评估行为分离质量

        评估不同网络行为模式之间的分离质量，包括计算类内距离、类间距离和行为相似性等指标。

        Args:
            X (np.ndarray): 特征向量数组，shape (n_samples, n_features)
                - 每个样本的特征向量
            labels (np.ndarray): 行为标签数组，shape (n_samples,)
                - 每个样本的行为标签

        Returns:
            Dict: 包含行为分离质量评估结果的字典
                - behavior_stats: 各行为的统计特征，包括均值、标准差和样本数量
                - separation_metrics: 分离指标，包括类间距离、类内距离和分离指数
                - behavior_similarity: 行为相似性指标，包括相似性矩阵和平均相似性

        Examples:
            >>> from network_simulation.evaluation.evaluator import Evaluator
            >>> import numpy as np
            >>> evaluator = Evaluator()
            >>> # 创建示例特征和标签
            >>> X = np.vstack([
            ...     np.random.normal(0, 1, (50, 12)),  # 行为0
            ...     np.random.normal(5, 1, (50, 12)),  # 行为1
            ...     np.random.normal(10, 1, (50, 12))   # 行为2
            ... ])
            >>> labels = np.concatenate([
            ...     np.zeros(50),
            ...     np.ones(50),
            ...     np.ones(50) * 2
            ... ])
            >>> # 评估行为分离质量
            >>> results = evaluator.evaluate_behavior_separation(X, labels)
            >>> print(f"分离指数: {results['separation_metrics']['separation_index']:.4f}")
            >>> print(f"平均行为相似性: {results['behavior_similarity']['average_similarity']:.4f}")
        """
        unique_labels = np.unique(labels)

        # 计算每个行为的均值和标准差
        behavior_stats = {}
        for label in unique_labels:
            cluster_data = X[labels == label]
            behavior_stats[str(label)] = {
                "mean": cluster_data.mean(axis=0).tolist(),
                "std": cluster_data.std(axis=0).tolist(),
                "count": int(len(cluster_data)),
            }

        # 计算分离指标
        separation_metrics = self._calculate_separation_metrics(X, labels)

        # 计算行为相似性
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

        # 初始化相似性矩阵
        similarity_matrix = np.zeros((n_behaviors, n_behaviors))

        # 计算每对行为之间的相似性
        for i in range(n_behaviors):
            for j in range(n_behaviors):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # 与自身的完全相似性
                else:
                    # 计算行为均值之间的余弦相似性
                    mean_i = np.array(behavior_stats[behavior_ids[i]]["mean"])
                    mean_j = np.array(behavior_stats[behavior_ids[j]]["mean"])

                    # 余弦相似性
                    cosine_sim = np.dot(mean_i, mean_j) / (
                        np.linalg.norm(mean_i) * np.linalg.norm(mean_j)
                    )
                    similarity_matrix[i, j] = float(cosine_sim)

        # 计算平均相似性
        avg_similarity = np.mean(similarity_matrix[np.triu_indices(n_behaviors, k=1)])

        # 计算最小和最大相似性
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

        # 计算用于分离可视化的PCA
        pca = PCA(n_components=2)
        pca.fit_transform(X)

        # 计算簇间距离
        cluster_centers = []
        for label in unique_labels:
            cluster_data = X[labels == label]
            cluster_centers.append(cluster_data.mean(axis=0))
        cluster_centers = np.array(cluster_centers)

        # 计算平均簇间距离
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

        # 处理双流数据：delay1, delay2, loss_rate1, loss_rate2
        delay1 = df["delay1"].values
        delay2 = df["delay2"].values
        loss_rate1 = df["loss_rate1"].values
        loss_rate2 = df["loss_rate2"].values

        # 处理空数据情况
        if len(delay1) == 0 or len(loss_rate1) == 0:
            return {
                "delay1_mean": 0.0,
                "delay1_std": 0.0,
                "delay2_mean": 0.0,
                "delay2_std": 0.0,
                "loss_rate1_mean": 0.0,
                "loss_rate1_std": 0.0,
                "loss_rate2_mean": 0.0,
                "loss_rate2_std": 0.0,
                "loss_rate1_range": {
                    "min": 0.0,
                    "max": 0.0,
                },
                "loss_rate2_range": {
                    "min": 0.0,
                    "max": 0.0,
                },
                "wasserstein_distance": 0.0,  # 与真实数据比较的占位符
            }

        results = {
            "delay1_mean": float(np.mean(delay1)),
            "delay1_std": float(np.std(delay1)),
            "delay2_mean": float(np.mean(delay2)),
            "delay2_std": float(np.std(delay2)),
            "loss_rate1_mean": float(np.mean(loss_rate1)),
            "loss_rate1_std": float(np.std(loss_rate1)),
            "loss_rate2_mean": float(np.mean(loss_rate2)),
            "loss_rate2_std": float(np.std(loss_rate2)),
            "loss_rate1_range": {
                "min": float(np.min(loss_rate1)),
                "max": float(np.max(loss_rate1)),
            },
            "loss_rate2_range": {
                "min": float(np.min(loss_rate2)),
                "max": float(np.max(loss_rate2)),
            },
            "wasserstein_distance": 0.0,  # 与真实数据比较的占位符
        }

        return results

    def calculate_stats(self, df: pd.DataFrame, prefix: str) -> Dict:
        """计算单个数据框的统计指标

        计算数据框的统计指标，包括均值、标准差、最小值、最大值、中位数、偏度和峰度等。

        Args:
            df (pd.DataFrame): 包含网络模拟数据的数据框
                - 支持单通道数据（列名：delay, loss_rate）
                - 支持双通道数据（列名：delay1, loss_rate1, delay2, loss_rate2）
            prefix (str): 结果键名的前缀，用于区分不同数据来源

        Returns:
            Dict: 包含统计指标的字典，键名以指定前缀开头

        Examples:
            >>> from network_simulation.evaluation.evaluator import Evaluator
            >>> import pandas as pd
            >>> import numpy as np
            >>> evaluator = Evaluator()
            >>> # 生成示例双通道数据
            >>> n_samples = 1000
            >>> df = pd.DataFrame({
            ...     'delay1': np.random.normal(50, 10, n_samples),
            ...     'loss_rate1': np.random.choice([0.0, 0.5, 1.0], n_samples),
            ...     'delay2': np.random.normal(60, 15, n_samples),
            ...     'loss_rate2': np.random.choice([0.0, 0.5, 1.0], n_samples)
            ... })
            >>> # 计算统计指标
            >>> stats = evaluator.calculate_stats(df, 'test')
            >>> print(f"统计指标键名: {list(stats.keys())}")
            >>> print(f"上行延迟平均值: {stats['test_up_delay_mean']:.2f}")
        """
        from scipy import stats

        stats_dict = {}

        # 处理上下行数据
        for direction, delay_col, loss_col in [
            ("up", "delay1", "loss_rate1"),
            ("down", "delay2", "loss_rate2"),
        ]:
            stats_dict.update(
                {
                    f"{prefix}_{direction}_delay_mean": df[delay_col].mean(),
                    f"{prefix}_{direction}_delay_std": df[delay_col].std(),
                    f"{prefix}_{direction}_delay_min": df[delay_col].min(),
                    f"{prefix}_{direction}_delay_max": df[delay_col].max(),
                    f"{prefix}_{direction}_delay_median": df[delay_col].median(),
                    f"{prefix}_{direction}_delay_skew": stats.skew(df[delay_col]),
                    f"{prefix}_{direction}_delay_kurtosis": stats.kurtosis(
                        df[delay_col]
                    ),
                    f"{prefix}_{direction}_loss_mean": df[loss_col].mean(),
                    f"{prefix}_{direction}_loss_std": df[loss_col].std(),
                    f"{prefix}_{direction}_loss_unique": len(df[loss_col].unique()),
                }
            )

        return stats_dict

    def evaluate_single_sample(
        self, original_file: Path, generated_file: Path, output_dir: Path
    ) -> Dict:
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

        report = []
        report.append("=" * 60)
        report.append(f"生成数据质量评估报告 - {os.path.basename(generated_file)}")
        report.append("=" * 60)
        report.append("")

        # 初始化KS检验结果字典
        ks_results = {}

        # 处理上下行数据
        for direction in ["up", "down"]:
            original_delay_mean = original_stats[f"original_{direction}_delay_mean"]
            generated_delay_mean = generated_stats[f"generated_{direction}_delay_mean"]
            original_delay_std = original_stats[f"original_{direction}_delay_std"]
            generated_delay_std = generated_stats[f"generated_{direction}_delay_std"]
            original_loss_mean = original_stats[f"original_{direction}_loss_mean"]
            generated_loss_mean = generated_stats[f"generated_{direction}_loss_mean"]
            original_loss_unique = original_stats[f"original_{direction}_loss_unique"]
            generated_loss_unique = generated_stats[
                f"generated_{direction}_loss_unique"
            ]

            report.append(f"1. {direction}行时延统计对比:")
            report.append(
                f"   原始数据 - 平均值: {original_delay_mean:.2f}, 标准差: {original_delay_std:.2f}"
            )
            report.append(
                f"   生成数据 - 平均值: {generated_delay_mean:.2f}, 标准差: {generated_delay_std:.2f}"
            )
            report.append(
                f"   平均值差异: {abs(original_delay_mean - generated_delay_mean):.2f} ({abs(original_delay_mean - generated_delay_mean) / original_delay_mean * 100:.2f}%)"
            )
            report.append(
                f"   标准差差异: {abs(original_delay_std - generated_delay_std):.2f} ({abs(original_delay_std - generated_delay_std) / original_delay_std * 100:.2f}%)"
            )

            report.append("")
            report.append(f"2. {direction}行丢包率统计对比:")
            report.append(
                f"   原始数据 - 平均值: {original_loss_mean:.4f}, 标准差: {original_stats[f'original_{direction}_loss_std']:.4f}"
            )
            report.append(
                f"   生成数据 - 平均值: {generated_loss_mean:.4f}, 标准差: {generated_stats[f'generated_{direction}_loss_std']:.4f}"
            )
            report.append(f"   原始数据丢包率唯一值: {original_loss_unique}")
            report.append(f"   生成数据丢包率唯一值: {generated_loss_unique}")

            report.append("")
            report.append(f"3. {direction}行分布相似性评估:")
            # KS检验
            delay_ks_stat, delay_ks_pvalue = stats.ks_2samp(
                original_df[f"delay{1 if direction == 'up' else 2}"],
                generated_df[f"delay{1 if direction == 'up' else 2}"],
            )
            # 保存KS检验结果到字典
            ks_results[f"{direction}_delay_ks_stat"] = delay_ks_stat
            ks_results[f"{direction}_delay_ks_pvalue"] = delay_ks_pvalue

            report.append(
                f"   时延KS检验 - 统计量: {delay_ks_stat:.4f}, p值: {delay_ks_pvalue:.4f}"
            )
            if delay_ks_pvalue > 0.05:
                report.append(f"   结论: {direction}行时延分布相似 (p > 0.05)")
            else:
                report.append(f"   结论: {direction}行时延分布存在显著差异 (p ≤ 0.05)")

            report.append("")

        # 生成数据有效性检查
        report.append("")
        report.append("4. 生成数据有效性检查:")

        # 检查上下行数据的有效性
        invalid_delay1 = (generated_df["delay1"] < 0).sum()
        invalid_delay2 = (generated_df["delay2"] < 0).sum()
        invalid_loss1 = (
            (generated_df["loss_rate1"] < 0) | (generated_df["loss_rate1"] > 1)
        ).sum()
        invalid_loss2 = (
            (generated_df["loss_rate2"] < 0) | (generated_df["loss_rate2"] > 1)
        ).sum()
        invalid_delay = invalid_delay1 + invalid_delay2
        invalid_loss = invalid_loss1 + invalid_loss2
        report.append(f"   无效上行时延值数量: {invalid_delay1}")
        report.append(f"   无效下行时延值数量: {invalid_delay2}")
        report.append(f"   无效上行丢包率值数量: {invalid_loss1}")
        report.append(f"   无效下行丢包率值数量: {invalid_loss2}")

        report.append(
            f"   数据有效性: {'100%' if invalid_delay == 0 and invalid_loss == 0 else f'{(1 - (invalid_delay + invalid_loss) / len(generated_df)) * 100:.2f}%'}"
        )

        report.append("")
        report.append("5. 生成数据多样性评估:")
        # 计算生成数据的多样性指标
        # 计算上下行数据的多样性指标
        for direction, delay_col, loss_col in [
            ("上行", "delay1", "loss_rate1"),
            ("下行", "delay2", "loss_rate2"),
        ]:
            delay_range = generated_df[delay_col].max() - generated_df[delay_col].min()
            loss_rate_values = sorted(generated_df[loss_col].unique())
            report.append(f"   {direction}时延范围: {delay_range:.2f}")
            report.append(f"   {direction}丢包率覆盖值: {loss_rate_values}")
            report.append(f"   {direction}丢包率覆盖类别数: {len(loss_rate_values)}")
        else:
            # 计算单通道数据的多样性指标
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
            "up_delay_ks_stat": ks_results["up_delay_ks_stat"],
            "up_delay_ks_pvalue": ks_results["up_delay_ks_pvalue"],
            "down_delay_ks_stat": ks_results["down_delay_ks_stat"],
            "down_delay_ks_pvalue": ks_results["down_delay_ks_pvalue"],
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

        # 计算平均指标 - 针对双流数据
        if results:
            avg_up_delay_mean_diff = np.mean(
                [
                    abs(
                        r["generated_stats"]["generated_up_delay_mean"]
                        - r["original_stats"]["original_up_delay_mean"]
                    )
                    for r in results
                ]
            )
            avg_down_delay_mean_diff = np.mean(
                [
                    abs(
                        r["generated_stats"]["generated_down_delay_mean"]
                        - r["original_stats"]["original_down_delay_mean"]
                    )
                    for r in results
                ]
            )
            avg_up_loss_mean_diff = np.mean(
                [
                    abs(
                        r["generated_stats"]["generated_up_loss_mean"]
                        - r["original_stats"]["original_up_loss_mean"]
                    )
                    for r in results
                ]
            )
            avg_down_loss_mean_diff = np.mean(
                [
                    abs(
                        r["generated_stats"]["generated_down_loss_mean"]
                        - r["original_stats"]["original_down_loss_mean"]
                    )
                    for r in results
                ]
            )

            report.append("1. 平均指标")
            report.append(f"   上行平均时延平均值差异: {avg_up_delay_mean_diff:.2f}")
            report.append(f"   下行平均时延平均值差异: {avg_down_delay_mean_diff:.2f}")
            report.append(f"   上行平均丢包率平均值差异: {avg_up_loss_mean_diff:.4f}")
            report.append(f"   下行平均丢包率平均值差异: {avg_down_loss_mean_diff:.4f}")
            report.append("")

        report.append("2. 样本评估详情:")
        for i, result in enumerate(results):
            report.append(f"   样本组 {i + 1}:")
            report.append(f"     文件名: {os.path.basename(result['generated_file'])}")

            # 计算数据有效性
            generated_df = pd.read_csv(result["generated_file"])
            total_samples = len(generated_df)
            data_validity = (
                1 - (result["invalid_delay"] + result["invalid_loss"]) / total_samples
            ) * 100
            report.append(f"     数据有效性: {data_validity:.2f}%")

            # 显示上下行的统计差异
            report.append("     上行统计差异:")
            report.append(
                f"       时延平均值差异: {abs(result['original_stats']['original_up_delay_mean'] - result['generated_stats']['generated_up_delay_mean']):.2f}"
            )
            report.append(
                f"       丢包率平均值差异: {abs(result['original_stats']['original_up_loss_mean'] - result['generated_stats']['generated_up_loss_mean']):.4f}"
            )
            report.append("     下行统计差异:")
            report.append(
                f"       时延平均值差异: {abs(result['original_stats']['original_down_delay_mean'] - result['generated_stats']['generated_down_delay_mean']):.2f}"
            )
            report.append(
                f"       丢包率平均值差异: {abs(result['original_stats']['original_down_loss_mean'] - result['generated_stats']['generated_down_loss_mean']):.4f}"
            )

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
                    return {
                        key: convert_numpy_types(value) for key, value in obj.items()
                    }
                else:
                    return obj

            # 转换原始统计信息和生成统计信息
            original_stats = convert_numpy_types(first_result["original_stats"])
            generated_stats = convert_numpy_types(first_result["generated_stats"])

            comprehensive_results["original_stats"] = original_stats
            comprehensive_results["generated_stats"] = generated_stats
            comprehensive_results["up_delay_ks_stat"] = float(
                first_result["up_delay_ks_stat"]
            )
            comprehensive_results["up_delay_ks_pvalue"] = float(
                first_result["up_delay_ks_pvalue"]
            )
            comprehensive_results["down_delay_ks_stat"] = float(
                first_result["down_delay_ks_stat"]
            )
            comprehensive_results["down_delay_ks_pvalue"] = float(
                first_result["down_delay_ks_pvalue"]
            )
            comprehensive_results["invalid_delay"] = int(first_result["invalid_delay"])
            comprehensive_results["invalid_loss"] = int(first_result["invalid_loss"])
            comprehensive_results["total_samples"] = int(
                len(pd.read_csv(first_result["generated_file"]))
            )

        # 调用save()方法，生成综合评估报告
        self.save(comprehensive_results, output_dir)

    def evaluate_batch_samples(
        self, input_generation_dir: Path, output_evaluation_dir: Path
    ) -> None:
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
            [
                Path(f)
                for f in glob.glob(
                    str(input_generation_dir / "original_sample_6000*.csv")
                )
            ]
        )
        generated_files = sorted(
            [
                Path(f)
                for f in glob.glob(
                    str(input_generation_dir / "generated_sample_6000*.csv")
                )
            ]
        )
        logger.info(
            f"找到原始样本文件数: {len(original_files)}, 生成样本文件数: {len(generated_files)}"
        )

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
            logger.info(
                f"开始评估样本对: {original_file.name} 和 {generated_file.name}"
            )
            result = self.evaluate_single_sample(
                original_file, generated_file, output_evaluation_dir
            )
            all_results.append(result)

        # 生成汇总报告
        self.generate_summary_report(all_results, output_evaluation_dir)

    def _evaluate_indistinguishability(self, _df: pd.DataFrame) -> Dict:
        """评估生成的数据是否与真实数据无法区分"""
        # 目前，我们将使用占位值
        # 稍后，我们将训练一个判别器模型

        results = {
            "discriminator_auc": 0.5,  # 完美的不可区分性为0.5
            "tstr_keep_rate": 0.95,  # TSTR（在合成数据上训练，在真实数据上测试）率的占位符
        }

        return results

    def _evaluate_dynamic_rationality(self, df: pd.DataFrame) -> Dict:
        """评估生成数据的动态合理性"""
        # 处理双流数据：delay1, delay2, loss_rate1, loss_rate2
        delay1 = df["delay1"].values
        delay2 = df["delay2"].values
        loss_rate1 = df["loss_rate1"].values
        loss_rate2 = df["loss_rate2"].values

        # 计算两个流的自相关
        delay1_acf = self._calculate_acf(delay1, lag=5)
        delay2_acf = self._calculate_acf(delay2, lag=5)

        # 计算两个流的突发特性
        burst_stats1 = self._calculate_burst_statistics(loss_rate1)
        burst_stats2 = self._calculate_burst_statistics(loss_rate2)

        results = {
            "delay1_acf_5": float(delay1_acf),
            "delay2_acf_5": float(delay2_acf),
            "burst_statistics1": burst_stats1,
            "burst_statistics2": burst_stats2,
            "behavior_alignment_accuracy": 0.9,  # 行为对齐检查的占位符
        }

        return results

    def _calculate_acf(self, data: np.ndarray, lag: int) -> float:
        """计算指定滞后的自相关函数"""
        if len(data) < lag + 1:
            return 0.0

        # 归一化数据
        data_normalized = data - np.mean(data)

        # 计算自协方差
        numerator = np.sum(data_normalized[:-lag] * data_normalized[lag:])
        denominator = np.sum(data_normalized**2)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _calculate_burst_statistics(self, loss_rates: np.ndarray) -> Dict:
        """计算丢包率序列的突发统计量"""
        # 处理空数据情况
        if len(loss_rates) == 0:
            return {
                "num_bursts": 0,
                "avg_burst_duration": 0.0,
                "burst_frequency": 0.0,
            }

        # 识别突发时段（丢包率 > 0.1）
        burst_periods = loss_rates > 0.1

        # 计算突发数量
        num_bursts = 0
        in_burst = False
        for is_burst in burst_periods:
            if is_burst and not in_burst:
                num_bursts += 1
                in_burst = True
            elif not is_burst:
                in_burst = False

        # 计算平均突发持续时间
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
        """保存评估结果到目录"""
        logger.info(f"开始保存评估结果到目录: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存结果到JSON文件
        json_path = output_dir / "evaluation_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"已保存评估结果到JSON文件: {json_path}")
        logger.info("所有评估结果已保存完成")

    def save_with_visualization(
        self,
        results: Dict,
        output_dir: Path,
        X: np.ndarray = None,
        labels: np.ndarray = None,
        transition_matrix: np.ndarray = None,
    ) -> None:
        """保存带有可视化的评估结果"""
        # 先保存评估结果
        self.save(results, output_dir)

        # 在这里导入可视化模块以避免循环依赖
        from network_simulation.visualization.visualizer import Visualizer

        # 初始化可视化器
        visualizer = Visualizer(output_dir)

        # Generate all visualizations if required data is provided
        if X is not None and labels is not None and transition_matrix is not None:
            visualizer.generate_all_visualizations(
                X, labels, transition_matrix, output_dir / "plots"
            )

        # Generate evaluation report
        visualizer.generate_evaluation_report(results, output_dir)

        logger.info("评估结果和可视化已保存完成")
