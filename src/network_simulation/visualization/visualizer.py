#!/usr/bin/env python3
"""
网络行为可视化模块
负责可视化网络行为模式和评估结果
"""

import numpy as np
from pathlib import Path
from typing import Dict, List

from ..utils.logger import get_logger

from .base_visualizer import BaseVisualizer
from .feature_space_visualizer import FeatureSpaceVisualizer
from .behavior_visualizer import BehaviorVisualizer
from .report_generator import ReportGenerator


logger = get_logger(__name__)


class Visualizer:
    """主可视化类，整合所有可视化功能"""

    def __init__(self, output_dir: Path):
        """初始化主可视化类

        初始化主可视化类，整合所有可视化功能模块。

        Args:
            output_dir (Path): 可视化结果的输出目录路径

        Examples:
            >>> from network_simulation.visualization.visualizer import Visualizer
            >>> from pathlib import Path
            >>> visualizer = Visualizer(Path("output/visualizations"))
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_visualizer = BaseVisualizer(output_dir)
        self.feature_visualizer = FeatureSpaceVisualizer(output_dir)
        self.behavior_visualizer = BehaviorVisualizer(output_dir)
        self.report_generator = ReportGenerator(self.base_visualizer)
        # 从results_visualizer模块导入并初始化
        from .results_visualizer import ResultsVisualizer

        self.results_visualizer = ResultsVisualizer(output_dir)

    def visualize_evaluation_results(self, _results: Dict) -> str:
        """可视化评估结果并返回HTML片段

        可视化评估结果，生成包含图表和统计信息的HTML片段。

        Args:
            _results (Dict): 包含评估结果的字典

        Returns:
            str: 包含评估结果可视化的HTML片段

        Examples:
            >>> from network_simulation.visualization.visualizer import Visualizer
            >>> from pathlib import Path
            >>> visualizer = Visualizer(Path("output/visualizations"))
            >>> results = {"statistical_fidelity": {"delay1_mean": 50.0}}
            >>> html = visualizer.visualize_evaluation_results(results)
            >>> print(html)
        """
        # 后续将实现
        return "<h2>评估结果可视化</h2>"

    def visualize_pca_scatter(
        self, X: np.ndarray, labels: np.ndarray, method: str
    ) -> str:
        """可视化PCA散点图并返回HTML片段

        使用主成分分析(PCA)对特征空间进行降维，并可视化不同行为模式的分布。

        Args:
            X (np.ndarray): 特征矩阵，shape (n_samples, n_features)
            labels (np.ndarray): 行为标签数组，shape (n_samples,)
            method (str): 可视化方法名称，用于区分不同的可视化结果

        Returns:
            str: 包含PCA散点图的HTML片段

        Examples:
            >>> from network_simulation.visualization.visualizer import Visualizer
            >>> from pathlib import Path
            >>> import numpy as np
            >>> visualizer = Visualizer(Path("output/visualizations"))
            >>> X = np.random.randn(100, 12)
            >>> labels = np.random.randint(0, 3, 100)
            >>> html = visualizer.visualize_pca_scatter(X, labels, "pca")
            >>> print(html)
        """
        return self.feature_visualizer.visualize_pca_scatter(X, labels, method)

    def visualize_pca_variance(self, X: np.ndarray, method: str) -> str:
        """可视化PCA方差解释并返回HTML片段

        可视化PCA降维后各主成分的方差解释比例，用于评估降维效果。

        Args:
            X (np.ndarray): 特征矩阵，shape (n_samples, n_features)
            method (str): 可视化方法名称，用于区分不同的可视化结果

        Returns:
            str: 包含PCA方差解释图的HTML片段

        Examples:
            >>> from network_simulation.visualization.visualizer import Visualizer
            >>> from pathlib import Path
            >>> import numpy as np
            >>> visualizer = Visualizer(Path("output/visualizations"))
            >>> X = np.random.randn(100, 12)
            >>> html = visualizer.visualize_pca_variance(X, "pca_variance")
            >>> print(html)
        """
        return self.feature_visualizer.visualize_pca_variance(X, method)

    def visualize_tsne_scatter(
        self, X: np.ndarray, labels: np.ndarray, method: str
    ) -> str:
        """可视化t-SNE散点图并返回HTML片段

        使用t-SNE(t-distributed Stochastic Neighbor Embedding)对特征空间进行降维，
        并可视化不同行为模式的分布，适合展示高维数据的聚类结构。

        Args:
            X (np.ndarray): 特征矩阵，shape (n_samples, n_features)
            labels (np.ndarray): 行为标签数组，shape (n_samples,)
            method (str): 可视化方法名称，用于区分不同的可视化结果

        Returns:
            str: 包含t-SNE散点图的HTML片段

        Examples:
            >>> from network_simulation.visualization.visualizer import Visualizer
            >>> from pathlib import Path
            >>> import numpy as np
            >>> visualizer = Visualizer(Path("output/visualizations"))
            >>> X = np.random.randn(100, 12)
            >>> labels = np.random.randint(0, 3, 100)
            >>> html = visualizer.visualize_tsne_scatter(X, labels, "tsne")
            >>> print(html)
        """
        return self.feature_visualizer.visualize_tsne_scatter(X, labels, method)

    def visualize_umap_scatter(
        self, X: np.ndarray, labels: np.ndarray, method: str
    ) -> str:
        """可视化UMAP散点图并返回HTML片段

        使用UMAP(Uniform Manifold Approximation and Projection)对特征空间进行降维，
        并可视化不同行为模式的分布，适合保留高维数据的局部结构。

        Args:
            X (np.ndarray): 特征矩阵，shape (n_samples, n_features)
            labels (np.ndarray): 行为标签数组，shape (n_samples,)
            method (str): 可视化方法名称，用于区分不同的可视化结果

        Returns:
            str: 包含UMAP散点图的HTML片段

        Examples:
            >>> from network_simulation.visualization.visualizer import Visualizer
            >>> from pathlib import Path
            >>> import numpy as np
            >>> visualizer = Visualizer(Path("output/visualizations"))
            >>> X = np.random.randn(100, 12)
            >>> labels = np.random.randint(0, 3, 100)
            >>> html = visualizer.visualize_umap_scatter(X, labels, "umap")
            >>> print(html)
        """
        return self.feature_visualizer.visualize_umap_scatter(X, labels, method)

    def visualize_feature_distribution(
        self, X: np.ndarray, labels: np.ndarray, feature_names: List[str], method: str
    ) -> str:
        """可视化特征分布并返回HTML片段

        可视化不同行为模式的特征分布，用于比较不同行为模式的特征差异。

        Args:
            X (np.ndarray): 特征矩阵，shape (n_samples, n_features)
            labels (np.ndarray): 行为标签数组，shape (n_samples,)
            feature_names (List[str]): 特征名称列表，用于图表标题
            method (str): 可视化方法名称，用于区分不同的可视化结果

        Returns:
            str: 包含特征分布可视化的HTML片段

        Examples:
            >>> from network_simulation.visualization.visualizer import Visualizer
            >>> from pathlib import Path
            >>> import numpy as np
            >>> visualizer = Visualizer(Path("output/visualizations"))
            >>> X = np.random.randn(100, 3)
            >>> labels = np.random.randint(0, 3, 100)
            >>> feature_names = ["delay_std", "loss_nonzero_ratio", "max_congestion_run"]
            >>> html = visualizer.visualize_feature_distribution(X, labels, feature_names, "feature_dist")
            >>> print(html)
        """
        return self.behavior_visualizer.visualize_feature_distribution(
            X, labels, feature_names, method
        )

    def visualize_correlation_heatmap(
        self, X: np.ndarray, feature_names: List[str], method: str
    ) -> str:
        """可视化特征相关性热力图并返回HTML片段

        可视化特征之间的相关性热力图，用于分析特征之间的相互关系。

        Args:
            X (np.ndarray): 特征矩阵，shape (n_samples, n_features)
            feature_names (List[str]): 特征名称列表，用于图表坐标轴
            method (str): 可视化方法名称，用于区分不同的可视化结果

        Returns:
            str: 包含特征相关性热力图的HTML片段

        Examples:
            >>> from network_simulation.visualization.visualizer import Visualizer
            >>> from pathlib import Path
            >>> import numpy as np
            >>> visualizer = Visualizer(Path("output/visualizations"))
            >>> X = np.random.randn(100, 5)
            >>> feature_names = ["delay_std", "loss_nonzero_ratio", "max_congestion_run", "delay_trend", "delay_acf"]
            >>> html = visualizer.visualize_correlation_heatmap(X, feature_names, "correlation")
            >>> print(html)
        """
        return self.behavior_visualizer.visualize_correlation_heatmap(
            X, feature_names, method
        )

    def visualize_transition_matrix(
        self, transition_matrix: np.ndarray, method: str
    ) -> str:
        """可视化状态转移矩阵热力图并返回HTML片段

        可视化状态转移矩阵的热力图，用于分析网络行为模式之间的转移关系。

        Args:
            transition_matrix (np.ndarray): 状态转移矩阵，shape (n_behaviors, n_behaviors)
                - 每个元素表示从行为i转移到行为j的概率
            method (str): 可视化方法名称，用于区分不同的可视化结果

        Returns:
            str: 包含状态转移矩阵热力图的HTML片段

        Examples:
            >>> from network_simulation.visualization.visualizer import Visualizer
            >>> from pathlib import Path
            >>> import numpy as np
            >>> visualizer = Visualizer(Path("output/visualizations"))
            >>> transition_matrix = np.array([
            ...     [0.7, 0.2, 0.1],
            ...     [0.3, 0.5, 0.2],
            ...     [0.2, 0.3, 0.5]
            ... ])
            >>> html = visualizer.visualize_transition_matrix(transition_matrix, "transition")
            >>> print(html)
        """
        return self.behavior_visualizer.visualize_transition_matrix(
            transition_matrix, method
        )

    def generate_html_report(
        self,
        results: Dict,
        X: np.ndarray,
        feature_names: List[str],
        raw_data=None,
        features_df=None,
    ) -> None:
        """生成综合HTML报告

        生成综合HTML报告，包含多种可视化图表、统计信息和评估结果。

        Args:
            results (Dict): 包含评估结果的字典
            X (np.ndarray): 特征矩阵，shape (n_samples, n_features)
            feature_names (List[str]): 特征名称列表
            raw_data (optional): 原始数据，用于生成原始数据样本可视化
            features_df (optional): 特征数据框，用于生成特征空间可视化

        Examples:
            >>> from network_simulation.visualization.visualizer import Visualizer
            >>> from pathlib import Path
            >>> import numpy as np
            >>> visualizer = Visualizer(Path("output/visualizations"))
            >>> results = {"method": "rule"}
            >>> X = np.random.randn(100, 12)
            >>> feature_names = [f"feat_{i}" for i in range(12)]
            >>> visualizer.generate_html_report(results, X, feature_names)
        """
        self.report_generator.generate_html_report(
            results, X, feature_names, raw_data, features_df
        )

    def _generate_separation_table(
        self, separation_metrics: Dict, feature_names: List[str]
    ) -> str:
        """生成分离度表格的HTML格式

        生成分离度表格的HTML格式，用于展示不同行为模式的分离度指标。

        Args:
            separation_metrics (Dict): 分离度指标字典，包含类内距离、类间距离等
            feature_names (List[str]): 特征名称列表

        Returns:
            str: 包含分离度表格的HTML片段

        Examples:
            >>> from network_simulation.visualization.visualizer import Visualizer
            >>> from pathlib import Path
            >>> import numpy as np
            >>> visualizer = Visualizer(Path("output/visualizations"))
            >>> separation_metrics = {
            ...     "avg_inter_cluster_distance": 5.0,
            ...     "avg_intra_cluster_distance": 1.0,
            ...     "separation_index": 5.0
            ... }
            >>> feature_names = ["delay_std", "loss_nonzero_ratio", "max_congestion_run"]
            >>> html = visualizer._generate_separation_table(separation_metrics, feature_names)
            >>> print(html)
        """
        return self.report_generator._generate_separation_table_markdown(
            separation_metrics, feature_names
        )

    def visualize_raw_data_samples(
        self, _raw_data, _features_df, _labels, _method
    ) -> str:
        """可视化原始数据样本并返回HTML片段

        可视化原始数据样本，包括时间序列、统计分布等。

        Args:
            _raw_data: 原始数据，包含网络模拟数据的DataFrame
            _features_df: 特征数据框，包含提取的特征
            _labels: 行为标签数组
            _method: 可视化方法名称

        Returns:
            str: 包含原始数据样本可视化的HTML片段

        Examples:
            >>> from network_simulation.visualization.visualizer import Visualizer
            >>> from pathlib import Path
            >>> visualizer = Visualizer(Path("output/visualizations"))
            >>> html = visualizer.visualize_raw_data_samples(None, None, None, "raw_data")
            >>> print(html)
        """
        return "<h2>原始数据样本可视化</h2>"

    def generate_all_visualizations(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        transition_matrix: np.ndarray,
        output_dir: Path,
        feature_names: List[str] = None,
        direction: str = "up",
    ) -> None:
        """生成所有评估可视化图表"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成转移可视化
        self.behavior_visualizer.generate_transition_plots(
            transition_matrix, output_dir
        )

        # 生成行为分离可视化
        self.feature_visualizer.generate_behavior_separation_plots(
            X, labels, output_dir, feature_names, direction
        )

    def generate_evaluation_report(self, results: Dict, output_dir: Path) -> None:
        """生成评估报告"""
        output_dir.mkdir(parents=True, exist_ok=True)
        # 生成评估报告，使用evaluation_results作为方法名
        self.report_generator.generate_evaluation_markdown_report(results, output_dir)
        self.report_generator.generate_evaluation_html_report(results, output_dir)

    def visualize_results(
        self,
        input_original_file: Path,
        input_generated_file: Path,
        is_main_visualization: bool = False,
    ) -> Path:
        """可视化生成结果与原始数据的对比

        该函数生成多种可视化图表，对比原始网络数据和生成的网络数据，
        包括时间序列对比、统计分布对比和相关性对比。

        Args:
            input_original_file: 原始样本数据文件路径
            input_generated_file: 生成样本数据文件路径
            is_main_visualization: 是否为主要可视化（生成综合报告所需的图片）

        Returns:
            Path: 可视化结果的输出目录路径
        """
        return self.results_visualizer.visualize_results(
            input_original_file, input_generated_file, is_main_visualization
        )

    def visualize_batch_results(self, input_generation_dir: Path) -> None:
        """批量可视化结果

        Args:
            input_generation_dir: 生成样本的目录路径，包含original_sample_6000*.csv和generated_sample_6000*.csv文件
        """
        self.results_visualizer.visualize_batch_results(input_generation_dir)


# 导出所有可视化类，方便直接导入使用
export = {
    "BaseVisualizer": BaseVisualizer,
    "FeatureSpaceVisualizer": FeatureSpaceVisualizer,
    "BehaviorVisualizer": BehaviorVisualizer,
    "ReportGenerator": ReportGenerator,
    "Visualizer": Visualizer,
}

__all__ = list(export.keys())
