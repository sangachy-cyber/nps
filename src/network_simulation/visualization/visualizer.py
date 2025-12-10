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
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_visualizer = BaseVisualizer(output_dir)
        self.feature_visualizer = FeatureSpaceVisualizer(output_dir)
        self.behavior_visualizer = BehaviorVisualizer(output_dir)
        self.report_generator = ReportGenerator(self.base_visualizer)

    def visualize_evaluation_results(self, _results: Dict) -> str:
        """可视化评估结果并返回HTML片段"""
        # 后续将实现
        return "<h2>评估结果可视化</h2>"

    def visualize_pca_scatter(
        self, X: np.ndarray, labels: np.ndarray, method: str
    ) -> str:
        """可视化PCA散点图并返回HTML片段"""
        return self.feature_visualizer.visualize_pca_scatter(X, labels, method)

    def visualize_pca_variance(self, X: np.ndarray, method: str) -> str:
        """可视化PCA方差解释并返回HTML片段"""
        return self.feature_visualizer.visualize_pca_variance(X, method)

    def visualize_tsne_scatter(
        self, X: np.ndarray, labels: np.ndarray, method: str
    ) -> str:
        """可视化t-SNE散点图并返回HTML片段"""
        return self.feature_visualizer.visualize_tsne_scatter(X, labels, method)

    def visualize_umap_scatter(
        self, X: np.ndarray, labels: np.ndarray, method: str
    ) -> str:
        """可视化UMAP散点图并返回HTML片段"""
        return self.feature_visualizer.visualize_umap_scatter(X, labels, method)

    def visualize_feature_distribution(
        self, X: np.ndarray, labels: np.ndarray, feature_names: List[str], method: str
    ) -> str:
        """可视化特征分布并返回HTML片段"""
        return self.behavior_visualizer.visualize_feature_distribution(X, labels, feature_names, method)

    def visualize_correlation_heatmap(
        self, X: np.ndarray, feature_names: List[str], method: str
    ) -> str:
        """可视化特征相关性热力图并返回HTML片段"""
        return self.behavior_visualizer.visualize_correlation_heatmap(X, feature_names, method)

    def visualize_transition_matrix(
        self, transition_matrix: np.ndarray, method: str
    ) -> str:
        """可视化状态转移矩阵热力图并返回HTML片段"""
        return self.behavior_visualizer.visualize_transition_matrix(transition_matrix, method)

    def generate_html_report(
        self, results: Dict, X: np.ndarray, feature_names: List[str], raw_data=None, features_df=None
    ) -> None:
        """生成综合HTML报告"""
        self.report_generator.generate_html_report(results, X, feature_names, raw_data, features_df)

    def _generate_separation_table(
        self, separation_metrics: Dict, feature_names: List[str]
    ) -> str:
        """生成分离度表格的HTML格式"""
        return self.report_generator._generate_separation_table_markdown(separation_metrics, feature_names)

    def visualize_raw_data_samples(
        self, _raw_data, _features_df, _labels, _method
    ) -> str:
        """可视化原始数据样本并返回HTML片段"""
        return "<h2>原始数据样本可视化</h2>"

    def generate_all_visualizations(
        self, X: np.ndarray, labels: np.ndarray, transition_matrix: np.ndarray, output_dir: Path,
        feature_names: List[str] = None, direction: str = "up"
    ) -> None:
        """生成所有评估可视化图表"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成转移可视化
        self.behavior_visualizer.generate_transition_plots(transition_matrix, output_dir)

        # 生成行为分离可视化
        self.feature_visualizer.generate_behavior_separation_plots(X, labels, output_dir, feature_names, direction)

    def generate_evaluation_report(
        self, results: Dict, output_dir: Path
    ) -> None:
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
        return self.results_visualizer.visualize_results(input_original_file, input_generated_file, is_main_visualization)

    def visualize_batch_results(
        self,
        input_generation_dir: Path
    ) -> None:
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
    "Visualizer": Visualizer
}

__all__ = list(export.keys())
