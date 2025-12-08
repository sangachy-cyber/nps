#!/usr/bin/env python3
"""
网络行为可视化模块
负责可视化网络行为模式和评估结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from pathlib import Path
from typing import Dict, List
import base64
from io import BytesIO

from ..utils.logger import get_logger
from config import REPORTS_DIR


logger = get_logger(__name__)

class Visualizer:
    """可视化网络行为模式和评估结果"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置中文支持，Linux系统优先使用文泉驿正黑
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    def visualize_evaluation_results(self, results: Dict) -> str:
        """可视化评估结果并返回HTML片段"""
        # 后续将实现
        return "<h2>评估结果可视化</h2>"

    def visualize_pca_scatter(
        self, X: np.ndarray, labels: np.ndarray, method: str
    ) -> str:
        """可视化PCA散点图并返回HTML片段"""
        # 执行PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # 创建散点图
        plt.figure(figsize=(10, 7))
        unique_labels = np.unique(labels)
        colors = sns.color_palette("hsv", len(unique_labels))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=[color],
                label=f"行为 {label}",
                alpha=0.6,
            )

        plt.title(f"{method} 行为检测结果 PCA 散点图")
        plt.xlabel(f"PCA 维度 1 ({pca.explained_variance_ratio_[0]:.2%} 方差)")
        plt.ylabel(f"PCA 维度 2 ({pca.explained_variance_ratio_[1]:.2%} 方差)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存图表到缓冲区
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()

        return f'<img src="data:image/png;base64,{img_base64}" alt="PCA 散点图">'

    def visualize_pca_variance(self, X: np.ndarray, method: str) -> str:
        """可视化PCA方差解释并返回HTML片段"""
        # 执行PCA
        pca = PCA(n_components=min(8, X.shape[1]))
        pca.fit(X)

        # 创建方差解释图
        plt.figure(figsize=(10, 7))
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        plt.bar(
            range(1, len(explained_variance) + 1),
            explained_variance,
            alpha=0.7,
            align="center",
            label="单个PCA维度方差",
        )
        plt.step(
            range(1, len(cumulative_variance) + 1),
            cumulative_variance,
            where="mid",
            label="累计方差",
        )
        plt.ylabel("方差解释比例")
        plt.xlabel("PCA维度数量")
        plt.title(f"{method} 行为检测 PCA 方差解释图")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)

        # 保存图表到缓冲区
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()

        return f'<img src="data:image/png;base64,{img_base64}" alt="PCA 方差解释图">'

    def visualize_tsne_scatter(
        self, X: np.ndarray, labels: np.ndarray, method: str
    ) -> str:
        """可视化t-SNE散点图并返回HTML片段"""
        # 执行t-SNE
        perplexity = 30
        max_iter = 300
        tsne = TSNE(
            n_components=2, random_state=42, perplexity=perplexity, max_iter=max_iter
        )
        X_tsne = tsne.fit_transform(X)

        # 创建散点图
        plt.figure(figsize=(10, 7))
        unique_labels = np.unique(labels)
        colors = sns.color_palette("hsv", len(unique_labels))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(
                X_tsne[mask, 0],
                X_tsne[mask, 1],
                c=[color],
                label=f"行为 {label}",
                alpha=0.6,
            )

        plt.title(
            f"{method} 行为检测结果 t-SNE 散点图 (perplexity={perplexity}, max_iter={max_iter})"
        )
        plt.xlabel(f"t-SNE 维度 1 (perplexity={perplexity})")
        plt.ylabel(f"t-SNE 维度 2 (max_iter={max_iter})")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存图表到缓冲区
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()

        return f'<img src="data:image/png;base64,{img_base64}" alt="t-SNE 散点图">'

    def visualize_umap_scatter(
        self, X: np.ndarray, labels: np.ndarray, method: str
    ) -> str:
        """可视化UMAP散点图并返回HTML片段"""
        # 执行UMAP
        n_neighbors = 15
        min_dist = 0.1
        umap = UMAP(
            n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=min_dist
        )
        X_umap = umap.fit_transform(X)

        # 创建散点图
        plt.figure(figsize=(10, 7))
        unique_labels = np.unique(labels)
        colors = sns.color_palette("hsv", len(unique_labels))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(
                X_umap[mask, 0],
                X_umap[mask, 1],
                c=[color],
                label=f"行为 {label}",
                alpha=0.6,
            )

        plt.title(
            f"{method} 行为检测结果 UMAP 散点图 (n_neighbors={n_neighbors}, min_dist={min_dist})"
        )
        plt.xlabel(f"UMAP 维度 1 (n_neighbors={n_neighbors})")
        plt.ylabel(f"UMAP 维度 2 (min_dist={min_dist})")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存图表到缓冲区
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()

        return f'<img src="data:image/png;base64,{img_base64}" alt="UMAP 散点图">'

    def visualize_feature_distribution(
        self, X: np.ndarray, labels: np.ndarray, feature_names: List[str], method: str
    ) -> str:
        """可视化特征分布并返回HTML片段"""
        unique_labels = np.unique(labels)
        n_features = len(feature_names)

        # 创建子图
        fig, axes = plt.subplots(
            n_features, 1, figsize=(12, 3 * n_features), sharex=False
        )

        for i, (feature_name, ax) in enumerate(zip(feature_names, axes)):
            for label in unique_labels:
                mask = labels == label
                sns.histplot(
                    X[mask, i], ax=ax, label=f"行为 {label}", alpha=0.5, kde=True
                )
            ax.set_title(f"{feature_name} 分布")
            ax.set_xlabel(feature_name)
            ax.set_ylabel("频率")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表到缓冲区
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()

        return f'<img src="data:image/png;base64,{img_base64}" alt="特征分布直方图">'

    def visualize_correlation_heatmap(
        self, X: np.ndarray, feature_names: List[str], method: str
    ) -> str:
        """可视化特征相关性热力图并返回HTML片段"""
        # 计算相关系数矩阵
        corr_matrix = np.corrcoef(X.T)

        # 创建热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            xticklabels=feature_names,
            yticklabels=feature_names,
            square=True,
        )
        plt.title(f"{method} 行为检测 特征相关性热力图")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # 保存图表到缓冲区
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()

        return f'<img src="data:image/png;base64,{img_base64}" alt="特征相关性热力图">'

    def visualize_transition_matrix(
        self, transition_matrix: np.ndarray, method: str
    ) -> str:
        """可视化状态转移矩阵热力图并返回HTML片段"""
        # 创建热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            transition_matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=[f"行为 {i}" for i in range(transition_matrix.shape[1])],
            yticklabels=[f"行为 {i}" for i in range(transition_matrix.shape[0])],
            square=True,
        )
        plt.title(f"{method} 行为检测 状态转移矩阵热力图")
        plt.xlabel("下一行为")
        plt.ylabel("当前行为")
        plt.tight_layout()

        # 保存图表到缓冲区
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()

        return (
            f'<img src="data:image/png;base64,{img_base64}" alt="状态转移矩阵热力图">'
        )

    def generate_html_report(
        self,
        results: Dict,
        X: np.ndarray,
        feature_names: List[str],
        raw_data: pd.DataFrame = None,
        features_df: pd.DataFrame = None,
    ) -> None:
        """生成综合HTML报告"""
        method = results["method"]

        # 使用配置的报告目录作为上一层目录
        # 确保可视化目录结构正确
        plots_dir = REPORTS_DIR / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        behavior_samples_dir = plots_dir / "behavior_samples"
        behavior_samples_dir.mkdir(exist_ok=True)
        
        timelines_dir = plots_dir / "timelines"
        timelines_dir.mkdir(exist_ok=True)
        
        feature_space_dir = plots_dir / "feature_space"
        feature_space_dir.mkdir(exist_ok=True)

        # Save standalone visualizations to their respective directories
        if raw_data is not None and features_df is not None:
            # Save behavior samples (1-3 per class)
            self._save_behavior_samples(raw_data, features_df, results["labels"], method, behavior_samples_dir)
            
            # Save label timeline
            self._save_label_timeline(raw_data, features_df, results["labels"], method, timelines_dir)
            
            # Save typical samples
            self._save_typical_samples(raw_data, features_df, results["labels"], method, behavior_samples_dir)
        
        # Save feature space visualizations
        self._save_feature_space_visualizations(X, results["labels"], method, feature_space_dir, feature_names)
        
        # Save transition-related visualizations
        transition_matrix = np.array(results["transition_matrix"])
        self._save_interactive_transition_graph(transition_matrix, results["labels"], method, plots_dir)
        self._save_transition_metrics(transition_matrix, results["labels"], method, plots_dir)

        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>网络行为模式发现报告 - {method} 行为检测</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #eee; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>网络行为模式发现报告</h1>
                <h2>行为检测方法：{"规则" if method == "rule" else method}</h2>

                <div class="section">
            <h2>1. 行为转移质量评估</h2>
            <table class="metrics-table">
                <tr>
                    <th>指标名称</th>
                    <th>数值</th>
                    <th>说明</th>
                </tr>
                <tr>
                    <td>平均转移熵</td>
                    <td>{results["metrics"]["average_transition_entropy"]:.4f}</td>
                    <td>衡量状态转移不确定性，越低越确定</td>
                </tr>
                <tr>
                    <td>转移稀疏性</td>
                    <td>{results["metrics"]["transition_sparsity"]:.4f}</td>
                    <td>非零转移概率占比，反映行为切换复杂度</td>
                </tr>
            </table>
        </div>

                <div class="section">
                    <h2>2. 特征降维可视化</h2>
                    <p>不同降维方法的对比：</p>
                    <p>- PCA：线性降维，保留最大方差</p>
                    <p>- t-SNE：非线性降维，专注于局部结构，适合可视化高维数据</p>
                    <p>- UMAP：非线性降维，同时保留局部和全局结构，运行速度更快</p>

                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0;">
                        <div style="text-align: center;">
                            <h3>2.1 PCA 散点图</h3>
                            {self.visualize_pca_scatter(X, np.array(results["labels"]), method)}
                        </div>
                        <div style="text-align: center;">
                            <h3>2.2 PCA 方差解释图</h3>
                            {self.visualize_pca_variance(X, method)}
                        </div>
                        <div style="text-align: center;">
                            <h3>2.3 t-SNE 散点图</h3>
                            {self.visualize_tsne_scatter(X, np.array(results["labels"]), method)}
                        </div>
                        <div style="text-align: center;">
                            <h3>2.4 UMAP 散点图</h3>
                            {self.visualize_umap_scatter(X, np.array(results["labels"]), method)}
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h2>3. 特征分析</h2>
                    <h3>3.1 特征分布直方图</h3>
                    {self.visualize_feature_distribution(X, np.array(results["labels"]), feature_names, method)}

                    <h3>3.2 特征相关性热力图</h3>
                    {self.visualize_correlation_heatmap(X, feature_names, method)}
                </div>

                <div class="section">
                    <h2>4. 行为转移分析</h2>
                    <h3>4.1 状态转移矩阵热力图</h3>
                    {self.visualize_transition_matrix(np.array(results["transition_matrix"]), method)}
                </div>

                <div class="section">
                    <h2>5. 行为特征分析</h2>
                    <h3>5.1 各行为类别特征均值</h3>
                    <table class="metrics-table">
                        <tr>
                            <th>行为类别</th>
                            {"".join([f"<th>{name}</th>" for name in feature_names])}
                        </tr>
                        {self._generate_separation_table(results["separation_metrics"], feature_names)}
                    </table>
                </div>

                <div class="section">
            <h2>6. 标签时间轴图</h2>
            <p>全局行为分布可视化，展示不同网络行为在时间轴上的分布：</p>
            {self.visualize_label_timeline(raw_data, features_df, results["labels"], method) if (raw_data is not None and features_df is not None) else "<p>原始数据未提供，无法生成标签时间轴图</p>"}
        </div>

        <div class="section">
            <h2>7. 原始数据样本可视化</h2>
            <p>以下是每个行为类别的典型样本对应的原始时延和丢包率可视化：</p>
            {self.visualize_raw_data_samples(raw_data, features_df, results["labels"], method) if (raw_data is not None and features_df is not None) else "<p>原始数据未提供，无法生成原始数据样本可视化</p>"}
        </div>
            </div>
        </body>
        </html>
        """

        # Save HTML report to configured directory with appropriate name
        report_file = REPORTS_DIR / f"behavior_pattern_report_{method}.html"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML报告已生成：{report_file}")
        logger.info(f"可视化资产已保存到目录：{plots_dir}")
    
    def _save_behavior_samples(
        self,
        raw_data: pd.DataFrame,
        features_df: pd.DataFrame,
        labels: List[int],
        method: str,
        output_dir: Path
    ) -> None:
        """保存每类2个行为样本图到behavior_samples目录，优化显示效果"""
        import random
        
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        
        # 定义行为标签映射
        behavior_labels = {
            0: "稳定",
            1: "弱突发",
            2: "强突发",
            3: "瞬时峰值",
            4: "高延迟无丢包",
            5: "高丢包低延迟",
            6: "强突发高延迟",
            7: "复杂网络行为"
        }
        
        for label in unique_labels:
            label_mask = labels == label
            label_indices = np.where(label_mask)[0]
            
            if len(label_indices) > 0:
                # Get actual behavior name
                behavior_name = behavior_labels.get(label, f"行为 {label}")
                
                # 每个类别挑2个样本
                num_samples = min(2, len(label_indices))
                sample_indices = random.sample(list(label_indices), num_samples)
                
                for i, sample_idx in enumerate(sample_indices):
                    # Get window start and end indices from features_df
                    window_start = int(features_df.iloc[sample_idx]["window_start"])
                    window_end = int(features_df.iloc[sample_idx]["window_end"])
                    
                    # Extract raw data for this window
                    window_data = raw_data.iloc[window_start:window_end]
                    
                    # Create dual-axis plot for delay and loss_rate
                    plt.figure(figsize=(10, 6))
                    
                    # Plot delay on primary y-axis，统一最大值2000ms
                    ax1 = plt.subplot(111)
                    ax1.plot(
                        window_data["timestamp"],
                        window_data["delay"],
                        "b-",
                        linewidth=2,
                        label="时延 (ms)",
                    )
                    ax1.set_xlabel("时间")
                    ax1.set_ylabel("时延 (ms)", color="b")
                    ax1.tick_params("y", colors="b")
                    ax1.set_ylim(0, 2000)  # 统一时延最大值2000ms
                    ax1.grid(True, alpha=0.3)
                    
                    # Plot loss_rate on secondary y-axis，优化0值显示
                    ax2 = ax1.twinx()
                    ax2.plot(
                        window_data["timestamp"],
                        window_data["loss_rate"],
                        "r-",
                        linewidth=2,
                        label="丢包率",
                    )
                    ax2.set_ylabel("丢包率", color="r")
                    ax2.tick_params("y", colors="r")
                    ax2.set_ylim(-0.01, 1.1)  # 从-0.01开始，优化0值显示
                    
                    # 为丢包率添加轻微抖动，使0值更容易区分
                    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
                    
                    # Add title and legend
                    method_name = "规则" if method == "rule" else method
                    plt.title(f"{behavior_name} - 样本 {i + 1} ({method_name} 检测)")
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
                    
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Save plot to file
                    sample_file = output_dir / f"behavior_{label}_sample_{i + 1}.png"
                    plt.savefig(sample_file, dpi=150, bbox_inches="tight")
                    plt.close()
                    logger.debug(f"保存行为样本：{sample_file}")
    
    def _save_label_timeline(
        self,
        raw_data: pd.DataFrame,
        features_df: pd.DataFrame,
        labels: List[int],
        method: str,
        output_dir: Path
    ) -> None:
        """保存标签时间轴图到timelines目录，按6分钟拆分，每张图包含原始时延和丢包率（双Y轴）"""
        # Ensure labels is numpy array
        labels = np.array(labels)
        
        # Create color mapping for different labels - use high contrast color scheme
        unique_labels = np.unique(labels)
        
        # Use high contrast custom colors for better visibility
        # 稳定：淡绿色，弱突发：橙色，瞬时峰值：亮黄色，强突发：红色，其他：高对比度颜色
        custom_colors = {
            0: '#4CAF50',  # 稳定 - 深绿色
            1: '#FF9800',  # 弱突发 - 橙色
            2: '#F44336',  # 强突发 - 红色
            3: '#FFEB3B',  # 瞬时峰值 - 黄色
            4: '#2196F3',  # 高延迟无丢包 - 蓝色
            5: '#9C27B0',  # 其他 - 紫色
            6: '#FF5722',  # 其他 - 深橙色
            7: '#00BCD4'   # 其他 - 青色
        }
        
        # Verify all labels are within defined color mapping range
        assert np.all(np.isin(labels, list(custom_colors.keys()))), f"发现未定义的 label: {np.setdiff1d(labels, list(custom_colors.keys()))}"
        
        # Create custom color mapping function
        def color_map(label):
            return custom_colors.get(label, '#808080')  # 默认灰色
        
        # Define behavior label mapping
        behavior_labels = {
            0: "稳定",
            1: "弱突发",
            2: "强突发",
            3: "瞬时峰值",
            4: "高延迟无丢包",
            5: "高丢包低延迟",
            6: "强突发高延迟",
            7: "复杂网络行为"
        }
        
        # Ensure timestamp column is datetime type
        raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])
        
        # 为每个文件单独处理时间轴
        unique_files = raw_data['file_path'].unique()
        
        for current_file_path in unique_files:
            # 获取当前文件的所有数据
            file_data = raw_data[raw_data['file_path'] == current_file_path].copy()
            if file_data.empty:
                continue
            
            # 计算当前文件的相对时间（从0开始，单位：秒）
            file_data['relative_time'] = (file_data['timestamp'] - file_data['timestamp'].min()).dt.total_seconds()
            
            # 获取当前文件的时间范围
            file_start_time = file_data['timestamp'].min()
            file_end_time = file_data['timestamp'].max()
            
            # 计算时间 delta（6分钟）- 根据用户要求
            time_delta = pd.Timedelta(minutes=6)
            current_time = file_start_time
            
            # Counter for window files for this file
            window_count = 0
            
            while current_time < file_end_time:
                window_count += 1
                window_end = current_time + time_delta
                
                # Ensure last window doesn't exceed end time
                if window_end > file_end_time:
                    window_end = file_end_time
                
                # Extract data for current time window from the current file
                window_mask = (file_data['timestamp'] >= current_time) & (file_data['timestamp'] < window_end)
                window_data = file_data[window_mask]
                
                # Skip if no data in current window
                if len(window_data) == 0:
                    current_time = window_end
                    continue
                
                # Create plot for current window
                fig, ax1 = plt.subplots(figsize=(15, 8))
                
                # Resample data if needed, reduce data points to 500
                if len(window_data) > 500:
                    step = len(window_data) // 500
                    sampled_data = window_data.iloc[::step]
                else:
                    sampled_data = window_data
                
                # Ensure data is sorted by relative_time
                sampled_data = sampled_data.sort_values('relative_time')
                
                # Smooth the data with moving average
                sampled_data['delay_smooth'] = sampled_data['delay'].rolling(window=3, min_periods=1).mean()
                sampled_data['loss_rate_smooth'] = sampled_data['loss_rate'].rolling(window=3, min_periods=1).mean()
                
                # Debug information
                logger.debug(f"文件 {current_file_path} 窗口 {window_count} 数据点数量: {len(sampled_data)}")
                logger.debug(f"relative_time范围: {sampled_data['relative_time'].min()} - {sampled_data['relative_time'].max()}")
                logger.debug(f"delay范围: {sampled_data['delay'].min()} - {sampled_data['delay'].max()}")
                logger.debug(f"loss_rate范围: {sampled_data['loss_rate'].min()} - {sampled_data['loss_rate'].max()}")
                
                # Unified delay range: 0-2000ms
                y_min = 0
                y_max = 2000
                
                # Plot behavior label blocks
                for i, label in enumerate(labels):
                    # Get window start and end indices
                    window_start_idx = int(features_df.iloc[i]['window_start'])
                    window_end_idx = int(features_df.iloc[i]['window_end'])
                    
                    # Ensure window_end_idx doesn't exceed raw_data length
                    window_end_idx = min(window_end_idx, len(raw_data))
                    
                    # Get time range for this behavior window
                    win_time_start = raw_data.iloc[window_start_idx]['timestamp']
                    win_time_end = raw_data.iloc[window_end_idx-1]['timestamp']
                    
                    # Only plot if behavior window is within current file's time range and current window
                    if win_time_end < current_time or win_time_start >= window_end:
                        continue
                    
                    # Check if this behavior window belongs to current file
                    behavior_window_data = raw_data.iloc[window_start_idx:window_end_idx]
                    if current_file_path not in behavior_window_data['file_path'].values:
                        continue
                    
                    # Calculate overlapping relative time range
                    overlap_start = max(win_time_start, current_time)
                    overlap_end = min(win_time_end, window_end)
                    
                    # Convert to relative time using current file's start time
                    overlap_start_relative = (overlap_start - file_start_time).total_seconds()
                    overlap_end_relative = (overlap_end - file_start_time).total_seconds()
                    
                    # Plot the behavior block
                    ax1.fill_between(
                        x=[overlap_start_relative, overlap_end_relative],
                        y1=y_min,
                        y2=y_max,
                        facecolor=color_map(label),
                        alpha=0.25,  # 降低透明度到0.25，减少颜色混合
                        edgecolor='black',  # 统一边框色为黑色，强化边界
                        linewidth=1.0,  # 增加边框宽度
                        zorder=0  # 确保在最底层
                    )
                
                # 设置Y轴范围
                ax1.set_ylim(y_min, y_max)
                
                # 绘制平滑后的时延曲线 - 线条更细
                delay_line, = ax1.plot(sampled_data['relative_time'], sampled_data['delay_smooth'], 'b-', linewidth=1.5, alpha=0.8, label='时延 (ms)')
                
                # 确保Y轴有明确的数值标记
                y_ticks = np.linspace(y_min, y_max, 5)
                ax1.set_yticks(y_ticks)
                # 设置Y轴刻度标签为明确的数值
                ax1.set_yticklabels([f'{tick:.0f}' for tick in y_ticks], fontweight='bold', fontsize=12, color='b')
                
                # 设置Y轴标签和样式，确保可见
                ax1.set_ylabel('时延 (ms)', color='b', fontsize=14, fontweight='bold', rotation=90, labelpad=20)
                ax1.tick_params('y', colors='b', labelsize=12, width=3, length=15, direction='out')
                
                # 确保Y轴轴线可见
                ax1.spines['left'].set_visible(True)
                ax1.spines['left'].set_color('b')
                ax1.spines['left'].set_linewidth(3)
                
                # 设置网格线
                ax1.grid(True, alpha=0.3, linestyle='--')
                
                # Create right Y-axis for loss rate, 使用更细的线条
                ax2 = ax1.twinx()
                # 绘制平滑后的丢包率，使用更细的实线
                loss_line, = ax2.plot(sampled_data['relative_time'], sampled_data['loss_rate_smooth'], 'r-', linewidth=1.5, alpha=0.8, label='丢包率')
                ax2.set_ylabel('丢包率', color='r', fontsize=14, fontweight='bold')
                ax2.tick_params('y', colors='r', labelsize=12, width=3, length=15, direction='out')
                ax2.set_ylim(-0.01, 1.1)  # Loss rate from -0.01 to 1.1, optimize 0 value display
                
                # 确保右侧Y轴轴线可见
                ax2.spines['right'].set_visible(True)
                ax2.spines['right'].set_color('r')
                ax2.spines['right'].set_linewidth(3)
                
                # Add reference line for loss rate to enhance readability
                ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
                
                # Get legends from both Y-axes
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                
                # Create behavior label legend handles
                behavior_handles = []
                behavior_labels_legend = []
                for label in unique_labels:
                    handle = plt.Rectangle((0, 0), 1, 1, facecolor=color_map(label), alpha=1.0, edgecolor=color_map(label), linewidth=0.5)
                    behavior_handles.append(handle)
                    behavior_labels_legend.append(behavior_labels.get(label, f"行为 {label}"))
                
                # Combine all legends
                all_handles = lines1 + lines2 + behavior_handles
                all_labels = labels1 + labels2 + behavior_labels_legend
                
                # Add combined legend at upper right
                ax1.legend(all_handles, all_labels, loc='upper right', fontsize=10, ncol=2)
                
                # Set plot title and labels
                file_name = current_file_path.split('/')[-1]  # Only keep filename
                
                # Calculate minutes in file for this window
                minutes_in_file = (current_time - file_start_time).total_seconds() / 60
                
                ax1.set_title(f"全局行为分布时间轴 - {file_name} - 对应文件的第 {int(minutes_in_file)}~{int(minutes_in_file+6)} 分钟 ({method} 检测)", fontsize=14)
                ax1.set_xlabel("相对时间 (秒)", fontsize=12)
                
                # Improve time ticks, show every 100 seconds
                x_min = sampled_data['relative_time'].min()
                x_max = sampled_data['relative_time'].max()
                x_ticks = np.arange(x_min, x_max + 1, 100)  # Every 100 seconds
                ax1.set_xticks(x_ticks)
                
                # Set X-axis tick labels format to seconds
                ax1.set_xticklabels([f'{tick:.0f}s' for tick in x_ticks], fontsize=10)
                
                # Optimize X-axis tick label rotation
                plt.xticks(rotation=45, ha='right')
                
                # Adjust layout
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.25)  # Increase bottom margin
                
                # Save plot to file with file-specific window count
                import os
                timeline_file = output_dir / f"{os.path.basename(current_file_path)}_timeline_window_{window_count}.png"
                plt.savefig(timeline_file, dpi=150, bbox_inches="tight")
                plt.close()
                logger.debug(f"保存标签时间轴图：{timeline_file}")
                
                # Move to next time window
                current_time = window_end
    
    def _save_feature_space_visualizations(
        self,
        X: np.ndarray,
        labels: List[int],
        method: str,
        output_dir: Path,
        feature_names: List[str]
    ) -> None:
        """保存特征空间降维图到feature_space目录"""
        labels = np.array(labels)
        
        # Save t-SNE scatter plot
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)
        X_tsne = tsne.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(labels)
        colors = sns.color_palette("hsv", len(unique_labels))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(
                X_tsne[mask, 0],
                X_tsne[mask, 1],
                c=[color],
                label=f"类别 {label}",
                alpha=0.6,
            )
        
        plt.title(f"行为特征 t-SNE 降维图")
        plt.xlabel("t-SNE 维度 1")
        plt.ylabel("t-SNE 维度 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        tsne_file = output_dir / "tsne.png"
        plt.savefig(tsne_file, dpi=150, bbox_inches="tight")
        plt.close()
        logger.debug(f"保存t-SNE降维图：{tsne_file}")
        
        # Save UMAP scatter plot
        umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        X_umap = umap.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(
                X_umap[mask, 0],
                X_umap[mask, 1],
                c=[color],
                label=f"类别 {label}",
                alpha=0.6,
            )
        
        plt.title(f"行为特征 UMAP 降维图")
        plt.xlabel("UMAP 维度 1")
        plt.ylabel("UMAP 维度 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        umap_file = output_dir / "umap.png"
        plt.savefig(umap_file, dpi=150, bbox_inches="tight")
        plt.close()
        logger.debug(f"保存UMAP降维图：{umap_file}")

    def _save_interactive_timeline(
        self,
        raw_data: pd.DataFrame,
        features_df: pd.DataFrame,
        labels: List[int],
        method: str,
        output_dir: Path,
        window_count: int
    ) -> None:
        """保存交互式时间轴图到HTML文件，支持交互操作"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            logger.error("Plotly 库未安装，无法生成交互式时间轴")
            return
        
        # 确保labels是numpy数组
        labels = np.array(labels)
        
        # 为不同标签创建颜色映射
        custom_colors = {
            0: '#98FB98',  # 稳定 - 淡绿色
            1: '#FFA500',  # 弱突发 - 橙色
            2: '#FF0000',  # 强突发 - 红色
            3: '#FFFF00',  # 瞬时峰值 - 亮黄色
            4: '#1E90FF',  # 高延迟无丢包 - 蓝色
            5: '#FF69B4',  # 其他 - 粉红色
            6: '#9400D3'   # 其他 - 紫色
        }
        
        # 计算相对时间
        raw_data['relative_time'] = (raw_data['timestamp'] - raw_data['timestamp'].min()).dt.total_seconds()
        
        # 定义行为标签映射
        behavior_labels = {
            0: "稳定",
            1: "弱突发",
            2: "强突发",
            3: "瞬时峰值",
            4: "高延迟无丢包",
            5: "其他",
            6: "其他"
        }
        
        # 创建交互式图表
        fig = go.Figure()
        
        # 添加时延曲线
        fig.add_trace(go.Scatter(
            x=raw_data['relative_time'],
            y=raw_data['delay'],
            name='时延 (ms)',
            line=dict(color='blue', width=3),
            yaxis='y1'
        ))
        
        # 添加丢包率曲线
        fig.add_trace(go.Scatter(
            x=raw_data['relative_time'],
            y=raw_data['loss_rate'],
            name='丢包率',
            line=dict(color='red', width=2, dash='dash'),
            yaxis='y2'
        ))
        
        # 添加行为标签色块
        for i, label in enumerate(labels):
            window_start_idx = int(features_df.iloc[i]['window_start'])
            window_end_idx = int(features_df.iloc[i]['window_end'])
            
            # 获取时间范围
            win_time_start = raw_data.iloc[window_start_idx]['relative_time']
            win_time_end = raw_data.iloc[window_end_idx-1]['relative_time']
            
            fig.add_vrect(
                x0=win_time_start,
                x1=win_time_end,
                fillcolor=custom_colors.get(label, '#808080'),
                opacity=0.2,
                line_width=0,
                annotation_text=behavior_labels.get(label, f"行为 {label}"),
                annotation_position="top left"
            )
        
        # 更新布局
        fig.update_layout(
            title=f"全局行为分布时间轴 - 时间段 {window_count} ({window_count*5-5}~{window_count*5}分钟) ({method} 检测)",
            xaxis_title="相对时间 (秒)",
            yaxis=dict(
                title='时延 (ms)',
                titlefont=dict(color='blue'),
                tickfont=dict(color='blue'),
                range=[0, 2000]
            ),
            yaxis2=dict(
                title='丢包率',
                titlefont=dict(color='red'),
                tickfont=dict(color='red'),
                overlaying='y',
                side='right',
                range=[0, 1.1]
            ),
            legend=dict(x=0, y=1),
            hovermode='x unified',
            height=600,
            width=1000
        )
        
        # 保存为HTML文件
        html_file = output_dir / f"{method}_timeline_window_{window_count}_interactive.html"
        fig.write_html(html_file, full_html=True, include_plotlyjs='cdn')
        logger.debug(f"保存交互式时间轴图：{html_file}")
    
    def _save_interactive_transition_graph(
        self,
        transition_matrix: np.ndarray,
        labels: List[int],
        method: str,
        output_dir: Path
    ) -> None:
        """保存交互式转移图到HTML文件"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            logger.error("Plotly 库未安装，无法生成交互式转移图")
            return
        
        # 确保labels是numpy数组
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        num_states = len(unique_labels)
        
        # 创建标签到索引的映射
        label_to_index = {label: i for i, label in enumerate(sorted(unique_labels))}
        
        # 创建交互式热力图
        fig = go.Figure(data=go.Heatmap(
            z=transition_matrix,
            x=[f"行为 {label}" for label in sorted(unique_labels)],
            y=[f"行为 {label}" for label in sorted(unique_labels)],
            hoverongaps=False,
            text=transition_matrix.round(3),
            texttemplate='%{text}',
            colorscale='YlGnBu',
            hovertemplate='从 行为 %{y} 到 行为 %{x}: <br>转移概率 = %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"{method} 规则检测 - 行为转移矩阵",
            xaxis_title="目标状态",
            yaxis_title="起始状态",
            width=800,
            height=800,
            hovermode='closest'
        )
        
        # 保存为HTML文件
        html_file = output_dir / "interactive_transition_graph.html"
        fig.write_html(html_file, full_html=True, include_plotlyjs='cdn')
        logger.debug(f"保存交互式转移图：{html_file}")
    
    def _save_typical_samples(
        self,
        raw_data: pd.DataFrame,
        features_df: pd.DataFrame,
        labels: List[int],
        method: str,
        output_dir: Path
    ) -> None:
        """保存典型样本可视化到PNG文件"""
        # 确保labels是numpy数组
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        
        # 创建典型样本图
        plt.figure(figsize=(15, 2 * len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            # 找到该标签的所有样本
            label_mask = labels == label
            label_indices = np.where(label_mask)[0]
            
            if len(label_indices) > 0:
                # 选择第一个样本作为典型样本
                sample_idx = label_indices[0]
                
                # 获取窗口的起始和结束索引
                window_start = int(features_df.iloc[sample_idx]["window_start"])
                window_end = int(features_df.iloc[sample_idx]["window_end"])
                
                # 提取原始数据
                window_data = raw_data.iloc[window_start:window_end]
                
                # 创建子图
                ax = plt.subplot(len(unique_labels), 1, i + 1)
                
                # 绘制时延
                ax.plot(
                    window_data["timestamp"],
                    window_data["delay"],
                    "b-",
                    linewidth=2,
                    label="时延 (ms)",
                )
                ax.set_xlabel("时间")
                ax.set_ylabel("时延 (ms)", color="b")
                ax.tick_params("y", colors="b")
                ax.grid(True, alpha=0.3)
                
                # 绘制丢包率
                ax2 = ax.twinx()
                ax2.plot(
                    window_data["timestamp"],
                    window_data["loss_rate"],
                    "r-",
                    linewidth=2,
                    label="丢包率",
                )
                ax2.set_ylabel("丢包率", color="r")
                ax2.tick_params("y", colors="r")
                ax2.set_ylim(0, 1.1)
                
                # 添加标题和图例
                ax.set_title(f"行为 {label} - 典型样本")
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
                
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 保存为PNG文件
        typical_file = output_dir / "typical_samples.png"
        plt.savefig(typical_file, dpi=150, bbox_inches="tight")
        plt.close()
        logger.debug(f"保存典型样本：{typical_file}")

    def _save_transition_metrics(
        self,
        transition_matrix: np.ndarray,
        labels: List[int],
        method: str,
        output_dir: Path
    ) -> None:
        """保存转移指标可视化到PNG文件"""
        # 确保labels是numpy数组
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        num_states = len(unique_labels)
        
        # 计算转移指标
        # 1. 出度分布（每个状态转移到其他状态的数量）
        out_degree = np.sum(transition_matrix > 0, axis=1)
        
        # 2. 入度分布（每个状态被其他状态转移到的数量）
        in_degree = np.sum(transition_matrix > 0, axis=0)
        
        # 3. 转移熵（每个状态的不确定性）
        transition_entropy = np.zeros(num_states)
        for i in range(num_states):
            row = transition_matrix[i, :]
            row = row[row > 0]  # 只考虑非零概率
            if len(row) > 0:
                transition_entropy[i] = -np.sum(row * np.log2(row))
        
        # 4. 平均转移概率（每个状态的平均转移概率）
        avg_transition_prob = np.mean(transition_matrix, axis=1)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 子图1: 出度分布
        axes[0, 0].bar(range(num_states), out_degree)
        axes[0, 0].set_xlabel('行为状态')
        axes[0, 0].set_ylabel('出度')
        axes[0, 0].set_title('每个行为状态的出度分布')
        axes[0, 0].set_xticks(range(num_states))
        axes[0, 0].set_xticklabels([f'行为 {label}' for label in sorted(unique_labels)], rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 子图2: 入度分布
        axes[0, 1].bar(range(num_states), in_degree)
        axes[0, 1].set_xlabel('行为状态')
        axes[0, 1].set_ylabel('入度')
        axes[0, 1].set_title('每个行为状态的入度分布')
        axes[0, 1].set_xticks(range(num_states))
        axes[0, 1].set_xticklabels([f'行为 {label}' for label in sorted(unique_labels)], rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 子图3: 转移熵分布
        axes[1, 0].bar(range(num_states), transition_entropy)
        axes[1, 0].set_xlabel('行为状态')
        axes[1, 0].set_ylabel('转移熵')
        axes[1, 0].set_title('每个行为状态的转移熵分布')
        axes[1, 0].set_xticks(range(num_states))
        axes[1, 0].set_xticklabels([f'行为 {label}' for label in sorted(unique_labels)], rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 子图4: 平均转移概率分布
        axes[1, 1].bar(range(num_states), avg_transition_prob)
        axes[1, 1].set_xlabel('行为状态')
        axes[1, 1].set_ylabel('平均转移概率')
        axes[1, 1].set_title('每个行为状态的平均转移概率')
        axes[1, 1].set_xticks(range(num_states))
        axes[1, 1].set_xticklabels([f'行为 {label}' for label in sorted(unique_labels)], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 调整布局
        plt.suptitle(f'{method} 规则检测 - 行为转移指标', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # 保存为PNG文件
        metrics_file = output_dir / "transition_metrics.png"
        plt.savefig(metrics_file, dpi=150, bbox_inches="tight")
        plt.close()
        logger.debug(f"保存转移指标可视化：{metrics_file}")

    def _generate_separation_table(
        self, separation_metrics: Dict, feature_names: List[str]
    ) -> str:
        """生成分离度指标的HTML表格"""
        table_rows = []
        for label in separation_metrics["label_means"].keys():
            means = separation_metrics["label_means"][label]
            row = f"<tr><td>{label}</td>"
            row += "".join([f"<td>{mean:.4f}</td>" for mean in means])
            row += "</tr>"
            table_rows.append(row)
        return "".join(table_rows)

    def visualize_raw_data_samples(
        self,
        raw_data: pd.DataFrame,
        features_df: pd.DataFrame,
        labels: np.ndarray,
        method: str,
    ) -> str:
        """可视化每个行为类别的原始数据样本"""
        import random

        html_snippets = ["<h2>原始数据样本可视化</h2>"]

        # Ensure labels is a numpy array
        labels = np.array(labels)

        # 定义行为标签映射
        behavior_labels = {
            0: "稳定",
            1: "弱突发",
            2: "强突发",
            3: "瞬时峰值",
            4: "高延迟无丢包",
            5: "高丢包低延迟",
            6: "强突发高延迟",
            7: "复杂网络行为"
        }

        # Get unique labels
        unique_labels = np.unique(labels)

        for label in unique_labels:
            # Find all windows with this label
            label_mask = labels == label
            label_indices = np.where(label_mask)[0]

            if len(label_indices) > 0:
                # Get actual behavior name
                behavior_name = behavior_labels.get(label, f"行为 {label}")
                
                # Add category title
                html_snippets.append(
                    f"<h3>{behavior_name} - 原始数据样本</h3>"
                )
                html_snippets.append(
                    '<div style="display: flex; flex-wrap: wrap; gap: 20px;">'
                )

                # Select 2 random samples for this label
                num_samples = min(2, len(label_indices))
                sample_indices = random.sample(list(label_indices), num_samples)

                for i, sample_idx in enumerate(sample_indices):
                    # Get window start and end indices from features_df
                    window_start = int(features_df.iloc[sample_idx]["window_start"])
                    window_end = int(features_df.iloc[sample_idx]["window_end"])

                    # Extract raw data for this window
                    window_data = raw_data.iloc[window_start:window_end]

                    # Create dual-axis plot for delay and loss_rate
                    plt.figure(figsize=(10, 6))

                    # Plot delay on primary y-axis，统一最大值2000ms
                    ax1 = plt.subplot(111)
                    ax1.plot(
                        window_data["timestamp"],
                        window_data["delay"],
                        "b-",
                        linewidth=2,
                        label="时延 (ms)",
                    )
                    ax1.set_xlabel("时间")
                    ax1.set_ylabel("时延 (ms)", color="b")
                    ax1.tick_params("y", colors="b")
                    ax1.set_ylim(0, 2000)  # 统一时延最大值2000ms
                    ax1.grid(True, alpha=0.3)

                    # Plot loss_rate on secondary y-axis，优化0值显示
                    ax2 = ax1.twinx()
                    ax2.plot(
                        window_data["timestamp"],
                        window_data["loss_rate"],
                        "r-",
                        linewidth=2,
                        label="丢包率",
                    )
                    ax2.set_ylabel("丢包率", color="r")
                    ax2.tick_params("y", colors="r")
                    ax2.set_ylim(-0.01, 1.1)  # 从-0.01开始，优化0值显示
                    
                    # 为丢包率添加轻微抖动，使0值更容易区分
                    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)

                    # Add title and legend
                    method_name = "规则" if method == "rule" else method
                    plt.title(f"{behavior_name} - 样本 {i + 1} ({method_name} 行为检测)")
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    # Save plot to buffer
                    buffer = BytesIO()
                    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
                    buffer.seek(0)
                    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
                    plt.close()

                    # Add to HTML snippets
                    html_snippets.append('<div style="flex: 1; min-width: 300px;">')
                    html_snippets.append(f"<h4>样本 {i + 1}</h4>")
                    html_snippets.append(
                        f'<img src="data:image/png;base64,{img_base64}" alt="{behavior_name} 原始数据样本 {i + 1}" style="width: 100%; height: auto;">'
                    )
                    html_snippets.append("</div>")

                # Close the flex container
                html_snippets.append("</div>")

        return "".join(html_snippets)

    def visualize_label_timeline(
        self,
        raw_data: pd.DataFrame,
        features_df: pd.DataFrame,
        labels: np.ndarray,
        method: str,
    ) -> str:
        """可视化全局行为分布的标签时间轴，按6分钟拆分，每张图包含原始时延和丢包率（双Y轴）"""
        html_snippets = ["<h2>全局行为分布时间轴</h2>"]
        html_snippets.append("<p>注：时间轴按6分钟拆分，每张图展示一个时间段的网络行为，时延最大值统一为2000ms，X轴显示相对时间间隔（秒）</p>")

        # 确保labels是numpy数组
        labels = np.array(labels)
        
        # 为不同标签创建颜色映射 - 使用高对比度颜色组合
        unique_labels = np.unique(labels)
        
        # 使用高对比度的自定义颜色映射，确保在各种条件下都能清晰区分
        # 稳定：淡绿色，弱突发：橙色，瞬时峰值：亮黄色，强突发：红色，其他：高对比度颜色
        custom_colors = {
            0: '#4CAF50',  # 稳定 - 深绿色
            1: '#FF9800',  # 弱突发 - 橙色
            2: '#F44336',  # 强突发 - 红色
            3: '#FFEB3B',  # 瞬时峰值 - 黄色
            4: '#2196F3',  # 高延迟无丢包 - 蓝色
            5: '#9C27B0',  # 其他 - 紫色
            6: '#FF5722',  # 其他 - 深橙色
            7: '#00BCD4'   # 其他 - 青色
        }
        
        # 验证所有标签都在定义的颜色映射范围内
        assert np.all(np.isin(labels, list(custom_colors.keys()))), f"发现未定义的 label: {np.setdiff1d(labels, list(custom_colors.keys()))}"
        
        # 创建自定义颜色映射函数
        def color_map(label):
            return custom_colors.get(label, '#808080')  # 默认灰色
        
        # 定义行为标签映射
        behavior_labels = {
            0: "稳定",
            1: "弱突发",
            2: "强突发",
            3: "瞬时峰值",
            4: "高延迟无丢包",
            5: "高丢包低延迟",
            6: "强突发高延迟",
            7: "复杂网络行为"
        }
        
        # 确保timestamp列是datetime类型
        raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])
        
        # 为每个文件单独处理时间轴
        unique_files = raw_data['file_path'].unique()
        
        for current_file_path in unique_files:
            # 获取当前文件的所有数据
            file_data = raw_data[raw_data['file_path'] == current_file_path].copy()
            if file_data.empty:
                continue
            
            # 计算当前文件的相对时间（从0开始，单位：秒）
            file_data['relative_time'] = (file_data['timestamp'] - file_data['timestamp'].min()).dt.total_seconds()
            
            # 获取当前文件的时间范围
            file_start_time = file_data['timestamp'].min()
            file_end_time = file_data['timestamp'].max()
            
            # 计算总时长，按6分钟拆分
            time_delta = pd.Timedelta(minutes=6)
            current_time = file_start_time
            
            # 用于记录已处理的时间窗口
            window_count = 0
            
            while current_time < file_end_time:
                window_count += 1
                window_end = current_time + time_delta
                
                # 确保最后一个窗口不超过结束时间
                if window_end > file_end_time:
                    window_end = file_end_time
                
                # 获取当前时间窗口对应的原始数据
                window_mask = (file_data['timestamp'] >= current_time) & (file_data['timestamp'] < window_end)
                window_data = file_data[window_mask]
                
                # 如果当前窗口没有数据，跳过
                if len(window_data) == 0:
                    current_time = window_end
                    continue
                
                # 创建当前窗口的图表
                fig, ax1 = plt.subplots(figsize=(15, 8))
                
                # 重采样数据，进一步减少数据点数量，解决曲线过于密集问题
                # 使用500个样本点，保持趋势的同时大幅减少线条密度
                if len(window_data) > 500:
                    # 使用更稀疏的均匀采样
                    step = len(window_data) // 500
                    sampled_data = window_data.iloc[::step]
                else:
                    sampled_data = window_data
                
                # 确保数据按照relative_time排序
                sampled_data = sampled_data.sort_values('relative_time')
                
                # 对数据进行平滑处理，使用移动平均减少噪声
                # 对时延和丢包率分别应用移动平均
                sampled_data['delay_smooth'] = sampled_data['delay'].rolling(window=3, min_periods=1).mean()
                sampled_data['loss_rate_smooth'] = sampled_data['loss_rate'].rolling(window=3, min_periods=1).mean()
                
                # 调试：查看时延数据的统计信息
                logger.debug(f"文件 {current_file_path} 窗口 {window_count} 数据点数量: {len(sampled_data)}")
                logger.debug(f"relative_time范围: {sampled_data['relative_time'].min()} - {sampled_data['relative_time'].max()}")
                logger.debug(f"delay范围: {sampled_data['delay'].min()} - {sampled_data['delay'].max()}")
                logger.debug(f"loss_rate范围: {sampled_data['loss_rate'].min()} - {sampled_data['loss_rate'].max()}")
                
                # 统一时延范围为0-2000ms
                y_min = 0
                y_max = 2000
                
                # 直接在ax1上绘制行为标签色块，不使用额外的twiny()轴
                # 先绘制色块，确保在最底层
                for i, label in enumerate(labels):
                    window_start_idx = int(features_df.iloc[i]['window_start'])
                    window_end_idx = int(features_df.iloc[i]['window_end'])
                    window_end_idx = min(window_end_idx, len(raw_data))
                    
                    win_time_start = raw_data.iloc[window_start_idx]['timestamp']
                    win_time_end = raw_data.iloc[window_end_idx-1]['timestamp']
                    
                    if win_time_end < current_time or win_time_start >= window_end:
                        continue
                    
                    # 检查该行为窗口是否属于当前文件
                    behavior_window_data = raw_data.iloc[window_start_idx:window_end_idx]
                    if current_file_path not in behavior_window_data['file_path'].values:
                        continue
                    
                    overlap_start = max(win_time_start, current_time)
                    overlap_end = min(win_time_end, window_end)
                    
                    # 使用当前文件的起始时间计算相对时间
                    overlap_start_relative = (overlap_start - file_start_time).total_seconds()
                    overlap_end_relative = (overlap_end - file_start_time).total_seconds()
                    
                    # 直接在ax1上绘制色块，提高透明度到0.4，并添加边框
                    ax1.fill_between(
                        x=[overlap_start_relative, overlap_end_relative],
                        y1=y_min,
                        y2=y_max,
                        facecolor=color_map(label),
                        alpha=0.4,  # 提高透明度到0.4
                        edgecolor=color_map(label),  # 添加同色边框
                        linewidth=0.5,  # 边框宽度
                        zorder=0  # 确保在最底层
                    )
                
                # 设置Y轴范围
                ax1.set_ylim(y_min, y_max)
                
                # 绘制平滑后的时延曲线（左Y轴），使用更细的线条
                delay_line, = ax1.plot(sampled_data['relative_time'], sampled_data['delay_smooth'], 'b-', linewidth=1.5, alpha=0.8, label='时延 (ms)')
                
                # 确保Y轴有明确的数值标记
                y_ticks = np.linspace(y_min, y_max, 5)
                ax1.set_yticks(y_ticks)
                # 设置Y轴刻度标签为明确的数值
                ax1.set_yticklabels([f'{tick:.0f}' for tick in y_ticks], fontweight='bold', fontsize=12, color='b')
                
                # 设置Y轴标签和样式，确保可见
                ax1.set_ylabel('时延 (ms)', color='b', fontsize=14, fontweight='bold', rotation=90, labelpad=20)
                ax1.tick_params('y', colors='b', labelsize=12, width=3, length=15, direction='out')
                
                # 确保Y轴轴线可见
                ax1.spines['left'].set_visible(True)
                ax1.spines['left'].set_color('b')
                ax1.spines['left'].set_linewidth(3)
                
                # 设置网格线
                ax1.grid(True, alpha=0.3, linestyle='--')
                
                # Create right Y-axis for loss rate, 使用更细的线条
                ax2 = ax1.twinx()
                # 绘制平滑后的丢包率，使用更细的实线
                loss_line, = ax2.plot(sampled_data['relative_time'], sampled_data['loss_rate_smooth'], 'r-', linewidth=1.5, alpha=0.8, label='丢包率')
                ax2.set_ylabel('丢包率', color='r', fontsize=14, fontweight='bold')
                ax2.tick_params('y', colors='r', labelsize=12, width=3, length=15, direction='out')
                ax2.set_ylim(-0.01, 1.1)  # 丢包率从-0.01开始，到1.1结束
                
                # 确保右侧Y轴轴线可见
                ax2.spines['right'].set_visible(True)
                ax2.spines['right'].set_color('r')
                ax2.spines['right'].set_linewidth(3)
                
                # 为丢包率添加参考线，增强可读性
                ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
                
                # 获取两个Y轴的图例
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                
                # 创建行为标签图例
                behavior_handles = []
                behavior_labels_legend = []
                for label in unique_labels:
                    handle = plt.Rectangle((0, 0), 1, 1, facecolor=color_map(label), alpha=0.4, edgecolor=color_map(label), linewidth=0.5)
                    behavior_handles.append(handle)
                    behavior_labels_legend.append(behavior_labels.get(label, f"行为 {label}"))
                
                # 合并所有图例
                all_handles = lines1 + lines2 + behavior_handles
                all_labels = labels1 + labels2 + behavior_labels_legend
                
                # 在右上角添加合并后的图例，2列显示
                ax1.legend(all_handles, all_labels, loc='upper right', fontsize=10, ncol=2)
                
                # 设置图表标题和标签
                # 处理不同操作系统的路径分隔符
                import os
                file_name = os.path.basename(current_file_path)
                
                # 计算当前文件在该窗口的相对时间
                minutes_in_file = (current_time - file_start_time).total_seconds() / 60
                
                ax1.set_title(f"全局行为分布时间轴 - {file_name} - 对应文件的第 {int(minutes_in_file)}~{int(minutes_in_file+6)} 分钟 ({method} 规则检测)", fontsize=14)
                ax1.set_xlabel("相对时间 (秒)", fontsize=12)
                
                # 改进时间刻度，增加秒级刻度显示
                # 设置X轴刻度密度，每100秒显示一个刻度
                x_min = sampled_data['relative_time'].min()
                x_max = sampled_data['relative_time'].max()
                x_ticks = np.arange(x_min, x_max + 1, 100)  # 每100秒一个刻度
                ax1.set_xticks(x_ticks)
                
                # 设置X轴刻度标签格式为秒级
                ax1.set_xticklabels([f'{tick:.0f}s' for tick in x_ticks], fontsize=10)
                
                # 优化X轴刻度标签旋转角度，避免重叠
                plt.xticks(rotation=45, ha='right')
                
                # 调整布局
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.25)  # 增加底部边距，避免X轴标签被截断
                
                # 保存图表到缓冲区
                buffer = BytesIO()
                plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
                plt.close()
                
                # 添加到HTML片段
                html_snippets.append('<div class="timeline-window" style="margin: 20px 0; padding: 10px; background: #f8f9fa; border-radius: 8px;">')
                html_snippets.append(f'<h3>{file_name} - 对应文件的第 {int(minutes_in_file)}~{int(minutes_in_file+6)} 分钟</h3>')
                html_snippets.append(f'<img src="data:image/png;base64,{img_base64}" alt="全局行为分布时间轴 - 时间段 {window_count}" style="width: 100%; height: auto;">')
                html_snippets.append('</div>')
                
                # 移动到下一个时间窗口
                current_time = window_end
        
        return "".join(html_snippets)
