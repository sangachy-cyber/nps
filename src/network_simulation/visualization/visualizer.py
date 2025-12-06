#!/usr/bin/env python3
"""
Network Behavior Visualization Module
Responsible for visualizing network behavior patterns and evaluation results
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


class Visualizer:
    """Visualizes network behavior patterns and evaluation results"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set Chinese font support, prefer WenQuanYi Zen Hei on Linux
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    def visualize_evaluation_results(self, results: Dict) -> str:
        """Visualize evaluation results and return HTML snippet"""
        # This will be implemented later
        return "<h2>评估结果可视化</h2>"

    def visualize_pca_scatter(
        self, X: np.ndarray, labels: np.ndarray, method: str
    ) -> str:
        """Visualize PCA scatter plot and return HTML snippet"""
        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Create scatter plot
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(labels)
        colors = sns.color_palette("hsv", len(unique_labels))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=[color],
                label=f"类别 {label}",
                alpha=0.6,
            )

        plt.title(f"{method} 聚类结果 PCA 散点图")
        plt.xlabel(f"主成分 1 ({pca.explained_variance_ratio_[0]:.2%} 方差)")
        plt.ylabel(f"主成分 2 ({pca.explained_variance_ratio_[1]:.2%} 方差)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()

        return f'<img src="data:image/png;base64,{img_base64}" alt="PCA 散点图">'

    def visualize_pca_variance(self, X: np.ndarray, method: str) -> str:
        """Visualize PCA variance explained and return HTML snippet"""
        # Perform PCA
        pca = PCA(n_components=min(8, X.shape[1]))
        pca.fit(X)

        # Create variance explained plot
        plt.figure(figsize=(10, 6))
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        plt.bar(
            range(1, len(explained_variance) + 1),
            explained_variance,
            alpha=0.7,
            align="center",
            label="单个主成分方差",
        )
        plt.step(
            range(1, len(cumulative_variance) + 1),
            cumulative_variance,
            where="mid",
            label="累计方差",
        )
        plt.ylabel("方差解释比例")
        plt.xlabel("主成分数量")
        plt.title(f"{method} 聚类 PCA 方差解释图")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)

        # Save plot to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()

        return f'<img src="data:image/png;base64,{img_base64}" alt="PCA 方差解释图">'

    def visualize_tsne_scatter(
        self, X: np.ndarray, labels: np.ndarray, method: str
    ) -> str:
        """Visualize t-SNE scatter plot and return HTML snippet"""
        # Perform t-SNE
        perplexity = 30
        max_iter = 300
        tsne = TSNE(
            n_components=2, random_state=42, perplexity=perplexity, max_iter=max_iter
        )
        X_tsne = tsne.fit_transform(X)

        # Create scatter plot
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

        plt.title(
            f"{method} 聚类结果 t-SNE 散点图 (perplexity={perplexity}, max_iter={max_iter})"
        )
        plt.xlabel(f"t-SNE 维度 1 (perplexity={perplexity})")
        plt.ylabel(f"t-SNE 维度 2 (max_iter={max_iter})")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()

        return f'<img src="data:image/png;base64,{img_base64}" alt="t-SNE 散点图">'

    def visualize_umap_scatter(
        self, X: np.ndarray, labels: np.ndarray, method: str
    ) -> str:
        """Visualize UMAP scatter plot and return HTML snippet"""
        # Perform UMAP
        n_neighbors = 15
        min_dist = 0.1
        umap = UMAP(
            n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=min_dist
        )
        X_umap = umap.fit_transform(X)

        # Create scatter plot
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(labels)
        colors = sns.color_palette("hsv", len(unique_labels))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(
                X_umap[mask, 0],
                X_umap[mask, 1],
                c=[color],
                label=f"类别 {label}",
                alpha=0.6,
            )

        plt.title(
            f"{method} 聚类结果 UMAP 散点图 (n_neighbors={n_neighbors}, min_dist={min_dist})"
        )
        plt.xlabel(f"UMAP 维度 1 (n_neighbors={n_neighbors})")
        plt.ylabel(f"UMAP 维度 2 (min_dist={min_dist})")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()

        return f'<img src="data:image/png;base64,{img_base64}" alt="UMAP 散点图">'

    def visualize_feature_distribution(
        self, X: np.ndarray, labels: np.ndarray, feature_names: List[str], method: str
    ) -> str:
        """Visualize feature distributions and return HTML snippet"""
        unique_labels = np.unique(labels)
        n_features = len(feature_names)

        # Create subplots
        fig, axes = plt.subplots(
            n_features, 1, figsize=(12, 3 * n_features), sharex=False
        )

        for i, (feature_name, ax) in enumerate(zip(feature_names, axes)):
            for label in unique_labels:
                mask = labels == label
                sns.histplot(
                    X[mask, i], ax=ax, label=f"类别 {label}", alpha=0.5, kde=True
                )
            ax.set_title(f"{feature_name} 分布")
            ax.set_xlabel(feature_name)
            ax.set_ylabel("频率")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()

        return f'<img src="data:image/png;base64,{img_base64}" alt="特征分布直方图">'

    def visualize_correlation_heatmap(
        self, X: np.ndarray, feature_names: List[str], method: str
    ) -> str:
        """Visualize correlation heatmap and return HTML snippet"""
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)

        # Create heatmap
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
        plt.title(f"{method} 聚类 特征相关性热力图")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save plot to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()

        return f'<img src="data:image/png;base64,{img_base64}" alt="特征相关性热力图">'

    def visualize_transition_matrix(
        self, transition_matrix: np.ndarray, method: str
    ) -> str:
        """Visualize transition matrix heatmap and return HTML snippet"""
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            transition_matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=[f"类别 {i}" for i in range(transition_matrix.shape[1])],
            yticklabels=[f"类别 {i}" for i in range(transition_matrix.shape[0])],
            square=True,
        )
        plt.title(f"{method} 聚类 状态转移矩阵热力图")
        plt.xlabel("下一状态")
        plt.ylabel("当前状态")
        plt.tight_layout()

        # Save plot to buffer
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
        """Generate comprehensive HTML report"""
        method = results["method"]

        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>网络行为模式发现报告 - {method} 聚类</title>
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
                <h2>聚类方法：{method}</h2>
                
                <div class="section">
                    <h2>1. 聚类质量评估</h2>
                    <table class="metrics-table">
                        <tr>
                            <th>指标名称</th>
                            <th>数值</th>
                            <th>说明</th>
                        </tr>
                        <tr>
                            <td>轮廓系数</td>
                            <td>{results["metrics"]["silhouette_score"]:.4f}</td>
                            <td>>0.5 表示良好聚类</td>
                        </tr>
                        <tr>
                            <td>Calinski-Harabasz 指数</td>
                            <td>{results["metrics"]["calinski_harabasz_score"]:.2f}</td>
                            <td>值越高，簇间分离越好</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>2. 行为转移质量评估</h2>
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
                    <h2>3. 降维可视化</h2>
                    <p>不同降维方法的对比：</p>
                    <p>- PCA：线性降维，保留最大方差</p>
                    <p>- t-SNE：非线性降维，专注于局部结构，适合可视化高维数据</p>
                    <p>- UMAP：非线性降维，同时保留局部和全局结构，运行速度更快</p>
                    
                    <h3>3.1 PCA 散点图</h3>
                    {self.visualize_pca_scatter(X, np.array(results["labels"]), method)}
                    
                    <h3>3.2 PCA 方差解释图</h3>
                    {self.visualize_pca_variance(X, method)}
                    
                    <h3>3.3 t-SNE 散点图</h3>
                    {self.visualize_tsne_scatter(X, np.array(results["labels"]), method)}
                    
                    <h3>3.4 UMAP 散点图</h3>
                    {self.visualize_umap_scatter(X, np.array(results["labels"]), method)}
                </div>
                
                <div class="section">
                    <h2>4. 特征分析</h2>
                    <h3>4.1 特征分布直方图</h3>
                    {self.visualize_feature_distribution(X, np.array(results["labels"]), feature_names, method)}
                    
                    <h3>4.2 特征相关性热力图</h3>
                    {self.visualize_correlation_heatmap(X, feature_names, method)}
                </div>
                
                <div class="section">
                    <h2>5. 行为转移分析</h2>
                    <h3>5.1 状态转移矩阵热力图</h3>
                    {self.visualize_transition_matrix(np.array(results["transition_matrix"]), method)}
                </div>
                
                <div class="section">
                    <h2>6. 行为分离度分析</h2>
                    <h3>6.1 各行为类别特征均值</h3>
                    <table class="metrics-table">
                        <tr>
                            <th>行为类别</th>
                            {"".join([f"<th>{name}</th>" for name in feature_names])}
                        </tr>
                        {self._generate_separation_table(results["separation_metrics"], feature_names)}
                    </table>
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

        # Save HTML report
        report_file = self.output_dir / f"behavior_pattern_report_{method}.html"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"HTML报告已生成：{report_file}")

    def _generate_separation_table(
        self, separation_metrics: Dict, feature_names: List[str]
    ) -> str:
        """Generate HTML table for separation metrics"""
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
        """Visualize raw data samples for each behavior category"""
        import random

        html_snippets = ["<h2>原始数据样本可视化</h2>"]

        # Ensure labels is a numpy array
        labels = np.array(labels)

        # Get unique labels
        unique_labels = np.unique(labels)

        for label in unique_labels:
            # Find all windows with this label
            label_mask = labels == label
            label_indices = np.where(label_mask)[0]

            if len(label_indices) > 0:
                # Add category title
                html_snippets.append(
                    f"<h3>行为类别 {label} - 原始数据样本 (随机选择4个)</h3>"
                )
                html_snippets.append(
                    '<div style="display: flex; flex-wrap: wrap; gap: 20px;">'
                )

                # Select up to 4 random samples for this label
                num_samples = min(4, len(label_indices))
                sample_indices = random.sample(list(label_indices), num_samples)

                for i, sample_idx in enumerate(sample_indices):
                    # Get window start and end indices from features_df
                    window_start = features_df.iloc[sample_idx]["window_start"]
                    window_end = features_df.iloc[sample_idx]["window_end"]

                    # Extract raw data for this window
                    window_data = raw_data.iloc[window_start:window_end]

                    # Create dual-axis plot for delay and loss_rate
                    plt.figure(figsize=(10, 6))

                    # Plot delay on primary y-axis
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
                    ax1.grid(True, alpha=0.3)

                    # Plot loss_rate on secondary y-axis
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
                    ax2.set_ylim(0, 1.1)  # Loss rate is between 0 and 1

                    # Add title and legend
                    plt.title(f"行为类别 {label} - 样本 {i + 1} ({method} 聚类)")
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
                        f'<img src="data:image/png;base64,{img_base64}" alt="行为类别 {label} 原始数据样本 {i + 1}" style="width: 100%; height: auto;">'
                    )
                    html_snippets.append("</div>")

                # Close the flex container
                html_snippets.append("</div>")

        return "".join(html_snippets)
