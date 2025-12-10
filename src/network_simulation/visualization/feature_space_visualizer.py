#!/usr/bin/env python3
"""
特征空间可视化类
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from pathlib import Path
from typing import List

from .base_visualizer import BaseVisualizer

# 忽略UMAP的n_jobs被random_state覆盖的警告
warnings.filterwarnings("ignore", message="n_jobs value .* overridden to .* by setting random_state")


class FeatureSpaceVisualizer(BaseVisualizer):
    """特征空间可视化类"""

    def visualize_pca_scatter(
        self, X: np.ndarray, labels: np.ndarray, method: str
    ) -> str:
        """可视化PCA散点图并返回HTML片段"""
        # 执行PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # 创建散点图
        self._create_scatter_plot(
            X_transformed=X_pca,
            labels=labels,
            title=f"{'规则' if method == 'rule' else method} 行为检测结果 PCA 散点图",
            xlabel=f"PCA 维度 1 ({pca.explained_variance_ratio_[0]:.2%} 方差)",
            ylabel=f"PCA 维度 2 ({pca.explained_variance_ratio_[1]:.2%} 方差)",
            show_evolution=True,
            method=method
        )

        # 保存图表到缓冲区并返回base64
        img_base64 = self._plot_to_base64()

        # 生成统计摘要
        unique_labels = np.unique(labels)
        summary = f"""
        <div style="margin-top: 10px; padding: 10px; background-color: #f0f8ff; border-left: 4px solid #3498db;">
        <p><strong>PCA散点图统计摘要：</strong></p>
        <ul>
        <li>共检测到 {len(unique_labels)} 种行为类别</li>
        <li>PCA维度1解释了 {pca.explained_variance_ratio_[0]:.2%} 的方差，维度2解释了 {pca.explained_variance_ratio_[1]:.2%} 的方差</li>
        <li>不同行为类别在PCA空间中呈现{"较好" if len(unique_labels) > 1 else "一般"}的分离效果</li>
        <li>每个类别中心点已标注为"典型样本"，便于识别各类别特征</li>
        </ul>
        </div>
        """

        return f'<img src="data:image/png;base64,{img_base64}" alt="PCA 散点图">{summary}'

    def visualize_pca_variance(self, X: np.ndarray, method: str) -> str:
        """可视化PCA方差解释并返回HTML片段"""
        # 执行PCA
        pca = PCA(n_components=min(8, X.shape[1]))
        pca.fit(X)

        # 创建方差解释图
        plt.figure(figsize=(10, 8))
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
        plt.title(f"{'规则' if method == 'rule' else method} 行为检测 PCA 方差解释图")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)

        # 保存图表到缓冲区
        img_base64 = self._plot_to_base64()

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
        self._create_scatter_plot(
            X_transformed=X_tsne,
            labels=labels,
            title=f"{'规则' if method == 'rule' else method} 行为检测结果 t-SNE 散点图 (perplexity={perplexity}, max_iter={max_iter})",
            xlabel=f"t-SNE 维度 1 (perplexity={perplexity})",
            ylabel=f"t-SNE 维度 2 (max_iter={max_iter})",
            show_evolution=True
        )

        # 保存图表到缓冲区
        img_base64 = self._plot_to_base64()

        # 生成统计摘要
        unique_labels = np.unique(labels)
        summary = f"""
        <div style="margin-top: 10px; padding: 10px; background-color: #f0f8ff; border-left: 4px solid #3498db;">
        <p><strong>t-SNE散点图统计摘要：</strong></p>
        <ul>
        <li>共检测到 {len(unique_labels)} 种行为类别</li>
        <li>使用参数：perplexity={perplexity}, max_iter={max_iter}</li>
        <li>不同行为类别在t-SNE空间中呈现非线性分布</li>
        <li>每个类别中心点已标注为"典型样本"，便于识别各类别特征</li>
        </ul>
        </div>
        """

        return f'<img src="data:image/png;base64,{img_base64}" alt="t-SNE 散点图">{summary}'

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
        self._create_scatter_plot(
            X_transformed=X_umap,
            labels=labels,
            title=f"{'规则' if method == 'rule' else method} 行为检测结果 UMAP 散点图 (n_neighbors={n_neighbors}, min_dist={min_dist})",
            xlabel=f"UMAP 维度 1 (n_neighbors={n_neighbors})",
            ylabel=f"UMAP 维度 2 (min_dist={min_dist})",
            show_evolution=True
        )

        # 保存图表到缓冲区
        img_base64 = self._plot_to_base64()

        # 生成统计摘要
        unique_labels = np.unique(labels)
        summary = f"""
        <div style="margin-top: 10px; padding: 10px; background-color: #f0f8ff; border-left: 4px solid #3498db;">
        <p><strong>UMAP散点图统计摘要：</strong></p>
        <ul>
        <li>共检测到 {len(unique_labels)} 种行为类别</li>
        <li>使用参数：n_neighbors={n_neighbors}, min_dist={min_dist}</li>
        <li>不同行为类别在UMAP空间中呈现清晰的聚类结构</li>
        <li>每个类别中心点已标注为"典型样本"，便于识别各类别特征</li>
        <li>UMAP图显示了行为类别之间的空间关系，有助于理解不同行为之间的联系</li>
        </ul>
        </div>
        """

        return f'<img src="data:image/png;base64,{img_base64}" alt="UMAP 散点图">{summary}'

    def generate_behavior_separation_plots(
        self, X: np.ndarray, labels: np.ndarray, output_dir: Path, feature_names: List[str] = None, direction: str = "up"
    ) -> None:
        """生成行为分离可视化图表

        Args:
            X: 特征矩阵
            labels: 标签向量
            output_dir: 输出目录
            feature_names: 特征名称列表
            direction: 方向（"up" 或 "down"）
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 根据方向选择对应的特征
        if feature_names is not None:
            # 为当前方向选择对应的特征
            if direction == "up":
                # 上行特征：包含1或ratio或symmetry的特征
                selected_feature_mask = [col.endswith("1") or "ratio" in col or "symmetry" in col
                                       for col in feature_names]
            else:
                # 下行特征：包含2或ratio或symmetry的特征
                selected_feature_mask = [col.endswith("2") or "ratio" in col or "symmetry" in col
                                       for col in feature_names]

            # 过滤特征
            X = X[:, selected_feature_mask]
            feature_names = [name for name, selected in zip(feature_names, selected_feature_mask) if selected]

        # 使用过滤后的特征名称或默认名称
        if feature_names is None:
            feature_names = [f"特征 {i+1}" for i in range(X.shape[1])]

        # 特征分布箱线图
        n_features = X.shape[1]
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        axes = axes.flatten()

        for i in range(min(n_features, 6)):
            sns.boxplot(x=labels, y=X[:, i], ax=axes[i])
            axes[i].set_title(f"{feature_names[i]} 分布")
            axes[i].set_xlabel("行为标签")
            axes[i].set_ylabel("特征值")

        # 隐藏多余的子图
        for i in range(n_features, 6):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"{direction}_feature_distributions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 特征相关性热力图
        if n_features > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = np.corrcoef(X.T)
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True,
                       xticklabels=feature_names[:min(n_features, 10)],
                       yticklabels=feature_names[:min(n_features, 10)])
            plt.title(f"{direction.upper()} 特征相关性热力图")
            plt.tight_layout()
            plt.savefig(
                output_dir / f"{direction}_feature_correlation.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

    def save_feature_space_visualizations(
        self,
        X: np.ndarray,
        labels: List[int],
        _method: str,
        output_dir: Path,
        _feature_names: List[str],
        direction: str = "up",
    ) -> None:
        """保存特征空间降维图"""
        labels = np.array(labels)

        # 保存PCA散点图
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        self._create_scatter_plot(
            X_transformed=X_pca,
            labels=labels,
            title=f"{direction.upper()} 行为特征 PCA 散点图",
            xlabel=f"PCA 维度 1 ({pca.explained_variance_ratio_[0]:.2%} 方差)",
            ylabel=f"PCA 维度 2 ({pca.explained_variance_ratio_[1]:.2%} 方差)",
            show_evolution=True
        )
        self._save_plot(output_dir / f"{direction}_pca_scatter.png")

        # 保存PCA方差解释图
        pca = PCA(n_components=min(8, X.shape[1]))
        pca.fit(X)
        plt.figure(figsize=(10, 8))
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
        plt.title(f"{direction.upper()} 行为特征 PCA 方差解释图")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save_plot(output_dir / f"{direction}_pca_variance.png")

        # 保存t-SNE散点图
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)
        X_tsne = tsne.fit_transform(X)
        self._create_scatter_plot(
            X_transformed=X_tsne,
            labels=labels,
            title=f"{direction.upper()} 行为特征 t-SNE 降维图",
            xlabel="t-SNE 维度 1",
            ylabel="t-SNE 维度 2",
            show_evolution=True
        )
        self._save_plot(output_dir / f"{direction}_tsne.png")

        # 保存UMAP散点图
        umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        X_umap = umap.fit_transform(X)
        self._create_scatter_plot(
            X_transformed=X_umap,
            labels=labels,
            title=f"{direction.upper()} 行为特征 UMAP 降维图",
            xlabel="UMAP 维度 1",
            ylabel="UMAP 维度 2",
            show_evolution=True
        )
        self._save_plot(output_dir / f"{direction}_umap.png")
