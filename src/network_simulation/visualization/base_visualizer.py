#!/usr/bin/env python3
"""
基础可视化类，提供通用可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import base64
from io import BytesIO

from ..utils.logger import get_logger
from config import BEHAVIOR_CATEGORY_MAP, BEHAVIOR_CATEGORY_COLORS


logger = get_logger(__name__)


class BaseVisualizer:
    """基础可视化类，提供通用可视化功能"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置中文支持，Linux系统优先使用文泉驿正黑
        plt.rcParams["font.sans-serif"] = [
            "WenQuanYi Zen Hei",
            "SimHei",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False

    def _save_plot(self, output_path: Path, dpi: int = 150) -> None:
        """保存图表到指定路径"""
        plt.savefig(output_path, format="png", dpi=dpi, bbox_inches="tight")
        plt.close()
        logger.debug(f"保存图表：{output_path}")

    def _create_scatter_plot(
        self,
        X_transformed: np.ndarray,
        labels: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        show_evolution: bool = True,
        method: str = None
    ) -> None:
        """创建通用散点图"""
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(labels)
        category_centers = {}

        # 绘制散点并计算中心点
        for label in unique_labels:
            mask = labels == label
            behavior_name = BEHAVIOR_CATEGORY_MAP.get(label, f"行为 {label}")
            color = BEHAVIOR_CATEGORY_COLORS.get(label, "#808080")
            plt.scatter(
                X_transformed[mask, 0],
                X_transformed[mask, 1],
                c=[color],
                label=behavior_name,
                alpha=0.6,
            )

            # 计算并存储该类别的中心点
            center = np.mean(X_transformed[mask], axis=0)
            category_centers[label] = center

            # 标注典型样本位置
            plt.annotate(
                f"{behavior_name}\n典型样本",
                xy=center,
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc=color, alpha=0.7),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                fontsize=9,
                ha="left"
            )

        # 添加演化方向箭头 - 仅在规则方法下显示
        if show_evolution and len(category_centers) >= 2 and method == "rule":
            self._add_evolution_arrows(category_centers)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    def _add_evolution_arrows(self, category_centers: Dict[int, np.ndarray]) -> None:
        """添加演化方向箭头"""
        # 按类别ID排序，假设ID越小代表越稳定
        sorted_labels = sorted(category_centers.keys())

        for i in range(len(sorted_labels) - 1):
            start_label = sorted_labels[i]
            end_label = sorted_labels[i + 1]

            start_center = category_centers[start_label]
            end_center = category_centers[end_label]

            # 绘制演化方向箭头
            plt.annotate(
                "",
                xy=end_center,
                xytext=start_center,
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle="arc3,rad=0.2",
                    color="darkred",
                    linewidth=2,
                    alpha=0.7,
                    zorder=5
                )
            )

            # 添加演化方向标签
            mid_point = (start_center + end_center) / 2
            label_offset = 0.05 * (end_center - start_center)
            label_pos = mid_point + label_offset
            start_name = BEHAVIOR_CATEGORY_MAP.get(start_label, f"行为 {start_label}")
            end_name = BEHAVIOR_CATEGORY_MAP.get(end_label, f"行为 {end_label}")
            plt.text(
                label_pos[0],
                label_pos[1],
                f"{start_name} → {end_name}",
                fontsize=9,
                fontweight="normal",
                color="darkred",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="lightyellow",
                    alpha=0.8,
                    edgecolor="darkred",
                    linewidth=1,
                    zorder=6
                )
            )

    def _plot_to_base64(self, dpi: int = 150) -> str:
        """将当前图表转换为base64编码"""
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()
        return img_base64
