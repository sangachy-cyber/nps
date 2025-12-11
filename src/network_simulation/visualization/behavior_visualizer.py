#!/usr/bin/env python3
"""
行为可视化类
负责生成行为相关的可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
import random

from network_simulation.utils.logger import get_logger
from .base_visualizer import BaseVisualizer
from config import BEHAVIOR_CATEGORY_MAP, BEHAVIOR_CATEGORY_COLORS

# 初始化日志记录器
logger = get_logger(__name__)


class BehaviorVisualizer(BaseVisualizer):
    """行为可视化类

    负责生成网络行为相关的可视化图表，包括行为转移矩阵、特征分布、相关性热力图
    以及行为时间轴等多种可视化效果，用于分析和展示网络行为模式。
    """

    def generate_transition_plots(
        self, transition_matrix: np.ndarray, output_dir: Path
    ) -> None:
        """生成行为转移可视化图表

        生成行为转移矩阵热力图和转移指标可视化图表，用于分析网络行为模式之间的转移关系。

        Args:
            transition_matrix (np.ndarray): 行为转移矩阵，shape (n_behaviors, n_behaviors)
            output_dir (Path): 输出目录路径

        Examples:
            >>> from network_simulation.visualization.behavior_visualizer import BehaviorVisualizer
            >>> from pathlib import Path
            >>> import numpy as np
            >>> behavior_visualizer = BehaviorVisualizer(Path("output/visualizations"))
            >>> transition_matrix = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
            >>> output_dir = Path("output/visualizations/transition")
            >>> behavior_visualizer.generate_transition_plots(transition_matrix, output_dir)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 转移矩阵热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            transition_matrix, annot=True, fmt=".3f", cmap="YlGnBu", square=True
        )
        plt.xlabel("目标状态")
        plt.ylabel("起始状态")
        plt.title("行为转移矩阵热力图")
        plt.tight_layout()
        plt.savefig(
            output_dir / "transition_matrix_heatmap.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 转移指标可视化
        num_states = transition_matrix.shape[0]

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
        axes[0, 0].set_xlabel("行为状态")
        axes[0, 0].set_ylabel("出度")
        axes[0, 0].set_title("每个行为状态的出度分布")
        axes[0, 0].set_xticks(range(num_states))
        axes[0, 0].set_xticklabels(
            [f"行为 {i}" for i in range(num_states)], rotation=45
        )
        axes[0, 0].grid(True, alpha=0.3)

        # 子图2: 入度分布
        axes[0, 1].bar(range(num_states), in_degree)
        axes[0, 1].set_xlabel("行为状态")
        axes[0, 1].set_ylabel("入度")
        axes[0, 1].set_title("每个行为状态的入度分布")
        axes[0, 1].set_xticks(range(num_states))
        axes[0, 1].set_xticklabels(
            [f"行为 {i}" for i in range(num_states)], rotation=45
        )
        axes[0, 1].grid(True, alpha=0.3)

        # 子图3: 转移熵分布
        axes[1, 0].bar(range(num_states), transition_entropy)
        axes[1, 0].set_xlabel("行为状态")
        axes[1, 0].set_ylabel("转移熵")
        axes[1, 0].set_title("每个行为状态的转移熵分布")
        axes[1, 0].set_xticks(range(num_states))
        axes[1, 0].set_xticklabels(
            [f"行为 {i}" for i in range(num_states)], rotation=45
        )
        axes[1, 0].grid(True, alpha=0.3)

        # 子图4: 平均转移概率分布
        axes[1, 1].bar(range(num_states), avg_transition_prob)
        axes[1, 1].set_xlabel("行为状态")
        axes[1, 1].set_ylabel("平均转移概率")
        axes[1, 1].set_title("每个行为状态的平均转移概率")
        axes[1, 1].set_xticks(range(num_states))
        axes[1, 1].set_xticklabels(
            [f"行为 {i}" for i in range(num_states)], rotation=45
        )
        axes[1, 1].grid(True, alpha=0.3)

        # 调整布局
        plt.suptitle("行为转移指标", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(output_dir / "transition_metrics.png", dpi=300, bbox_inches="tight")
        plt.close()

    def visualize_feature_distribution(
        self, X: np.ndarray, labels: np.ndarray, feature_names: List[str], _method: str
    ) -> str:
        """可视化特征分布并返回HTML片段

        可视化不同行为模式的特征分布，生成包含特征直方图的HTML片段。

        Args:
            X (np.ndarray): 特征矩阵，shape (n_samples, n_features)
            labels (np.ndarray): 行为标签数组，shape (n_samples,)
            feature_names (List[str]): 特征名称列表
            _method (str): 可视化方法名称（未使用）

        Returns:
            str: 包含特征分布直方图的HTML片段

        Examples:
            >>> from network_simulation.visualization.behavior_visualizer import BehaviorVisualizer
            >>> from pathlib import Path
            >>> import numpy as np
            >>> behavior_visualizer = BehaviorVisualizer(Path("output/visualizations"))
            >>> X = np.random.randn(100, 3)
            >>> labels = np.random.randint(0, 3, 100)
            >>> feature_names = ["delay_std", "loss_nonzero_ratio", "max_congestion_run"]
            >>> html = behavior_visualizer.visualize_feature_distribution(X, labels, feature_names, "rule")
            >>> print(html[:100])  # 显示HTML片段的前100个字符
        """
        unique_labels = np.unique(labels)
        n_features = len(feature_names)

        # 创建子图
        fig, axes = plt.subplots(
            n_features, 1, figsize=(12, 3 * n_features), sharex=False
        )

        for i, (feature_name, ax) in enumerate(zip(feature_names, axes)):
            for label in unique_labels:
                mask = labels == label
                behavior_name = BEHAVIOR_CATEGORY_MAP.get(label, f"行为 {label}")
                color = BEHAVIOR_CATEGORY_COLORS.get(label, "#808080")
                sns.histplot(
                    X[mask, i],
                    ax=ax,
                    label=behavior_name,
                    alpha=0.5,
                    kde=True,
                    color=color,
                )
            ax.set_title(f"{feature_name} 分布")
            ax.set_xlabel(feature_name)
            ax.set_ylabel("频率")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        img_base64 = self._plot_to_base64()
        return f'<img src="data:image/png;base64,{img_base64}" alt="特征分布直方图">'

    def visualize_correlation_heatmap(
        self, X: np.ndarray, feature_names: List[str], method: str
    ) -> str:
        """可视化特征相关性热力图并返回HTML片段

        生成特征相关性热力图，用于分析网络行为特征之间的相互关系。

        Args:
            X (np.ndarray): 特征矩阵，shape (n_samples, n_features)
            feature_names (List[str]): 特征名称列表
            method (str): 可视化方法名称

        Returns:
            str: 包含特征相关性热力图的HTML片段

        Examples:
            >>> from network_simulation.visualization.behavior_visualizer import BehaviorVisualizer
            >>> from pathlib import Path
            >>> import numpy as np
            >>> behavior_visualizer = BehaviorVisualizer(Path("output/visualizations"))
            >>> X = np.random.randn(100, 5)
            >>> feature_names = ["delay_std", "loss_nonzero_ratio", "max_congestion_run", "delay_trend", "delay_acf"]
            >>> html = behavior_visualizer.visualize_correlation_heatmap(X, feature_names, "rule")
            >>> print(html[:100])  # 显示HTML片段的前100个字符
        """
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
        plt.title(f"{'规则' if method == 'rule' else method} 行为检测 特征相关性热力图")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        img_base64 = self._plot_to_base64()
        return f'<img src="data:image/png;base64,{img_base64}" alt="特征相关性热力图">'

    def visualize_transition_matrix(
        self, transition_matrix: np.ndarray, method: str
    ) -> str:
        """可视化状态转移矩阵热力图并返回HTML片段

        生成状态转移矩阵热力图，用于分析网络行为模式之间的转移关系。

        Args:
            transition_matrix (np.ndarray): 状态转移矩阵，shape (n_behaviors, n_behaviors)
            method (str): 可视化方法名称

        Returns:
            str: 包含状态转移矩阵热力图的HTML片段

        Examples:
            >>> from network_simulation.visualization.behavior_visualizer import BehaviorVisualizer
            >>> from pathlib import Path
            >>> import numpy as np
            >>> behavior_visualizer = BehaviorVisualizer(Path("output/visualizations"))
            >>> transition_matrix = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
            >>> html = behavior_visualizer.visualize_transition_matrix(transition_matrix, "rule")
            >>> print(html[:100])  # 显示HTML片段的前100个字符
        """
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
        plt.title(
            f"{'规则' if method == 'rule' else method} 行为检测 状态转移矩阵热力图"
        )
        plt.xlabel("下一行为")
        plt.ylabel("当前行为")
        plt.tight_layout()

        img_base64 = self._plot_to_base64()
        return (
            f'<img src="data:image/png;base64,{img_base64}" alt="状态转移矩阵热力图">'
        )

    def save_behavior_samples(
        self,
        raw_data: pd.DataFrame,
        features_df: pd.DataFrame,
        labels: List[int],
        method: str,
        output_dir: Path,
        direction: str = "up",
    ) -> None:
        """保存每类2个行为样本图

        为每种行为模式保存2个典型样本的可视化图表，展示网络行为的具体表现。

        Args:
            raw_data (pd.DataFrame): 原始网络数据
            features_df (pd.DataFrame): 特征数据框，包含窗口信息
            labels (List[int]): 行为标签列表
            method (str): 可视化方法名称
            output_dir (Path): 输出目录路径
            direction (str, optional): 方向，"up"表示上行数据，"down"表示下行数据，默认为"up"

        Examples:
            >>> from network_simulation.visualization.behavior_visualizer import BehaviorVisualizer
            >>> from pathlib import Path
            >>> import pandas as pd
            >>> import numpy as np
            >>> behavior_visualizer = BehaviorVisualizer(Path("output/visualizations"))
            >>> # 创建模拟数据
            >>> raw_data = pd.DataFrame({
            ...     "timestamp": pd.date_range("2023-01-01", periods=1000, freq="ms"),
            ...     "delay1": np.random.randn(1000) * 10 + 50,
            ...     "delay2": np.random.randn(1000) * 10 + 50,
            ...     "loss_rate1": np.random.rand(1000) * 0.1,
            ...     "loss_rate2": np.random.rand(1000) * 0.1,
            ...     "file_path": ["test_file.csv"] * 1000
            ... })
            >>> features_df = pd.DataFrame({
            ...     "window_start": [0, 100, 200, 300, 400],
            ...     "window_end": [100, 200, 300, 400, 500]
            ... })
            >>> labels = [0, 1, 2, 0, 1]
            >>> output_dir = Path("output/visualizations/samples")
            >>> behavior_visualizer.save_behavior_samples(raw_data, features_df, labels, "rule", output_dir, "up")
        """
        labels = np.array(labels)
        unique_labels = np.unique(labels)

        for label in unique_labels:
            label_mask = labels == label
            label_indices = np.where(label_mask)[0]

            if len(label_indices) > 0:
                # 从配置中获取实际行为名称
                behavior_name = BEHAVIOR_CATEGORY_MAP.get(label, f"行为 {label}")

                # 每个类别挑2个样本
                num_samples = min(2, len(label_indices))
                sample_indices = random.sample(list(label_indices), num_samples)

                for i, sample_idx in enumerate(sample_indices):
                    self._save_single_behavior_sample(
                        raw_data,
                        features_df,
                        sample_idx,
                        label,
                        behavior_name,
                        method,
                        output_dir,
                        i,
                        direction,
                    )

    def _save_single_behavior_sample(
        self,
        raw_data: pd.DataFrame,
        features_df: pd.DataFrame,
        sample_idx: int,
        label: int,
        behavior_name: str,
        method: str,
        output_dir: Path,
        sample_number: int,
        direction: str = "up",
    ) -> None:
        """保存单个行为样本图

        保存单个网络行为样本的可视化图表，展示该样本的时延和丢包率变化情况。

        Args:
            raw_data (pd.DataFrame): 原始网络数据
            features_df (pd.DataFrame): 特征数据框，包含窗口信息
            sample_idx (int): 样本索引
            label (int): 行为标签
            behavior_name (str): 行为名称
            method (str): 可视化方法名称
            output_dir (Path): 输出目录路径
            sample_number (int): 样本编号
            direction (str, optional): 方向，"up"表示上行数据，"down"表示下行数据，默认为"up"
        """
        # Get window start and end indices from features_df
        window_start_idx = int(features_df.iloc[sample_idx]["window_start"])
        window_end_idx = int(features_df.iloc[sample_idx]["window_end"])

        # Ensure we don't exceed raw_data bounds
        window_start_idx = max(0, window_start_idx)
        window_end_idx = min(len(raw_data), window_end_idx)

        # Get time range for this window from raw_data
        win_time_start = raw_data.iloc[window_start_idx]["timestamp"]
        win_time_end = raw_data.iloc[window_end_idx - 1]["timestamp"]

        # Extract raw data for this window using time range
        window_data = raw_data[
            (raw_data["timestamp"] >= win_time_start)
            & (raw_data["timestamp"] <= win_time_end)
        ]

        # 跳过空窗口
        if window_data.empty:
            logger.warning(f"跳过空窗口: {win_time_start} - {win_time_end}")
            return

        # 获取样本的文件来源
        file_name = self._get_file_name(window_data)

        # 计算时间范围（秒）
        start_time = window_data["timestamp"].iloc[0]
        end_time = window_data["timestamp"].iloc[-1]
        time_duration = (end_time - start_time).total_seconds()

        # 为时延和丢包率创建双轴图表
        plt.figure(figsize=(10, 6))

        # Plot delay on primary y-axis，统一最大值2000ms
        ax1 = plt.subplot(111)
        # 同时绘制上下行时延
        ax1.plot(
            window_data["timestamp"],
            window_data["delay1"],
            "b-",
            linewidth=2,
            label="上行时延 (ms)",
        )
        ax1.plot(
            window_data["timestamp"],
            window_data["delay2"],
            "c-",
            linewidth=2,
            label="下行时延 (ms)",
        )
        ax1.set_xlabel("时间")
        ax1.set_ylabel("时延 (ms)", color="b")
        ax1.tick_params("y", colors="b")
        ax1.set_ylim(0, 2000)  # 统一时延最大值2000ms
        ax1.grid(True, alpha=0.3)

        # Plot loss_rate on secondary y-axis，优化0值显示
        ax2 = ax1.twinx()
        # 同时绘制上下行丢包率
        ax2.plot(
            window_data["timestamp"],
            window_data["loss_rate1"],
            "r-",
            linewidth=2,
            label="上行丢包率",
        )
        ax2.plot(
            window_data["timestamp"],
            window_data["loss_rate2"],
            "m-",
            linewidth=2,
            label="下行丢包率",
        )
        ax2.set_ylabel("丢包率", color="r")
        ax2.tick_params("y", colors="r")
        ax2.set_ylim(-0.01, 1.1)  # 从-0.01开始，优化0值显示

        # 为丢包率添加轻微抖动，使0值更容易区分
        ax2.axhline(y=0, color="r", linestyle="--", alpha=0.3)

        # 添加标题和图例
        method_name = "规则" if method == "rule" else method
        plt.title(
            f"{direction.upper()} {behavior_name} - 样本 {sample_number + 1} ({method_name} 检测)\n文件: {file_name} | 时间范围: {time_duration:.1f}秒"
        )
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.xticks(rotation=45)
        plt.tight_layout()

        # 保存图表到文件，带有方向前缀
        sample_file = (
            output_dir / f"{direction}_behavior_{label}_sample_{sample_number + 1}.png"
        )
        self._save_plot(sample_file)

    def _get_file_name(self, window_data: pd.DataFrame) -> str:
        """获取窗口数据的文件名

        Args:
            window_data (pd.DataFrame): 窗口数据

        Returns:
            str: 文件名

        Examples:
            >>> from network_simulation.visualization.behavior_visualizer import BehaviorVisualizer
            >>> from pathlib import Path
            >>> import pandas as pd
            >>> behavior_visualizer = BehaviorVisualizer(Path("output/visualizations"))
            >>> window_data = pd.DataFrame({"file_path": ["/path/to/test_file.csv"]})
            >>> file_name = behavior_visualizer._get_file_name(window_data)
            >>> print(file_name)  # 输出: test_file.csv
        """
        if "file_path" in window_data.columns:
            file_path = window_data["file_path"].iloc[0]
            return file_path.split("/")[-1]  # 只保留文件名
        return "unknown_file"

    def save_label_timeline(
        self,
        raw_data: pd.DataFrame,
        features_df: pd.DataFrame,
        labels: List[int],
        method: str,
        output_dir: Path,
        direction: str = "up",
    ) -> None:
        """保存标签时间轴图

        生成网络行为标签的时间轴图，展示不同行为模式在时间上的分布情况。

        Args:
            raw_data (pd.DataFrame): 原始网络数据
            features_df (pd.DataFrame): 特征数据框，包含窗口信息
            labels (List[int]): 行为标签列表
            method (str): 可视化方法名称
            output_dir (Path): 输出目录路径
            direction (str, optional): 方向，"up"表示上行数据，"down"表示下行数据，默认为"up"

        Examples:
            >>> from network_simulation.visualization.behavior_visualizer import BehaviorVisualizer
            >>> from pathlib import Path
            >>> import pandas as pd
            >>> import numpy as np
            >>> behavior_visualizer = BehaviorVisualizer(Path("output/visualizations"))
            >>> # 创建模拟数据
            >>> raw_data = pd.DataFrame({
            ...     "timestamp": pd.date_range("2023-01-01", periods=1000, freq="ms"),
            ...     "delay1": np.random.randn(1000) * 10 + 50,
            ...     "delay2": np.random.randn(1000) * 10 + 50,
            ...     "loss_rate1": np.random.rand(1000) * 0.1,
            ...     "loss_rate2": np.random.rand(1000) * 0.1,
            ...     "file_path": ["test_file.csv"] * 1000
            ... })
            >>> features_df = pd.DataFrame({
            ...     "window_start": [0, 100, 200, 300, 400],
            ...     "window_end": [100, 200, 300, 400, 500]
            ... })
            >>> labels = [0, 1, 2, 0, 1]
            >>> output_dir = Path("output/visualizations/timeline")
            >>> behavior_visualizer.save_label_timeline(raw_data, features_df, labels, "rule", output_dir, "up")
        """
        # 确保标签是numpy数组
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
            7: "复杂网络行为",
        }

        # Ensure timestamp column is datetime type
        raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"])

        # 为每个文件单独处理时间轴
        unique_files = raw_data["file_path"].unique()

        for current_file_path in unique_files:
            # 获取当前文件的所有数据
            file_data = raw_data[raw_data["file_path"] == current_file_path].copy()
            if file_data.empty:
                continue

            # 计算当前文件的相对时间（从0开始，单位：秒）
            file_data["relative_time"] = (
                file_data["timestamp"] - file_data["timestamp"].min()
            ).dt.total_seconds()

            # 获取当前文件的时间范围
            file_start_time = file_data["timestamp"].min()
            file_end_time = file_data["timestamp"].max()

            # 基于整个文件的时间范围创建一个完整的时间轴图，不进行切片
            # 这样可以确保行为块完整显示，不会被切成两半
            window_count = 1
            window_data = file_data.copy()
            current_time = file_start_time
            window_end = file_end_time

            # 为整个文件创建图表
            self._create_timeline_window_plot(
                window_data,
                file_data,
                current_time,
                window_end,
                file_start_time,
                labels,
                features_df,
                current_file_path,
                method,
                behavior_labels,
                window_count,
                output_dir,
                direction,
            )

    def _create_timeline_window_plot(
        self,
        window_data: pd.DataFrame,
        file_data: pd.DataFrame,
        current_time: pd.Timestamp,
        window_end: pd.Timestamp,
        file_start_time: pd.Timestamp,
        labels: np.ndarray,
        features_df: pd.DataFrame,
        current_file_path: str,
        method: str,
        behavior_labels: Dict[int, str],
        window_count: int,
        output_dir: Path,
        direction: str = "up",
    ) -> None:
        """创建时间轴窗口图

        创建单个时间轴窗口的可视化图表，展示网络行为随时间的变化情况。

        Args:
            window_data (pd.DataFrame): 窗口数据
            file_data (pd.DataFrame): 文件数据
            current_time (pd.Timestamp): 当前时间
            window_end (pd.Timestamp): 窗口结束时间
            file_start_time (pd.Timestamp): 文件开始时间
            labels (np.ndarray): 行为标签数组
            features_df (pd.DataFrame): 特征数据框
            current_file_path (str): 当前文件路径
            method (str): 可视化方法名称
            behavior_labels (Dict[int, str]): 行为标签映射
            window_count (int): 窗口计数
            output_dir (Path): 输出目录路径
            direction (str, optional): 方向，"up"表示上行数据，"down"表示下行数据，默认为"up"
        """
        # 如需要则重采样数据，将数据点减少到500个
        if len(window_data) > 500:
            step = len(window_data) // 500
            sampled_data = window_data.iloc[::step]
        else:
            sampled_data = window_data

        # 确保数据按相对时间排序
        sampled_data = sampled_data.sort_values("relative_time")

        # 使用移动平均平滑数据
        # 处理上下行时延
        sampled_data["delay1_smooth"] = (
            sampled_data["delay1"].rolling(window=3, min_periods=1).mean()
        )
        sampled_data["delay2_smooth"] = (
            sampled_data["delay2"].rolling(window=3, min_periods=1).mean()
        )
        # 处理上下行丢包率
        sampled_data["loss_rate1_smooth"] = (
            sampled_data["loss_rate1"].rolling(window=3, min_periods=1).mean()
        )
        sampled_data["loss_rate2_smooth"] = (
            sampled_data["loss_rate2"].rolling(window=3, min_periods=1).mean()
        )

        # Create custom color mapping function using config colors
        def color_map(label):
            return BEHAVIOR_CATEGORY_COLORS.get(label, "#808080")  # 默认灰色

        # Unified delay range: 0-2000ms
        y_min = 0
        y_max = 2000

        # Create plot
        fig, ax1 = plt.subplots(figsize=(15, 8))

        # Plot behavior label blocks
        self._plot_behavior_blocks(
            ax1,
            labels,
            features_df,
            current_time,
            window_end,
            current_file_path,
            file_start_time,
            raw_data=file_data,
            color_map=color_map,
            y_min=y_min,
            y_max=y_max,
        )

        # 绘制平滑后的上下行时延曲线
        ax1.plot(
            sampled_data["relative_time"],
            sampled_data["delay1_smooth"],
            "b-",
            linewidth=1.5,
            alpha=0.8,
            label="上行时延 (ms)",
        )
        ax1.plot(
            sampled_data["relative_time"],
            sampled_data["delay2_smooth"],
            "c-",
            linewidth=1.5,
            alpha=0.8,
            label="下行时延 (ms)",
        )

        # 设置Y轴样式
        self._setup_delay_axis(ax1, y_min, y_max)

        # 为丢包率创建右侧Y轴
        ax2 = ax1.twinx()
        # 绘制平滑后的上下行丢包率
        ax2.plot(
            sampled_data["relative_time"],
            sampled_data["loss_rate1_smooth"],
            "r-",
            linewidth=1.5,
            alpha=0.8,
            label="上行丢包率",
        )
        ax2.plot(
            sampled_data["relative_time"],
            sampled_data["loss_rate2_smooth"],
            "m-",
            linewidth=1.5,
            alpha=0.8,
            label="下行丢包率",
        )
        self._setup_loss_rate_axis(ax2)

        # 为丢包率添加参考线，提高可读性
        ax2.axhline(y=0, color="r", linestyle="--", alpha=0.3)

        # 合并图例
        unique_labels = np.unique(labels)
        self._combine_legends(ax1, ax2, unique_labels, color_map, behavior_labels)

        # 设置图表标题和标签
        file_name = current_file_path.split("/")[-1]  # 只保留文件名
        minutes_in_file = (current_time - file_start_time).total_seconds() / 60

        ax1.set_title(
            f"{direction.upper()} 行为分布时间轴 - {file_name} - 对应文件的第 {int(minutes_in_file)}~{int(minutes_in_file+6)} 分钟 ({method} 检测)",
            fontsize=14,
        )
        ax1.set_xlabel("相对时间 (秒)", fontsize=12)

        # 优化时间刻度，每100秒显示一个刻度
        x_min = sampled_data["relative_time"].min()
        x_max = sampled_data["relative_time"].max()
        x_ticks = np.arange(x_min, x_max + 1, 100)  # 每100秒一个刻度
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels([f"{tick:.0f}s" for tick in x_ticks], fontsize=10)

        # 优化X轴刻度标签旋转
        plt.xticks(rotation=45, ha="right")

        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)  # 增加底部边距

        # Save plot to file with file-specific window count and direction prefix
        import os

        timeline_file = (
            output_dir
            / f"{direction}_{os.path.basename(current_file_path)}_timeline_window_{window_count}.png"
        )
        self._save_plot(timeline_file)

    def _plot_behavior_blocks(
        self,
        ax1: plt.Axes,
        labels: np.ndarray,
        features_df: pd.DataFrame,
        current_time: pd.Timestamp,
        window_end: pd.Timestamp,
        current_file_path: str,
        file_start_time: pd.Timestamp,
        raw_data: pd.DataFrame,
        color_map: callable,
        y_min: float,
        y_max: float,
    ) -> None:
        """绘制行为块

        在时间轴图上绘制不同行为模式的彩色块，直观展示行为随时间的变化。

        Args:
            ax1 (plt.Axes): 绘图坐标轴
            labels (np.ndarray): 行为标签数组
            features_df (pd.DataFrame): 特征数据框
            current_time (pd.Timestamp): 当前时间
            window_end (pd.Timestamp): 窗口结束时间
            current_file_path (str): 当前文件路径
            file_start_time (pd.Timestamp): 文件开始时间
            raw_data (pd.DataFrame): 原始数据
            color_map (callable): 颜色映射函数
            y_min (float): Y轴最小值
            y_max (float): Y轴最大值
        """
        for i, label in enumerate(labels):
            # Get window start and end indices
            window_start_idx = int(features_df.iloc[i]["window_start"])
            window_end_idx = int(features_df.iloc[i]["window_end"])

            # Ensure window indices don't exceed raw_data length
            window_start_idx = max(window_start_idx, 0)
            window_end_idx = min(window_end_idx, len(raw_data))

            # Skip if window is invalid (start >= end)
            if window_start_idx >= window_end_idx:
                continue

            # Get time range for this behavior window
            win_time_start = raw_data.iloc[window_start_idx]["timestamp"]
            win_time_end = raw_data.iloc[window_end_idx - 1]["timestamp"]

            # Only plot if behavior window is within current window
            if win_time_end < current_time or win_time_start >= window_end:
                continue

            # Check if this behavior window belongs to current file
            behavior_window_data = raw_data.iloc[window_start_idx:window_end_idx]
            if current_file_path not in behavior_window_data["file_path"].values:
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
            )

    def _setup_delay_axis(self, ax1: plt.Axes, y_min: float, y_max: float) -> None:
        """设置时延轴样式

        设置时间轴图中时延轴的样式，包括坐标轴范围、刻度、标签和网格线。

        Args:
            ax1 (plt.Axes): 绘图坐标轴
            y_min (float): Y轴最小值
            y_max (float): Y轴最大值
        """
        ax1.set_ylim(y_min, y_max)

        # 确保Y轴有明确的数值标记
        y_ticks = np.linspace(y_min, y_max, 5)
        ax1.set_yticks(y_ticks)
        # 设置Y轴刻度标签为明确的数值
        ax1.set_yticklabels(
            [f"{tick:.0f}" for tick in y_ticks],
            fontweight="bold",
            fontsize=12,
            color="b",
        )

        # 设置Y轴标签和样式，确保可见
        ax1.set_ylabel(
            "时延 (ms)",
            color="b",
            fontsize=14,
            fontweight="bold",
            rotation=90,
            labelpad=20,
        )
        ax1.tick_params(
            "y", colors="b", labelsize=12, width=3, length=15, direction="out"
        )

        # 确保Y轴轴线可见
        ax1.spines["left"].set_visible(True)
        ax1.spines["left"].set_color("b")
        ax1.spines["left"].set_linewidth(3)

        # 设置网格线
        ax1.grid(True, alpha=0.3, linestyle="--")

    def _setup_loss_rate_axis(self, ax2: plt.Axes) -> None:
        """设置丢包率轴样式

        设置时间轴图中丢包率轴的样式，包括坐标轴范围、刻度、标签和轴线。

        Args:
            ax2 (plt.Axes): 绘图坐标轴
        """
        ax2.set_ylabel("丢包率", color="r", fontsize=14, fontweight="bold")
        ax2.tick_params(
            "y", colors="r", labelsize=12, width=3, length=15, direction="out"
        )
        ax2.set_ylim(
            -0.01, 1.1
        )  # Loss rate from -0.01 to 1.1, optimize 0 value display

        # 确保右侧Y轴轴线可见
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color("r")
        ax2.spines["right"].set_linewidth(3)

    def _combine_legends(
        self,
        ax1: plt.Axes,
        ax2: plt.Axes,
        unique_labels: np.ndarray,
        color_map: callable,
        behavior_labels: Dict[int, str],
    ) -> None:
        """合并图例

        合并时间轴图中两个Y轴的图例，并添加行为标签的图例，便于理解图表内容。

        Args:
            ax1 (plt.Axes): 第一个Y轴坐标轴（时延轴）
            ax2 (plt.Axes): 第二个Y轴坐标轴（丢包率轴）
            unique_labels (np.ndarray): 唯一行为标签数组
            color_map (callable): 颜色映射函数
            behavior_labels (Dict[int, str]): 行为标签映射
        """
        # Get legends from both Y-axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        # Create behavior label legend handles
        behavior_handles = []
        behavior_labels_legend = []
        for label in unique_labels:
            handle = plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=color_map(label),
                alpha=1.0,
                edgecolor=color_map(label),
                linewidth=0.5,
            )
            behavior_handles.append(handle)
            behavior_labels_legend.append(behavior_labels.get(label, f"行为 {label}"))

        # Combine all legends
        all_handles = lines1 + lines2 + behavior_handles
        all_labels = labels1 + labels2 + behavior_labels_legend

        # Add combined legend at upper right
        ax1.legend(all_handles, all_labels, loc="upper right", fontsize=10, ncol=2)
