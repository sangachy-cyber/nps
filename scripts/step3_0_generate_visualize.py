#!/usr/bin/env python3
"""
生成样本并可视化对比
"""

import sys
import os

sys.path.append(os.path.abspath("src"))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from network_simulation.feature_extraction.feature_extractor import FeatureExtractor
from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
from network_simulation.condition_generation.diffusion_model import (
    ConditionDiffusionModel,
)
from network_simulation.condition_generation.constraint_injector import (
    ConstraintInjector,
)
# 导入日志模块
from network_simulation.utils.logger import get_logger

# 获取日志记录器
logger = get_logger(__name__)

# 设置中文支持，兼容Windows、MacOS和Linux，Linux系统优先使用文泉驿正黑
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class GenerateAndVisualize:
    """生成网络状态样本并可视化对比类

    该类负责使用训练好的扩散模型生成网络状态样本，并与原始数据进行可视化对比。

    Attributes:
        input_data_path: 输入数据文件路径
        output_path: 输出目录路径
        device: 运行设备（CPU或GPU）
        feature_extractor: 特征提取器实例
        pattern_identifier: 模式识别器实例
        df: 处理后的原始数据DataFrame
        features_df: 提取的特征DataFrame
        behavior_ids: 行为标签数组
        valid_loss_values: 合法丢包值列表
        constraint_injector: 约束注入器实例
        selected_df: 选择的连续样本DataFrame
        selected_behavior_ids: 选择的连续样本对应的行为标签
        model: 条件扩散模型实例
        model_path: 模型文件路径
    """

    def __init__(
        self, input_data_path: Path, output_path: Path, model_path: Path = None
    ):
        """初始化生成和可视化实例

        Args:
            input_data_path: 输入数据文件路径
            output_path: 输出目录路径
            model_path: 预训练模型文件路径，可选
        """
        self.input_data_path = input_data_path
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 初始化设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")

        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor()

        # 初始化模式识别器
        self.pattern_identifier = PatternIdentifier(method="rule")

        # 加载和处理数据
        self.df = None
        self.features_df = None
        self.behavior_ids = None
        self.valid_loss_values = None

        # 初始化约束注入器
        self.constraint_injector = None

        # 选择连续6000个样本
        self.selected_df = None
        self.selected_behavior_ids = None

        # 初始化扩散模型
        self.model = None
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = output_path / "train_results/diffusion_model_final.pth"

    def load_data(self):
        """加载和处理数据

        该方法负责加载原始数据，预处理，提取特征，识别行为模式，
        并将行为标签扩展到每个样本点，最后选择连续的样本。
        """
        logger.info(f"正在从 {self.input_data_path} 加载数据")

        # 预处理原始数据
        logger.info("正在预处理原始数据...")
        self.df = self._preprocess_raw_data(self.input_data_path)

        # 提取特征
        logger.info("正在提取特征...")
        self.features_df = self.feature_extractor.extract(self.df)

        # 获取合法丢包值集合
        self.valid_loss_values = self.feature_extractor.valid_loss_values
        logger.info(f"合法丢包值: {self.valid_loss_values}")

        # 初始化约束注入器
        self.constraint_injector = ConstraintInjector(self.valid_loss_values)

        # 识别行为模式
        logger.info("正在识别行为模式...")
        patterns = self.pattern_identifier.identify(self.features_df, self.df)

        # 获取行为标签
        self.behavior_ids = np.array(patterns["labels"])
        logger.info(f"行为标签形状: {self.behavior_ids.shape}")
        logger.info(f"唯一行为: {np.unique(self.behavior_ids)}")

        # 将行为标签扩展到每个样本点
        self._expand_behavior_ids()

        # 选择连续6000个样本
        self._select_continuous_samples()

    def _preprocess_raw_data(self, file_path: Path) -> pd.DataFrame:
        """预处理原始数据文件，转换为标准格式

        该方法读取原始网络数据文件，解析延迟和丢包率数据，并将其转换为结构化的DataFrame。

        Args:
            file_path: 原始数据文件路径

        Returns:
            pd.DataFrame: 标准格式的DataFrame，包含timestamp、delay、loss_rate列
        """
        # 读取原始文件
        with open(file_path, "r") as f:
            lines = f.readlines()

        # 跳过前12行元数据和分隔线
        data_lines = lines[12:]

        # 提取开始时间
        start_time_line = [line for line in lines if "Start Time" in line][0]
        start_time_str = start_time_line.split(": ", 1)[1].strip()
        start_time = pd.Timestamp(start_time_str)

        # 提取时间间隔
        interval_line = [line for line in lines if "Interval(sec)" in line][0]
        interval_sec = float(interval_line.split(": ", 1)[1].strip())

        # 解析数据
        data = []
        for i, line in enumerate(data_lines):
            line = line.strip()
            if not line:
                continue

            # 分割数据行
            parts = line.split(",")
            if len(parts) < 6:
                continue

            # 提取Delay1和Loss1作为我们的delay和loss_rate
            delay = float(parts[0])

            # 当遇到时延大于2000的数据时，忽略它和它之后的所有数据
            if delay > 2000:
                logger.warning(f"在第{i}行发现时延大于2000ms，停止处理")
                break

            loss_rate = float(parts[1]) / 100  # 转换为0-1范围

            # 计算时间戳
            timestamp = start_time + pd.Timedelta(seconds=interval_sec * i)

            data.append(
                {"timestamp": timestamp, "delay": delay, "loss_rate": loss_rate}
            )

        # 创建DataFrame
        df = pd.DataFrame(data)
        logger.info(f"预处理后数据形状: {df.shape}")
        logger.info(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
        logger.info(f"时延范围: {df['delay'].min():.2f} 到 {df['delay'].max():.2f} ms")
        logger.info(f"丢包率范围: {df['loss_rate'].min():.4f} 到 {df['loss_rate'].max():.4f}")

        return df

    def _expand_behavior_ids(self):
        """将行为标签从窗口级别扩展到样本点级别

        该方法将窗口级别的行为标签扩展到每个样本点，使每个样本点都有对应的行为标签。
        """
        # 计算每个窗口的样本数
        window_samples = self.feature_extractor.window_samples  # 窗口样本数

        # 扩展行为标签
        expanded_ids = []
        for i, behavior_id in enumerate(self.behavior_ids):
            # 添加行为标签到扩展列表
            expanded_ids.extend([behavior_id] * window_samples)

        # 截断到原始数据长度
        expanded_ids = expanded_ids[: len(self.df)]
        self.behavior_ids = np.array(expanded_ids)
        logger.info(f"扩展后行为标签形状: {self.behavior_ids.shape}")

    def _select_continuous_samples(
        self, sample_length: int = 6000, start_idx: int = 1000
    ):
        """选择连续样本

        该方法从原始数据中选择连续的样本，并处理对应的行为标签。

        Args:
            sample_length: 选择样本的长度，默认6000个样本
            start_idx: 开始索引，默认从第1000个样本开始
        """
        logger.info(f"正在从索引 {start_idx} 开始选择 {sample_length} 个连续样本")

        # 确保有足够的样本
        if len(self.df) < start_idx + sample_length:
            start_idx = len(self.df) - sample_length
            logger.warning(f"调整起始索引为 {start_idx} 以确保有足够的样本")

        # 选择连续样本
        self.selected_df = self.df.iloc[start_idx : start_idx + sample_length].copy()
        self.selected_behavior_ids = self.behavior_ids[
            start_idx : start_idx + sample_length
        ]

        # 处理行为ID，重新映射为0-based索引
        unique_behaviors = np.unique(self.selected_behavior_ids)
        behavior_mapping = {}
        current_id = 0
        for behavior in unique_behaviors:
            behavior_mapping[behavior] = current_id
            current_id += 1

        # 重新映射行为ID
        self.selected_behavior_ids = np.array(
            [behavior_mapping[b] for b in self.selected_behavior_ids]
        )

        logger.info(f"选择的数据形状: {self.selected_df.shape}")
        logger.info(f"选择的行为标签形状: {self.selected_behavior_ids.shape}")
        logger.info(f"选择的唯一行为: {np.unique(self.selected_behavior_ids)}")

    def load_model(self):
        """加载预训练的扩散模型

        该方法加载训练好的扩散模型，用于生成网络状态样本。

        Raises:
            SystemExit: 如果模型文件不存在，退出程序
        """
        if not self.model_path.exists():
            logger.error(f"模型文件 {self.model_path} 不存在。请先训练模型。")
            sys.exit(1)

        logger.info(f"正在从 {self.model_path} 加载模型")

        # 加载模型权重和配置
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )

        # 从checkpoint获取行为映射，确定正确的行为数量
        behavior_mapping = checkpoint.get("behavior_mapping", {})
        actual_num_behaviors = (
            len(behavior_mapping) if len(behavior_mapping) > 0 else 2
        )  # 默认2种行为
        logger.info(f"从checkpoint获取的行为映射: {behavior_mapping}")
        logger.info(f"使用的行为数量: {actual_num_behaviors}")

        # 初始化模型
        self.model = ConditionDiffusionModel(
            input_dim=2,  # 输入维度：延迟和丢包率
            num_behaviors=actual_num_behaviors,  # 实际行为数量
            behavior_embed_dim=32,  # 行为嵌入维度
            T=1000,  # 扩散步数
        )

        # 加载模型状态，忽略不匹配的键
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model.to(self.device)
        self.model.eval()
        logger.info("模型加载成功")

        # 加载scaler参数
        self.delay_scaler_mean_ = checkpoint.get("delay_scaler_mean_", None)
        self.delay_scaler_scale_ = checkpoint.get("delay_scaler_scale_", None)
        logger.info(
            f"加载的scaler参数 - Mean: {self.delay_scaler_mean_}, Scale: {self.delay_scaler_scale_}"
        )

        # 保存行为映射
        self.behavior_mapping = behavior_mapping

    def generate_data(self):
        """生成网络状态样本数据

        该方法使用训练好的扩散模型，根据选择的连续样本对应的行为标签，
        生成对应的网络状态样本数据。

        Returns:
            pd.DataFrame: 生成的网络状态样本DataFrame，包含timestamp、delay、loss_rate列
        """
        logger.info("正在生成网络状态数据...")

        # 获取当前选择样本的原始行为ID分布
        original_selected_behaviors = np.unique(self.selected_behavior_ids)
        logger.info(f"当前选择样本的原始行为ID: {original_selected_behaviors}")
        logger.info(f"模型训练时的行为映射: {self.behavior_mapping}")

        # 转换行为标签为模型训练时使用的ID
        mapped_behavior_ids = []
        for behavior_id in self.selected_behavior_ids:
            # 找到最接近的映射行为ID
            # 由于训练时我们处理了行为ID，这里使用模型训练时的映射规则
            # 对于负标签(-1)，映射到0；其他映射到1
            if behavior_id == -1:
                mapped_behavior_ids.append(0)
            else:
                mapped_behavior_ids.append(1)
        mapped_behavior_ids = np.array(mapped_behavior_ids)
        logger.info(f"映射后的行为ID分布: {np.unique(mapped_behavior_ids)}")

        # 转换行为标签为张量
        behavior_ids_tensor = (
            torch.tensor(mapped_behavior_ids, dtype=torch.long)
            .unsqueeze(0)
            .to(self.device)
        )

        # 生成样本
        with torch.no_grad():
            generated = self.model.sample(behavior_ids_tensor, self.device)

        # 转换回numpy数组
        generated = generated.cpu().numpy()[0]

        # 从生成数据中提取归一化的延迟和丢包率
        delay_norm = generated[:, 0]  # 归一化的延迟
        loss_norm = generated[:, 1]  # 归一化的丢包率

        # 反归一化延迟
        if (
            hasattr(self, "delay_scaler_mean_")
            and self.delay_scaler_mean_ is not None
            and hasattr(self, "delay_scaler_scale_")
            and self.delay_scaler_scale_ is not None
        ):
            # 使用RobustScaler反变换
            delay = (delay_norm * self.delay_scaler_scale_) + self.delay_scaler_mean_
            logger.info(
                f"使用RobustScaler反变换 - Mean: {self.delay_scaler_mean_}, Scale: {self.delay_scaler_scale_}"
            )
        else:
            # 回退到Min-Max反变换（仅当scaler参数不可用时）
            delay_min = self.selected_df["delay"].min()
            delay_max = self.selected_df["delay"].max()
            delay = (delay_norm + 1) * (delay_max - delay_min) / 2 + delay_min
            logger.info("使用Min-Max反变换（回退方案）")

        # 反归一化丢包率
        loss_rate = self.constraint_injector.process_loss_rate(loss_norm)

        # 创建生成的数据Frame
        generated_df = self.selected_df.copy()
        generated_df["delay"] = delay
        generated_df["loss_rate"] = loss_rate

        # 优化后处理：确保生成的时延覆盖原始数据的分布
        # 1. 确保时延非负
        generated_df["delay"] = np.clip(generated_df["delay"], a_min=0.0, a_max=None)

        # 2. 统计信息
        logger.info(
            f"生成数据时延 - 最小值: {generated_df['delay'].min():.4f}, 最大值: {generated_df['delay'].max():.4f}, 平均值: {generated_df['delay'].mean():.4f}"
        )
        logger.info(
            f"原始数据时延 - 最小值: {self.selected_df['delay'].min():.4f}, 最大值: {self.selected_df['delay'].max():.4f}, 平均值: {self.selected_df['delay'].mean():.4f}"
        )

        return generated_df

    def visualize_comparison(self, generated_df: pd.DataFrame):
        """可视化对比原始数据和生成数据

        该方法生成多种可视化图表，对比原始网络数据和生成的网络数据，
        包括时间序列对比、统计分布对比和相关性对比。

        Args:
            generated_df: 生成的网络状态样本DataFrame
        """
        logger.info("正在可视化对比原始数据和生成数据...")

        # 创建可视化输出目录
        viz_dir = self.output_path / "visualization"
        viz_dir.mkdir(parents=True, exist_ok=True)

        # 1. 绘制时间序列对比图
        plt.figure(figsize=(15, 6))
        plt.subplot(2, 1, 1)
        plt.plot(
            self.selected_df["timestamp"],
            self.selected_df["delay"],
            label="原始数据",
            alpha=0.7,
        )
        plt.plot(
            generated_df["timestamp"],
            generated_df["delay"],
            label="生成数据",
            alpha=0.7,
        )
        plt.title("时延对比")
        plt.xlabel("时间")
        plt.ylabel("时延 (ms)")
        plt.legend()
        plt.grid(True)

        # 绘制丢包率对比
        plt.subplot(2, 1, 2)
        plt.plot(
            self.selected_df["timestamp"],
            self.selected_df["loss_rate"],
            label="原始数据",
            alpha=0.7,
        )
        plt.plot(
            generated_df["timestamp"],
            generated_df["loss_rate"],
            label="生成数据",
            alpha=0.7,
        )
        plt.title("丢包率对比")
        plt.xlabel("时间")
        plt.ylabel("丢包率")
        plt.ylim(0, 1)  # 设置丢包率y轴范围为0-1
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        comparison_plot = viz_dir / "comparison.png"
        plt.savefig(comparison_plot, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"时间序列对比图已保存到: {comparison_plot}")

        # 2. 绘制统计分布对比图
        plt.figure(figsize=(15, 6))

        # 时延直方图
        plt.subplot(2, 2, 1)
        plt.hist(self.selected_df["delay"], bins=50, alpha=0.7, label="原始数据")
        plt.hist(generated_df["delay"], bins=50, alpha=0.7, label="生成数据")
        plt.title("时延分布")
        plt.xlabel("时延 (ms)")
        plt.ylabel("频率")
        plt.legend()
        plt.grid(True)

        # 丢包率直方图
        plt.subplot(2, 2, 2)
        plt.hist(self.selected_df["loss_rate"], bins=20, alpha=0.7, label="原始数据")
        plt.hist(generated_df["loss_rate"], bins=20, alpha=0.7, label="生成数据")
        plt.title("丢包率分布")
        plt.xlabel("丢包率")
        plt.ylabel("频率")
        plt.xlim(0, 1)  # 设置丢包率x轴范围为0-1
        plt.legend()
        plt.grid(True)

        # 时延箱线图
        plt.subplot(2, 2, 3)
        plt.boxplot(
            [self.selected_df["delay"], generated_df["delay"]],
            labels=["原始数据", "生成数据"],
        )
        plt.title("时延箱线图")
        plt.ylabel("时延 (ms)")
        plt.grid(True)

        # 丢包率箱线图
        plt.subplot(2, 2, 4)
        plt.boxplot(
            [self.selected_df["loss_rate"], generated_df["loss_rate"]],
            labels=["原始数据", "生成数据"],
        )
        plt.title("丢包率箱线图")
        plt.ylabel("丢包率")
        plt.ylim(0, 1)  # 设置丢包率y轴范围为0-1
        plt.grid(True)

        plt.tight_layout()
        statistics_plot = viz_dir / "statistics.png"
        plt.savefig(statistics_plot, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"统计分布对比图已保存到: {statistics_plot}")

        # 3. 绘制相关性对比图
        plt.figure(figsize=(15, 6))

        # 原始数据相关性
        plt.subplot(1, 2, 1)
        plt.scatter(
            self.selected_df["delay"], self.selected_df["loss_rate"], alpha=0.5, s=10
        )
        plt.title("原始数据: 时延 vs 丢包率")
        plt.xlabel("时延 (ms)")
        plt.ylabel("丢包率")
        plt.ylim(0, 1)  # 设置丢包率y轴范围为0-1
        plt.grid(True)

        # 生成数据相关性
        plt.subplot(1, 2, 2)
        plt.scatter(generated_df["delay"], generated_df["loss_rate"], alpha=0.5, s=10)
        plt.title("生成数据: 时延 vs 丢包率")
        plt.xlabel("时延 (ms)")
        plt.ylabel("丢包率")
        plt.ylim(0, 1)  # 设置丢包率y轴范围为0-1
        plt.grid(True)

        plt.tight_layout()
        correlation_plot = viz_dir / "correlation.png"
        plt.savefig(correlation_plot, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"相关性对比图已保存到: {correlation_plot}")

    def run(self):
        """运行完整的生成和可视化流程

        该方法执行完整的生成和可视化流程，包括加载数据、加载模型、生成数据、
        保存数据和可视化对比。
        """
        # 加载和处理数据
        self.load_data()

        # 加载预训练模型
        self.load_model()

        # 生成网络状态样本数据
        generated_df = self.generate_data()

        # 保存生成的数据和原始数据
        generated_file = self.output_path / "generated_sample_6000.csv"
        original_file = self.output_path / "original_sample_6000.csv"
        generated_df.to_csv(generated_file, index=False)
        self.selected_df.to_csv(original_file, index=False)
        logger.info(f"生成数据已保存到: {generated_file}")
        logger.info(f"原始数据已保存到: {original_file}")

        # 可视化对比原始数据和生成数据
        self.visualize_comparison(generated_df)

        logger.info("生成和可视化流程完成！")


def main():
    """主函数入口

    解析命令行参数，创建生成和可视化实例，并运行完整流程。

    命令行参数：
        python generate_and_visualize.py <input_data_path> <output_path> [model_path]

    参数说明：
        input_data_path: 输入数据文件路径
        output_path: 输出目录路径
        model_path: 预训练模型文件路径，可选
    """
    if len(sys.argv) < 3:
        logger.error(
            "用法: python generate_and_visualize.py <input_data_path> <output_path> [model_path]"
        )
        sys.exit(1)

    input_data_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    model_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    # 创建生成和可视化实例
    generator = GenerateAndVisualize(input_data_path, output_path, model_path)

    # 运行完整流程
    generator.run()


if __name__ == "__main__":
    main()
