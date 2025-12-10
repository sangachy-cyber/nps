#!/usr/bin/env python3
"""
网络结果可视化模块
负责可视化生成结果与原始数据的对比
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from network_simulation.utils.logger import get_logger
from config import PLOTS_DIR, DEFAULT_SAMPLE_COMPARISON, DEFAULT_GENERATED_DISTRIBUTION, DEFAULT_ORIGINAL_DISTRIBUTION, DEFAULT_GENERATED_TIMELINE, DEFAULT_ORIGINAL_TIMELINE

# 初始化日志记录器
logger = get_logger(__name__)

# 设置中文支持，兼容Windows、MacOS和Linux，Linux系统优先使用文泉驿正黑
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class ResultsVisualizer:
    """结果可视化器类

    该类生成多种可视化图表，对比原始网络数据和生成的网络数据，
    包括时间序列对比、统计分布对比和相关性对比。
    """

    def __init__(self, output_dir: Path):
        """初始化结果可视化器

        Args:
            output_dir: 可视化结果的输出目录路径
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 确保PLOTS_DIR存在
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"初始化结果可视化器，输出目录: {output_dir}")

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

        Raises:
            FileNotFoundError: 如果输入文件不存在
            ValueError: 如果数据格式不符合要求
        """
        logger.info(f"正在可视化 {input_original_file} 和 {input_generated_file} 的对比结果")

        # 加载原始数据和生成数据
        original_df = pd.read_csv(input_original_file, parse_dates=["timestamp"])
        generated_df = pd.read_csv(input_generated_file, parse_dates=["timestamp"])

        # 1. 绘制参考样本与生成样本对比图（时序图）
        self._plot_sample_comparison(original_df, generated_df, is_main_visualization)

        # 2. 绘制分布直方图
        self._plot_distribution_comparison(original_df, generated_df, is_main_visualization)

        # 3. 绘制生成样本时序图
        self._plot_generated_timeline(generated_df, is_main_visualization)

        # 4. 绘制参考样本时序图
        self._plot_original_timeline(original_df, is_main_visualization)

        # 5. 计算并打印统计指标对比
        self._print_statistical_comparison(original_df, generated_df)

        logger.info("可视化完成！")
        return self.output_dir

    def _plot_sample_comparison(
        self,
        original_df: pd.DataFrame,
        generated_df: pd.DataFrame,
        is_main_visualization: bool
    ) -> None:
        """绘制参考样本与生成样本对比图（时序图）

        Args:
            original_df: 原始数据DataFrame
            generated_df: 生成数据DataFrame
            is_main_visualization: 是否为主要可视化
        """
        # 确保timestamp是datetime类型
        original_df['timestamp'] = pd.to_datetime(original_df['timestamp'])
        generated_df['timestamp'] = pd.to_datetime(generated_df['timestamp'])

        # 计算数据的总时长（分钟）
        total_minutes = (original_df['timestamp'].max() - original_df['timestamp'].min()).total_seconds() / 60

        # 定义每个片段的时长（5分钟）
        segment_duration = 5  # 分钟

        # 计算需要生成的片段数量
        num_segments = int(total_minutes / segment_duration) + 1

        if is_main_visualization:
            # 只生成前2个片段（10分钟数据，两张图）
            num_segments = min(num_segments, 2)

        for segment in range(num_segments):
            # 计算当前片段的时间范围
            start_time = original_df['timestamp'].min() + pd.Timedelta(minutes=segment*segment_duration)
            end_time = start_time + pd.Timedelta(minutes=segment_duration)

            # 过滤当前片段的数据
            original_segment = original_df[(original_df['timestamp'] >= start_time) & (original_df['timestamp'] < end_time)]
            generated_segment = generated_df[(generated_df['timestamp'] >= start_time) & (generated_df['timestamp'] < end_time)]

            # 如果当前片段没有数据，跳过
            if len(original_segment) == 0 or len(generated_segment) == 0:
                continue

            # 创建上下两个子图，分别显示参考样本和生成样本
            fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(15, 12))

            # 检查数据是否包含4列（上下行）
            has_4columns = 'delay1' in original_df.columns and 'delay2' in original_df.columns

            if has_4columns:
                # 参考样本子图 - 上下行
                ax1.plot(
                    original_segment["timestamp"], original_segment["delay1"], label="参考样本上行时延", alpha=0.7, color='blue'
                )
                ax1.plot(
                    original_segment["timestamp"], original_segment["delay2"], label="参考样本下行时延", alpha=0.7, color='cyan'
                )
                ax1.set_title(f"参考样本 (时间段 {segment+1}: {start_time.strftime('%H:%M:%S')} 至 {end_time.strftime('%H:%M:%S')})")
                ax1.set_xlabel("时间")
                ax1.set_ylabel("时延 (ms)")
                ax1.tick_params(axis='y', labelcolor='blue')
                ax1.grid(True, alpha=0.3)

                # 参考样本丢包率子图（与参考样本时延共享x轴）
                ax2 = ax1.twinx()
                ax2.plot(
                    original_segment["timestamp"], original_segment["loss_rate1"], label="参考样本上行丢包率", alpha=0.7, color='green', linestyle='--'
                )
                ax2.plot(
                    original_segment["timestamp"], original_segment["loss_rate2"], label="参考样本下行丢包率", alpha=0.7, color='lime', linestyle='--'
                )
                ax2.set_ylabel("丢包率")
                ax2.tick_params(axis='y', labelcolor='green')
                ax2.set_ylim(-0.01, 1.01)  # 设置丢包率y轴范围为-0.01到1.01

                # 合并参考样本图例
                lines1 = ax1.get_lines() + ax2.get_lines()
                labels1 = [line.get_label() for line in lines1]
                ax1.legend(lines1, labels1, loc='upper right', fontsize=10)

                # 生成样本子图 - 上下行
                ax3.plot(
                    generated_segment["timestamp"], generated_segment["delay1"], label="生成样本上行时延", alpha=0.7, color='orange'
                )
                ax3.plot(
                    generated_segment["timestamp"], generated_segment["delay2"], label="生成样本下行时延", alpha=0.7, color='darkorange'
                )
                ax3.set_title(f"生成样本 (时间段 {segment+1}: {start_time.strftime('%H:%M:%S')} 至 {end_time.strftime('%H:%M:%S')})")
                ax3.set_xlabel("时间")
                ax3.set_ylabel("时延 (ms)")
                ax3.tick_params(axis='y', labelcolor='orange')
                ax3.grid(True, alpha=0.3)

                # 生成样本丢包率子图（与生成样本时延共享x轴）
                ax4 = ax3.twinx()
                ax4.plot(
                    generated_segment["timestamp"], generated_segment["loss_rate1"], label="生成样本上行丢包率", alpha=0.7, color='red', linestyle='--'
                )
                ax4.plot(
                    generated_segment["timestamp"], generated_segment["loss_rate2"], label="生成样本下行丢包率", alpha=0.7, color='darkred', linestyle='--'
                )
                ax4.set_ylabel("丢包率")
                ax4.tick_params(axis='y', labelcolor='red')
                ax4.set_ylim(-0.01, 1.01)  # 设置丢包率y轴范围为-0.01到1.01
            else:
                # 参考样本子图 - 单流（兼容旧格式）
                ax1.plot(
                    original_segment["timestamp"], original_segment["delay"], label="参考样本时延", alpha=0.7, color='blue'
                )
                ax1.set_title(f"参考样本 (时间段 {segment+1}: {start_time.strftime('%H:%M:%S')} 至 {end_time.strftime('%H:%M:%S')})")
                ax1.set_xlabel("时间")
                ax1.set_ylabel("时延 (ms)")
                ax1.tick_params(axis='y', labelcolor='blue')
                ax1.grid(True, alpha=0.3)

                # 参考样本丢包率子图（与参考样本时延共享x轴）
                ax2 = ax1.twinx()
                ax2.plot(
                    original_segment["timestamp"], original_segment["loss_rate"], label="参考样本丢包率", alpha=0.7, color='green', linestyle='--'
                )
                ax2.set_ylabel("丢包率")
                ax2.tick_params(axis='y', labelcolor='green')
                ax2.set_ylim(-0.01, 1.01)  # 设置丢包率y轴范围为-0.01到1.01

                # 合并参考样本图例
                lines1 = ax1.get_lines() + ax2.get_lines()
                labels1 = [line.get_label() for line in lines1]
                ax1.legend(lines1, labels1, loc='upper right', fontsize=10)

                # 生成样本子图 - 单流（兼容旧格式）
                ax3.plot(
                    generated_segment["timestamp"], generated_segment["delay"], label="生成样本时延", alpha=0.7, color='orange'
                )
                ax3.set_title(f"生成样本 (时间段 {segment+1}: {start_time.strftime('%H:%M:%S')} 至 {end_time.strftime('%H:%M:%S')})")
                ax3.set_xlabel("时间")
                ax3.set_ylabel("时延 (ms)")
                ax3.tick_params(axis='y', labelcolor='orange')
                ax3.grid(True, alpha=0.3)

                # 生成样本丢包率子图（与生成样本时延共享x轴）
                ax4 = ax3.twinx()
                ax4.plot(
                    generated_segment["timestamp"], generated_segment["loss_rate"], label="生成样本丢包率", alpha=0.7, color='red', linestyle='--'
                )
                ax4.set_ylabel("丢包率")
                ax4.tick_params(axis='y', labelcolor='red')
                ax4.set_ylim(-0.01, 1.01)  # 设置丢包率y轴范围为-0.01到1.01

            # 合并生成样本图例
            lines2 = ax3.get_lines() + ax4.get_lines()
            labels2 = [line.get_label() for line in lines2]
            ax3.legend(lines2, labels2, loc='upper right', fontsize=10)

            plt.tight_layout()

            if is_main_visualization:
                # 保存到报告目录，用于综合报告
                if segment == 0:
                    sample_comparison_plot = PLOTS_DIR / DEFAULT_SAMPLE_COMPARISON
                else:
                    sample_comparison_plot = PLOTS_DIR / f"sample_comparison_segment_{segment+1}.png"
                plt.savefig(sample_comparison_plot, dpi=300, bbox_inches="tight")
                logger.info(f"参考样本与生成样本对比图已保存到: {sample_comparison_plot}")
            else:
                # 保存到组目录
                comparison_plot = self.output_dir / f"comparison_segment_{segment+1}.png"
                plt.savefig(comparison_plot, dpi=300, bbox_inches="tight")
                logger.info(f"时间序列对比图已保存到: {comparison_plot}")
            plt.close()

        # 对于主可视化，还需要生成一张包含所有数据的缩略图
        if is_main_visualization:
            # 创建上下两个子图，分别显示参考样本和生成样本
            fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(15, 10))

            # 检查数据是否包含4列（上下行）
            has_4columns = 'delay1' in original_df.columns and 'delay2' in original_df.columns

            if has_4columns:
                # 参考样本子图 - 上下行
                ax1.plot(
                    original_df["timestamp"], original_df["delay1"], label="参考样本上行时延", alpha=0.5, color='blue', linewidth=1
                )
                ax1.plot(
                    original_df["timestamp"], original_df["delay2"], label="参考样本下行时延", alpha=0.5, color='cyan', linewidth=1
                )
                ax1.set_title("参考样本（全部数据）")
                ax1.set_xlabel("时间")
                ax1.set_ylabel("时延 (ms)")
                ax1.grid(True, alpha=0.3)

                # 参考样本丢包率子图（与参考样本时延共享x轴）
                ax2 = ax1.twinx()
                ax2.plot(
                    original_df["timestamp"], original_df["loss_rate1"], label="参考样本上行丢包率", alpha=0.5, color='green', linestyle='--', linewidth=1
                )
                ax2.plot(
                    original_df["timestamp"], original_df["loss_rate2"], label="参考样本下行丢包率", alpha=0.5, color='lime', linestyle='--', linewidth=1
                )
                ax2.set_ylabel("丢包率")
                ax2.set_ylim(-0.01, 1.01)  # 设置丢包率y轴范围为-0.01到1.01

                # 合并参考样本图例
                lines1 = ax1.get_lines() + ax2.get_lines()
                labels1 = [line.get_label() for line in lines1]
                ax1.legend(lines1, labels1, loc='upper right', fontsize=10)

                # 生成样本子图 - 上下行
                ax3.plot(
                    generated_df["timestamp"], generated_df["delay1"], label="生成样本上行时延", alpha=0.5, color='orange', linewidth=1
                )
                ax3.plot(
                    generated_df["timestamp"], generated_df["delay2"], label="生成样本下行时延", alpha=0.5, color='darkorange', linewidth=1
                )
                ax3.set_title("生成样本（全部数据）")
                ax3.set_xlabel("时间")
                ax3.set_ylabel("时延 (ms)")
                ax3.grid(True, alpha=0.3)

                # 生成样本丢包率子图（与生成样本时延共享x轴）
                ax4 = ax3.twinx()
                ax4.plot(
                    generated_df["timestamp"], generated_df["loss_rate1"], label="生成样本上行丢包率", alpha=0.5, color='red', linestyle='--', linewidth=1
                )
                ax4.plot(
                    generated_df["timestamp"], generated_df["loss_rate2"], label="生成样本下行丢包率", alpha=0.5, color='darkred', linestyle='--', linewidth=1
                )
                ax4.set_ylabel("丢包率")
                ax4.set_ylim(-0.01, 1.01)  # 设置丢包率y轴范围为-0.01到1.01
            else:
                # 参考样本子图 - 单流（兼容旧格式）
                ax1.plot(
                    original_df["timestamp"], original_df["delay"], label="参考样本时延", alpha=0.5, color='blue', linewidth=1
                )
                ax1.set_title("参考样本（全部数据）")
                ax1.set_xlabel("时间")
                ax1.set_ylabel("时延 (ms)")
                ax1.grid(True, alpha=0.3)

                # 参考样本丢包率子图（与参考样本时延共享x轴）
                ax2 = ax1.twinx()
                ax2.plot(
                    original_df["timestamp"], original_df["loss_rate"], label="参考样本丢包率", alpha=0.5, color='green', linestyle='--', linewidth=1
                )
                ax2.set_ylabel("丢包率")
                ax2.set_ylim(-0.01, 1.01)  # 设置丢包率y轴范围为-0.01到1.01

                # 合并参考样本图例
                lines1 = ax1.get_lines() + ax2.get_lines()
                labels1 = [line.get_label() for line in lines1]
                ax1.legend(lines1, labels1, loc='upper right', fontsize=10)

                # 生成样本子图 - 单流（兼容旧格式）
                ax3.plot(
                    generated_df["timestamp"], generated_df["delay"], label="生成样本时延", alpha=0.5, color='orange', linewidth=1
                )
                ax3.set_title("生成样本（全部数据）")
                ax3.set_xlabel("时间")
                ax3.set_ylabel("时延 (ms)")
                ax3.grid(True, alpha=0.3)

                # 生成样本丢包率子图（与生成样本时延共享x轴）
                ax4 = ax3.twinx()
                ax4.plot(
                    generated_df["timestamp"], generated_df["loss_rate"], label="生成样本丢包率", alpha=0.5, color='red', linestyle='--', linewidth=1
                )
                ax4.set_ylabel("丢包率")
                ax4.set_ylim(-0.01, 1.01)  # 设置丢包率y轴范围为-0.01到1.01

            # 合并生成样本图例
            lines2 = ax3.get_lines() + ax4.get_lines()
            labels2 = [line.get_label() for line in lines2]
            ax3.legend(lines2, labels2, loc='upper right', fontsize=10)

            plt.tight_layout()

            # 保存缩略图
            sample_comparison_thumbnail = PLOTS_DIR / "sample_comparison_thumbnail.png"
            plt.savefig(sample_comparison_thumbnail, dpi=300, bbox_inches="tight")
            logger.info(f"参考样本与生成样本对比缩略图已保存到: {sample_comparison_thumbnail}")
            plt.close()

    def _plot_distribution_comparison(
        self,
        original_df: pd.DataFrame,
        generated_df: pd.DataFrame,
        is_main_visualization: bool
    ) -> None:
        """绘制分布直方图

        Args:
            original_df: 原始数据DataFrame
            generated_df: 生成数据DataFrame
            is_main_visualization: 是否为主要可视化
        """
        # 检查数据是否包含4列（上下行）
        has_4columns = 'delay1' in original_df.columns and 'delay2' in original_df.columns

        if has_4columns:
            # 上下行数据分布 - 2行4列布局
            plt.figure(figsize=(20, 12))

            # 原始数据分布
            plt.subplot(2, 4, 1)
            plt.hist(original_df["delay1"], bins=50, alpha=0.7, color='blue', label="参考样本上行")
            plt.title("参考样本上行时延分布")
            plt.xlabel("时延 (ms)")
            plt.ylabel("频率")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 4, 2)
            plt.hist(original_df["delay2"], bins=50, alpha=0.7, color='cyan', label="参考样本下行")
            plt.title("参考样本下行时延分布")
            plt.xlabel("时延 (ms)")
            plt.ylabel("频率")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 4, 3)
            plt.hist(original_df["loss_rate1"], bins=20, alpha=0.7, color='green', label="参考样本上行")
            plt.title("参考样本上行丢包率分布")
            plt.xlabel("丢包率")
            plt.ylabel("频率")
            plt.xlim(-0.01, 1.01)  # 设置丢包率范围为-0.01到1.01
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 4, 4)
            plt.hist(original_df["loss_rate2"], bins=20, alpha=0.7, color='lime', label="参考样本下行")
            plt.title("参考样本下行丢包率分布")
            plt.xlabel("丢包率")
            plt.ylabel("频率")
            plt.xlim(-0.01, 1.01)  # 设置丢包率范围为-0.01到1.01
            plt.legend()
            plt.grid(True)

            # 生成数据分布
            plt.subplot(2, 4, 5)
            plt.hist(generated_df["delay1"], bins=50, alpha=0.7, color='orange', label="生成样本上行")
            plt.title("生成样本上行时延分布")
            plt.xlabel("时延 (ms)")
            plt.ylabel("频率")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 4, 6)
            plt.hist(generated_df["delay2"], bins=50, alpha=0.7, color='darkorange', label="生成样本下行")
            plt.title("生成样本下行时延分布")
            plt.xlabel("时延 (ms)")
            plt.ylabel("频率")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 4, 7)
            plt.hist(generated_df["loss_rate1"], bins=20, alpha=0.7, color='red', label="生成样本上行")
            plt.title("生成样本上行丢包率分布")
            plt.xlabel("丢包率")
            plt.ylabel("频率")
            plt.xlim(-0.01, 1.01)  # 设置丢包率范围为-0.01到1.01
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 4, 8)
            plt.hist(generated_df["loss_rate2"], bins=20, alpha=0.7, color='darkred', label="生成样本下行")
            plt.title("生成样本下行丢包率分布")
            plt.xlabel("丢包率")
            plt.ylabel("频率")
            plt.xlim(-0.01, 1.01)  # 设置丢包率范围为-0.01到1.01
            plt.legend()
            plt.grid(True)
        else:
            # 单流数据分布 - 2行2列布局
            plt.figure(figsize=(15, 12))

            # 原始数据分布
            plt.subplot(2, 2, 1)
            plt.hist(original_df["delay"], bins=50, alpha=0.7, color='blue', label="参考样本")
            plt.title("参考样本时延分布")
            plt.xlabel("时延 (ms)")
            plt.ylabel("频率")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 2)
            plt.hist(original_df["loss_rate"], bins=20, alpha=0.7, color='blue', label="参考样本")
            plt.title("参考样本丢包率分布")
            plt.xlabel("丢包率")
            plt.ylabel("频率")
            plt.xlim(-0.01, 1.01)  # 设置丢包率范围为-0.01到1.01
            plt.legend()
            plt.grid(True)

            # 生成数据分布
            plt.subplot(2, 2, 3)
            plt.hist(generated_df["delay"], bins=50, alpha=0.7, color='orange', label="生成样本")
            plt.title("生成样本时延分布")
            plt.xlabel("时延 (ms)")
            plt.ylabel("频率")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 4)
            plt.hist(generated_df["loss_rate"], bins=20, alpha=0.7, color='orange', label="生成样本")
            plt.title("生成样本丢包率分布")
            plt.xlabel("丢包率")
            plt.ylabel("频率")
            plt.xlim(-0.01, 1.01)  # 设置丢包率范围为-0.01到1.01
            plt.legend()
            plt.grid(True)

        plt.tight_layout()

        if is_main_visualization:
            # 保存到报告目录，用于综合报告
            original_distribution_plot = PLOTS_DIR / DEFAULT_ORIGINAL_DISTRIBUTION
            generated_distribution_plot = PLOTS_DIR / DEFAULT_GENERATED_DISTRIBUTION

            # 保存参考样本分布
            plt.savefig(original_distribution_plot, dpi=300, bbox_inches="tight")
            logger.info(f"参考样本分布直方图已保存到: {original_distribution_plot}")

            # 保存生成样本分布
            plt.savefig(generated_distribution_plot, dpi=300, bbox_inches="tight")
            logger.info(f"生成样本分布直方图已保存到: {generated_distribution_plot}")
        else:
            # 保存到组目录
            statistics_plot = self.output_dir / "statistics.png"
            plt.savefig(statistics_plot, dpi=300, bbox_inches="tight")
            logger.info(f"统计分布对比图已保存到: {statistics_plot}")
        plt.close()

    def _plot_generated_timeline(
        self,
        generated_df: pd.DataFrame,
        is_main_visualization: bool
    ) -> None:
        """绘制生成样本时序图

        Args:
            generated_df: 生成数据DataFrame
            is_main_visualization: 是否为主要可视化
        """
        # 检查数据是否包含4列（上下行）
        has_4columns = 'delay1' in generated_df.columns and 'delay2' in generated_df.columns

        if has_4columns:
            # 上下行数据时序图 - 2行2列布局
            plt.figure(figsize=(20, 12))

            # 时延图 - 上行和下行
            plt.subplot(2, 2, 1)
            plt.plot(generated_df["timestamp"], generated_df["delay1"], alpha=0.7, color='orange', label="上行")
            plt.title("生成样本上行时延时序图")
            plt.xlabel("时间")
            plt.ylabel("时延 (ms)")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 2)
            plt.plot(generated_df["timestamp"], generated_df["delay2"], alpha=0.7, color='darkorange', label="下行")
            plt.title("生成样本下行时延时序图")
            plt.xlabel("时间")
            plt.ylabel("时延 (ms)")
            plt.legend()
            plt.grid(True)

            # 丢包率图 - 上行和下行
            plt.subplot(2, 2, 3)
            plt.plot(generated_df["timestamp"], generated_df["loss_rate1"], alpha=0.7, color='red', label="上行")
            plt.title("生成样本上行丢包率时序图")
            plt.xlabel("时间")
            plt.ylabel("丢包率")
            plt.ylim(-0.01, 1.01)  # 设置丢包率范围为-0.01到1.01
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 4)
            plt.plot(generated_df["timestamp"], generated_df["loss_rate2"], alpha=0.7, color='darkred', label="下行")
            plt.title("生成样本下行丢包率时序图")
            plt.xlabel("时间")
            plt.ylabel("丢包率")
            plt.ylim(-0.01, 1.01)  # 设置丢包率范围为-0.01到1.01
            plt.legend()
            plt.grid(True)
        else:
            # 单流数据时序图
            plt.figure(figsize=(15, 8))
            plt.subplot(2, 1, 1)
            plt.plot(generated_df["timestamp"], generated_df["delay"], alpha=0.7, color='orange')
            plt.title("生成样本时延时序图")
            plt.xlabel("时间")
            plt.ylabel("时延 (ms)")
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(generated_df["timestamp"], generated_df["loss_rate"], alpha=0.7, color='orange')
            plt.title("生成样本丢包率时序图")
            plt.xlabel("时间")
            plt.ylabel("丢包率")
            plt.ylim(-0.01, 1.01)  # 设置丢包率范围为-0.01到1.01
            plt.grid(True)

        plt.tight_layout()

        if is_main_visualization:
            # 保存到报告目录，用于综合报告
            generated_timeline_plot = PLOTS_DIR / DEFAULT_GENERATED_TIMELINE
            plt.savefig(generated_timeline_plot, dpi=300, bbox_inches="tight")
            logger.info(f"生成样本时序图已保存到: {generated_timeline_plot}")
        plt.close()

    def _plot_original_timeline(
        self,
        original_df: pd.DataFrame,
        is_main_visualization: bool
    ) -> None:
        """绘制参考样本时序图

        Args:
            original_df: 原始数据DataFrame
            is_main_visualization: 是否为主要可视化
        """
        # 检查数据是否包含4列（上下行）
        has_4columns = 'delay1' in original_df.columns and 'delay2' in original_df.columns

        if has_4columns:
            # 上下行数据时序图 - 2行2列布局
            plt.figure(figsize=(20, 12))

            # 时延图 - 上行和下行
            plt.subplot(2, 2, 1)
            plt.plot(original_df["timestamp"], original_df["delay1"], alpha=0.7, color='blue', label="上行")
            plt.title("参考样本上行时延时序图")
            plt.xlabel("时间")
            plt.ylabel("时延 (ms)")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 2)
            plt.plot(original_df["timestamp"], original_df["delay2"], alpha=0.7, color='cyan', label="下行")
            plt.title("参考样本下行时延时序图")
            plt.xlabel("时间")
            plt.ylabel("时延 (ms)")
            plt.legend()
            plt.grid(True)

            # 丢包率图 - 上行和下行
            plt.subplot(2, 2, 3)
            plt.plot(original_df["timestamp"], original_df["loss_rate1"], alpha=0.7, color='green', label="上行")
            plt.title("参考样本上行丢包率时序图")
            plt.xlabel("时间")
            plt.ylabel("丢包率")
            plt.ylim(-0.01, 1.01)  # 设置丢包率范围为-0.01到1.01
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 4)
            plt.plot(original_df["timestamp"], original_df["loss_rate2"], alpha=0.7, color='lime', label="下行")
            plt.title("参考样本下行丢包率时序图")
            plt.xlabel("时间")
            plt.ylabel("丢包率")
            plt.ylim(-0.01, 1.01)  # 设置丢包率范围为-0.01到1.01
            plt.legend()
            plt.grid(True)
        else:
            # 单流数据时序图
            plt.figure(figsize=(15, 8))
            plt.subplot(2, 1, 1)
            plt.plot(original_df["timestamp"], original_df["delay"], alpha=0.7, color='blue')
            plt.title("参考样本时延时序图")
            plt.xlabel("时间")
            plt.ylabel("时延 (ms)")
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(original_df["timestamp"], original_df["loss_rate"], alpha=0.7, color='blue')
            plt.title("参考样本丢包率时序图")
            plt.xlabel("时间")
            plt.ylabel("丢包率")
            plt.ylim(-0.01, 1.01)  # 设置丢包率范围为-0.01到1.01
            plt.grid(True)

        plt.tight_layout()

        if is_main_visualization:
            # 保存到报告目录，用于综合报告
            original_timeline_plot = PLOTS_DIR / DEFAULT_ORIGINAL_TIMELINE
            plt.savefig(original_timeline_plot, dpi=300, bbox_inches="tight")
            logger.info(f"参考样本时序图已保存到: {original_timeline_plot}")
        plt.close()

    def _print_statistical_comparison(
        self,
        original_df: pd.DataFrame,
        generated_df: pd.DataFrame
    ) -> None:
        """计算并打印统计指标对比

        Args:
            original_df: 原始数据DataFrame
            generated_df: 生成数据DataFrame
        """
        logger.info("\n统计对比:")
        logger.info("=" * 50)

        # 检查数据是否包含4列（上下行）
        has_4columns = 'delay1' in original_df.columns and 'delay2' in original_df.columns

        if has_4columns:
            # 上下行数据统计
            logger.info("\n时延统计:")

            # 上行时延统计
            delay1_original_stats = original_df["delay1"].describe()
            delay1_generated_stats = generated_df["delay1"].describe()
            logger.info("上行:")
            logger.info(
                f"参考样本 - 最小值: {delay1_original_stats['min']:.2f}, 平均值: {delay1_original_stats['mean']:.2f}, 最大值: {delay1_original_stats['max']:.2f}, 标准差: {delay1_original_stats['std']:.2f}"
            )
            logger.info(
                f"生成样本 - 最小值: {delay1_generated_stats['min']:.2f}, 平均值: {delay1_generated_stats['mean']:.2f}, 最大值: {delay1_generated_stats['max']:.2f}, 标准差: {delay1_generated_stats['std']:.2f}"
            )

            # 下行时延统计
            delay2_original_stats = original_df["delay2"].describe()
            delay2_generated_stats = generated_df["delay2"].describe()
            logger.info("下行:")
            logger.info(
                f"参考样本 - 最小值: {delay2_original_stats['min']:.2f}, 平均值: {delay2_original_stats['mean']:.2f}, 最大值: {delay2_original_stats['max']:.2f}, 标准差: {delay2_original_stats['std']:.2f}"
            )
            logger.info(
                f"生成样本 - 最小值: {delay2_generated_stats['min']:.2f}, 平均值: {delay2_generated_stats['mean']:.2f}, 最大值: {delay2_generated_stats['max']:.2f}, 标准差: {delay2_generated_stats['std']:.2f}"
            )

            # 丢包率统计
            logger.info("\n丢包率统计:")

            # 上行丢包率统计
            loss1_original_stats = original_df["loss_rate1"].describe()
            loss1_generated_stats = generated_df["loss_rate1"].describe()
            logger.info("上行:")
            logger.info(
                f"参考样本 - 最小值: {loss1_original_stats['min']:.4f}, 平均值: {loss1_original_stats['mean']:.4f}, 最大值: {loss1_original_stats['max']:.4f}, 标准差: {loss1_original_stats['std']:.4f}"
            )
            logger.info(
                f"生成样本 - 最小值: {loss1_generated_stats['min']:.4f}, 平均值: {loss1_generated_stats['mean']:.4f}, 最大值: {loss1_generated_stats['max']:.4f}, 标准差: {loss1_generated_stats['std']:.4f}"
            )

            # 下行丢包率统计
            loss2_original_stats = original_df["loss_rate2"].describe()
            loss2_generated_stats = generated_df["loss_rate2"].describe()
            logger.info("下行:")
            logger.info(
                f"参考样本 - 最小值: {loss2_original_stats['min']:.4f}, 平均值: {loss2_original_stats['mean']:.4f}, 最大值: {loss2_original_stats['max']:.4f}, 标准差: {loss2_original_stats['std']:.4f}"
            )
            logger.info(
                f"生成样本 - 最小值: {loss2_generated_stats['min']:.4f}, 平均值: {loss2_generated_stats['mean']:.4f}, 最大值: {loss2_generated_stats['max']:.4f}, 标准差: {loss2_generated_stats['std']:.4f}\n"
            )
        else:
            # 单流数据统计
            # 延迟统计
            delay_original_stats = original_df["delay"].describe()
            delay_generated_stats = generated_df["delay"].describe()

            logger.info("\n时延统计:")
            logger.info(
                f"参考样本 - 最小值: {delay_original_stats['min']:.2f}, 平均值: {delay_original_stats['mean']:.2f}, 最大值: {delay_original_stats['max']:.2f}, 标准差: {delay_original_stats['std']:.2f}"
            )
            logger.info(
                f"生成样本 - 最小值: {delay_generated_stats['min']:.2f}, 平均值: {delay_generated_stats['mean']:.2f}, 最大值: {delay_generated_stats['max']:.2f}, 标准差: {delay_generated_stats['std']:.2f}"
            )

            # 丢包率统计
            loss_original_stats = original_df["loss_rate"].describe()
            loss_generated_stats = generated_df["loss_rate"].describe()

            logger.info("\n丢包率统计:")
            logger.info(
                f"参考样本 - 最小值: {loss_original_stats['min']:.4f}, 平均值: {loss_original_stats['mean']:.4f}, 最大值: {loss_original_stats['max']:.4f}, 标准差: {loss_original_stats['std']:.4f}"
            )
            logger.info(
                f"生成样本 - 最小值: {loss_generated_stats['min']:.4f}, 平均值: {loss_generated_stats['mean']:.4f}, 最大值: {loss_generated_stats['max']:.4f}, 标准差: {loss_generated_stats['std']:.4f}\n"
            )

        logger.info("=" * 50)

    def visualize_batch_results(
        self,
        input_generation_dir: Path
    ) -> None:
        """批量可视化结果

        Args:
            input_generation_dir: 生成样本的目录路径，包含original_sample_6000*.csv和generated_sample_6000*.csv文件
        """
        # 查找所有样本文件（支持批量处理）
        original_files = list(input_generation_dir.glob("original_sample_6000*.csv"))
        generated_files = list(input_generation_dir.glob("generated_sample_6000*.csv"))

        # 排序文件，确保原始文件和生成文件一一对应
        original_files.sort()
        generated_files.sort()

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

        # 处理每组样本
        for i, (original_file, generated_file) in enumerate(zip(original_files, generated_files)):
            # 为每组样本创建独立的输出目录
            # 提取组信息，例如从 "original_sample_6000_group_1_behavior_0.csv" 中提取 "group_1_behavior_0"
            group_info = original_file.stem.split("_")[3:]
            group_dir_name = "_" + "_".join(group_info) if group_info else ""
            group_output_dir = self.output_dir / f"group{group_dir_name}"
            group_output_dir.mkdir(parents=True, exist_ok=True)

            # 创建组可视化器
            group_visualizer = ResultsVisualizer(group_output_dir)

            # 可视化当前组的结果
            # 第一张图片作为主要可视化，生成综合报告所需的图片
            is_main_visualization = (i == 0)
            group_visualizer.visualize_results(original_file, generated_file, is_main_visualization)
