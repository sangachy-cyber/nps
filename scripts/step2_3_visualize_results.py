#!/usr/bin/env python3
"""
步骤2.3：可视化结果
"""

import sys
import os

sys.path.append(os.path.abspath("src"))

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from network_simulation.utils.logger import get_logger

# 初始化日志记录器
logger = get_logger(__name__)

# 设置中文支持，兼容Windows、MacOS和Linux，Linux系统优先使用文泉驿正黑
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def visualize_results(
    input_original_file: Path,
    input_generated_file: Path,
    output_visualization_dir: Path,
):
    """可视化生成结果与原始数据的对比

    该函数生成多种可视化图表，对比原始网络数据和生成的网络数据，
    包括时间序列对比、统计分布对比和相关性对比。

    Args:
        input_original_file: 原始样本数据文件路径
        input_generated_file: 生成样本数据文件路径
        output_visualization_dir: 可视化结果的输出目录路径

    Returns:
        Path: 可视化结果的输出目录路径

    Raises:
        FileNotFoundError: 如果输入文件不存在
        ValueError: 如果数据格式不符合要求

    输入输出示例：
        输入：
            input_original_file: Path("data/generated/original_sample_6000.csv")
            input_generated_file: Path("data/generated/generated_sample_6000.csv")
            output_visualization_dir: Path("data/results/visualizations/")
        输出：
            PosixPath('data/results/visualizations/')
    """
    logger.info(f"正在可视化 {input_original_file} 和 {input_generated_file} 的对比结果")

    # 加载原始数据和生成数据
    original_df = pd.read_csv(input_original_file, parse_dates=["timestamp"])
    generated_df = pd.read_csv(input_generated_file, parse_dates=["timestamp"])

    # 创建可视化输出目录
    output_visualization_dir.mkdir(parents=True, exist_ok=True)

    # 1. 绘制时间序列对比图
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 1, 1)
    plt.plot(
        original_df["timestamp"], original_df["delay"], label="原始数据", alpha=0.7
    )
    plt.plot(
        generated_df["timestamp"], generated_df["delay"], label="生成数据", alpha=0.7
    )
    plt.title("时延对比")
    plt.xlabel("时间")
    plt.ylabel("时延 (ms)")
    plt.legend()
    plt.grid(True)

    # 绘制丢包率对比
    plt.subplot(2, 1, 2)
    plt.plot(
        original_df["timestamp"], original_df["loss_rate"], label="原始数据", alpha=0.7
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
    comparison_plot = output_visualization_dir / "comparison.png"
    plt.savefig(comparison_plot, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"时间序列对比图已保存到: {comparison_plot}")

    # 2. 绘制统计分布对比图
    plt.figure(figsize=(15, 6))

    # 时延直方图
    plt.subplot(2, 2, 1)
    plt.hist(original_df["delay"], bins=50, alpha=0.7, label="原始数据")
    plt.hist(generated_df["delay"], bins=50, alpha=0.7, label="生成数据")
    plt.title("时延分布")
    plt.xlabel("时延 (ms)")
    plt.ylabel("频率")
    plt.legend()
    plt.grid(True)

    # 丢包率直方图
    plt.subplot(2, 2, 2)
    plt.hist(original_df["loss_rate"], bins=20, alpha=0.7, label="原始数据")
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
        [original_df["delay"], generated_df["delay"]], labels=["原始数据", "生成数据"]
    )
    plt.title("时延箱线图")
    plt.ylabel("时延 (ms)")
    plt.grid(True)

    # 丢包率箱线图
    plt.subplot(2, 2, 4)
    plt.boxplot(
        [original_df["loss_rate"], generated_df["loss_rate"]],
        labels=["原始数据", "生成数据"],
    )
    plt.title("丢包率箱线图")
    plt.ylabel("丢包率")
    plt.ylim(0, 1)  # 设置丢包率y轴范围为0-1
    plt.grid(True)

    plt.tight_layout()
    statistics_plot = output_visualization_dir / "statistics.png"
    plt.savefig(statistics_plot, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"统计分布对比图已保存到: {statistics_plot}")

    # 3. 绘制相关性对比图
    plt.figure(figsize=(15, 6))

    # 原始数据相关性
    plt.subplot(1, 2, 1)
    plt.scatter(original_df["delay"], original_df["loss_rate"], alpha=0.5, s=10)
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
    correlation_plot = output_visualization_dir / "correlation.png"
    plt.savefig(correlation_plot, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"相关性对比图已保存到: {correlation_plot}")

    # 4. 计算并打印统计指标对比
    logger.info("\n统计对比:")
    logger.info("=" * 50)

    # 延迟统计
    delay_original_stats = original_df["delay"].describe()
    delay_generated_stats = generated_df["delay"].describe()

    logger.info("\n时延统计:")
    logger.info(
        f"原始数据 - 最小值: {delay_original_stats['min']:.2f}, 平均值: {delay_original_stats['mean']:.2f}, 最大值: {delay_original_stats['max']:.2f}, 标准差: {delay_original_stats['std']:.2f}"
    )
    logger.info(
        f"生成数据 - 最小值: {delay_generated_stats['min']:.2f}, 平均值: {delay_generated_stats['mean']:.2f}, 最大值: {delay_generated_stats['max']:.2f}, 标准差: {delay_generated_stats['std']:.2f}"
    )

    # 丢包率统计
    loss_original_stats = original_df["loss_rate"].describe()
    loss_generated_stats = generated_df["loss_rate"].describe()

    logger.info("\n丢包率统计:")
    logger.info(
        f"原始数据 - 最小值: {loss_original_stats['min']:.4f}, 平均值: {loss_original_stats['mean']:.4f}, 最大值: {loss_original_stats['max']:.4f}, 标准差: {loss_original_stats['std']:.4f}"
    )
    logger.info(
        f"生成数据 - 最小值: {loss_generated_stats['min']:.4f}, 平均值: {loss_generated_stats['mean']:.4f}, 最大值: {loss_generated_stats['max']:.4f}, 标准差: {loss_generated_stats['std']:.4f}\n"
    )

    logger.info("=" * 50)
    logger.info("可视化完成！")

    return output_visualization_dir


def main():
    """主函数入口

    解析命令行参数，调用visualize_results函数可视化生成结果与原始数据的对比。

    命令行参数：
        python scripts/step2_3_visualize_results.py <input_generation_dir> <output_visualization_dir>

    参数说明：
        input_generation_dir: 生成样本的目录路径，包含original_sample_6000*.csv和generated_sample_6000*.csv文件
        output_visualization_dir: 可视化结果的输出目录路径
    """
    if len(sys.argv) < 3:
        logger.error(
            "用法: python scripts/step2_3_visualize_results.py <input_generation_dir> <output_visualization_dir>"
        )
        sys.exit(1)

    input_generation_dir = Path(sys.argv[1])
    output_visualization_dir = Path(sys.argv[2])

    # 确保输出目录存在
    output_visualization_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有样本文件（支持批量处理）
    original_files = list(input_generation_dir.glob("original_sample_6000*.csv"))
    generated_files = list(input_generation_dir.glob("generated_sample_6000*.csv"))

    # 排序文件，确保原始文件和生成文件一一对应
    original_files.sort()
    generated_files.sort()

    if not original_files:
        logger.error(f"错误: 在 {input_generation_dir} 中未找到原始样本文件")
        sys.exit(1)

    if not generated_files:
        logger.error(f"错误: 在 {input_generation_dir} 中未找到生成样本文件")
        sys.exit(1)

    if len(original_files) != len(generated_files):
        logger.warning(
            f"警告: 原始样本文件数 ({len(original_files)}) 与生成样本文件数 ({len(generated_files)}) 不匹配"
        )

    # 处理每组样本
    for original_file, generated_file in zip(original_files, generated_files):
        # 为每组样本创建独立的输出目录
        # 提取组信息，例如从 "original_sample_6000_group_1_behavior_0.csv" 中提取 "group_1_behavior_0"
        group_info = original_file.stem.split("_")[3:]
        group_dir_name = "_" + "_".join(group_info) if group_info else ""
        group_output_dir = output_visualization_dir / f"group{group_dir_name}"
        group_output_dir.mkdir(parents=True, exist_ok=True)

        # 可视化当前组的结果
        visualize_results(original_file, generated_file, group_output_dir)

    logger.info("步骤2.3：可视化结果完成！")


if __name__ == "__main__":
    main()
