#!/usr/bin/env python3
"""
步骤1.3：发现行为模式
"""

import sys
import os

# 添加当前目录到Python路径，以便找到config模块
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

import pandas as pd
import numpy as np
from pathlib import Path
from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
from network_simulation.utils.logger import get_logger
from config import CLEANUP_OLD_FILES, KEEP_LATEST_FILES


# 获取日志记录器
logger = get_logger(__name__)


def discover_patterns(
    input_features_file: Path, input_processed_file: Path, output_patterns_dir: Path
):
    """从网络特征数据中发现行为模式

    该函数使用聚类算法（如HDBSCAN）从网络特征数据中识别不同的网络行为模式，并保存识别结果。

    Args:
        input_features_file: 特征数据文件路径，包含提取的网络特征
        input_processed_file: 处理后的数据文件路径，包含原始网络测量数据
        output_patterns_dir: 行为模式结果的输出目录路径

    Returns:
        Path: 行为模式结果的输出目录路径

    Raises:
        FileNotFoundError: 如果输入文件不存在
        ValueError: 如果数据格式不符合要求

    输入输出示例：
        输入：
            input_features_file: Path("data/results/features/merged_features.csv")
            input_processed_file: Path("data/processed/20251203_230356_b6x-playback_processed.csv")
            output_patterns_dir: Path("data/results/patterns/")
        输出：
            PosixPath('data/results/patterns/')
    """
    logger.info(f"正在从 {input_features_file} 发现行为模式")

    # 加载特征数据
    features_df = pd.read_csv(input_features_file)

    # 加载处理后的数据
    processed_df = pd.read_csv(input_processed_file, parse_dates=["timestamp"])

    # 初始化模式识别器 - 使用规则检测方法
    pattern_identifier = PatternIdentifier(method="rule")

    # 识别行为模式
    patterns = pattern_identifier.identify(features_df, processed_df)

    # 创建输出目录结构
    metadata_dir = output_patterns_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 添加可视化
    from network_simulation.visualization.visualizer import Visualizer

    visualizer = Visualizer(output_patterns_dir)

    # 提取特征矩阵
    feature_columns = [col for col in features_df.columns if col.startswith("feat_")]
    X = features_df[feature_columns].values

    # 生成HTML报告和可视化
    visualizer.generate_html_report(
        patterns, X, feature_columns, raw_data=processed_df, features_df=features_df
    )

    # 保存模式识别结果
    pattern_identifier.save(
        patterns, output_patterns_dir
    )

    logger.info(f"行为模式已保存到: {output_patterns_dir}")
    logger.info(f"上行行为标签长度: {len(patterns['labels_up'])}")
    logger.info(f"上行唯一行为类型: {np.unique(patterns['labels_up'])}")
    logger.info(f"下行行为标签长度: {len(patterns['labels_down'])}")
    logger.info(f"下行唯一行为类型: {np.unique(patterns['labels_down'])}")

    return output_patterns_dir


def cleanup_old_files(directory, pattern):
    """清理目录下的旧文件，只保留最新的KEEP_LATEST_FILES个文件

    Args:
        directory: 要清理的目录
        pattern: 要清理的文件模式
    """

    if not CLEANUP_OLD_FILES:
        return

    files = list(directory.glob(pattern))
    if len(files) <= KEEP_LATEST_FILES:
        return

    # 按修改时间排序，最新的在前
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # 删除旧文件
    for file in files[KEEP_LATEST_FILES:]:
        file.unlink()
        logger.info(f"已清理旧文件: {file}")


def main():
    """主函数入口

    解析命令行参数，调用discover_patterns函数从网络特征数据中发现行为模式。

    命令行参数：
        python scripts/step1_3_discover_patterns.py [input_features_file_or_dir] [input_processed_file_or_dir] [output_patterns_dir]

    参数说明：
        input_features_file_or_dir: 特征数据文件或目录路径 (默认: data/results/features)
        input_processed_file_or_dir: 处理后的数据文件或目录路径 (默认: data/processed)
        output_patterns_dir: 行为模式结果的输出目录路径 (默认: data/results/patterns)
    """
    import argparse
    from config import (
        FEATURES_DIR,
        PROCESSED_DIR,
        PATTERNS_DIR,
    )

    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="从网络特征数据中发现行为模式")
    parser.add_argument(
        "input_features",
        nargs="?",
        type=Path,
        default=FEATURES_DIR,
        help="特征数据文件或目录路径 (默认: data/results/features)",
    )
    parser.add_argument(
        "input_processed",
        nargs="?",
        type=Path,
        default=PROCESSED_DIR,
        help="处理后的数据文件或目录路径 (默认: data/processed)",
    )
    parser.add_argument(
        "output_patterns",
        nargs="?",
        type=Path,
        default=PATTERNS_DIR,
        help="行为模式结果的输出目录路径 (默认: data/results/patterns)",
    )

    args = parser.parse_args()

    input_features_path = args.input_features
    input_processed_path = args.input_processed
    output_patterns_dir = args.output_patterns

    # 确保输出目录存在
    output_patterns_dir.mkdir(parents=True, exist_ok=True)

    # 发现行为模式
    if input_features_path.is_file() and input_processed_path.is_file():
        # 如果输入是两个文件，处理单个文件对
        discover_patterns(
            input_features_path, input_processed_path, output_patterns_dir
        )
    elif input_features_path.is_dir() and input_processed_path.is_dir():
        # 如果输入是两个文件夹，使用合并后的特征文件
        # 检查是否存在合并后的特征文件
        merged_features_file = input_features_path / "merged_features.csv"
        if not merged_features_file.exists():
            logger.error(f"在 {input_features_path} 中未找到 merged_features.csv")
        logger.error("请先运行特征提取脚本生成合并特征文件")
        sys.exit(1)

        # 合并所有处理后的文件
        processed_files = list(input_processed_path.glob("*.csv"))
        if not processed_files:
            logger.error(f"在 {input_processed_path} 中未找到 .csv 文件")
        logger.error("请先运行数据处理脚本生成处理后的数据文件")
        sys.exit(1)

        # 合并所有处理文件，保留每个样本的来源文件信息
        all_processed_df = []
        for processed_file in processed_files:
            df = pd.read_csv(processed_file, parse_dates=["timestamp"])
            # 添加文件来源列
            df["file_source"] = processed_file.stem
            all_processed_df.append(df)
        merged_processed_df = pd.concat(all_processed_df, ignore_index=True)

        logger.info("\n正在使用合并特征和合并处理数据生成行为标签...")
        logger.info(f"合并后的处理数据总样本数: {len(merged_processed_df)}")

        # 临时保存合并后的处理数据
        temp_merged_file = input_processed_path / "merged_processed_data_temp.csv"
        merged_processed_df.to_csv(temp_merged_file, index=False)

        # 使用合并后的特征和合并后的处理数据生成行为标签
        discover_patterns(merged_features_file, temp_merged_file, output_patterns_dir)

        # 删除临时文件
        temp_merged_file.unlink()
    else:
        logger.error(
            f"输入类型不匹配 - 特征输入: {input_features_path.is_file() and '文件' or '目录'}, 处理后数据输入: {input_processed_path.is_file() and '文件' or '目录'}"
        )
        logger.error("请确保两个输入都是文件或都是目录")
        sys.exit(1)

    logger.info("步骤1.3：发现行为模式完成！")


if __name__ == "__main__":
    main()
