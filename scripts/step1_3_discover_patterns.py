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
import json
from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier


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
    print(f"正在从 {input_features_file} 发现行为模式")

    # 加载特征数据
    features_df = pd.read_csv(input_features_file)

    # 加载处理后的数据
    processed_df = pd.read_csv(input_processed_file, parse_dates=["timestamp"])

    # 加载合法丢包值 - 优先使用合并后的文件
    merged_valid_loss_file = (
        input_features_file.parent / "merged_valid_loss_values.json"
    )
    valid_loss_values_file = input_features_file.parent / "valid_loss_values.json"

    if merged_valid_loss_file.exists():
        # 使用合并后的合法丢包值
        with open(merged_valid_loss_file, "r") as f:
            valid_loss_values = json.load(f)
        print(f"使用合并后的合法丢包值: {valid_loss_values}")
    elif valid_loss_values_file.exists():
        # 否则使用单文件的合法丢包值
        with open(valid_loss_values_file, "r") as f:
            valid_loss_values = json.load(f)
        print(f"使用单文件的合法丢包值: {valid_loss_values}")
    else:
        # 如果都不存在，从数据中提取
        print("未找到valid_loss_values.json，从数据中提取...")
        # 从处理后的数据中提取合法丢包值
        processed_df = pd.read_csv(input_processed_file)
        if 'loss_rate' in processed_df.columns:
            loss_rates = processed_df['loss_rate'].values
            # 去重并排序
            valid_loss_values = sorted(list(set(loss_rates)))
            print(f"从数据中提取的合法丢包值: {valid_loss_values}")
        else:
            # 如果无法提取，使用数据的统计特征生成
            print("无法从数据中提取丢包率列，使用统计特征生成...")
            valid_loss_values = [0.0]  # 至少包含0.0
            print(f"生成的合法丢包值: {valid_loss_values}")

    # 初始化模式识别器
    pattern_identifier = PatternIdentifier(method="hdbscan")

    # 识别行为模式
    patterns = pattern_identifier.identify(features_df, processed_df)

    # 创建输出目录结构
    metadata_dir = output_patterns_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 保存模式识别结果
    pattern_identifier.save(
        patterns, output_patterns_dir, features_df, valid_loss_values
    )

    print(f"行为模式已保存到: {output_patterns_dir}")
    print(f"行为标签长度: {len(patterns['labels'])}")
    print(f"唯一行为类型: {np.unique(patterns['labels'])}")

    return output_patterns_dir


def cleanup_old_files(directory, pattern):
    """清理目录下的旧文件，只保留最新的KEEP_LATEST_FILES个文件

    Args:
        directory: 要清理的目录
        pattern: 要清理的文件模式
    """
    from config import CLEANUP_OLD_FILES, KEEP_LATEST_FILES

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
        print(f"已清理旧文件: {file}")


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
        DEFAULT_FEATURES_FILE,
        DEFAULT_PROCESSED_FILE,
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

    # 清理旧文件
    cleanup_old_files(output_patterns_dir, "behavior_labels_hdbscan.json")
    cleanup_old_files(output_patterns_dir, "behavior_hdbscan_model.pkl")
    cleanup_old_files(output_patterns_dir, "behavior_transition_graph_hdbscan.json")

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
            print(f"错误: 在 {input_features_path} 中未找到 merged_features.csv")
            print(f"请先运行特征提取脚本生成合并特征文件")
            sys.exit(1)

        # 合并所有处理后的文件
        processed_files = list(input_processed_path.glob("*.csv"))
        if not processed_files:
            print(f"错误: 在 {input_processed_path} 中未找到 .csv 文件")
            print(f"请先运行数据处理脚本生成处理后的数据文件")
            sys.exit(1)

        # 合并所有处理文件
        all_processed_df = []
        for processed_file in processed_files:
            df = pd.read_csv(processed_file, parse_dates=["timestamp"])
            all_processed_df.append(df)
        merged_processed_df = pd.concat(all_processed_df, ignore_index=True)

        print(f"\n正在使用合并特征和合并处理数据生成行为标签...")
        print(f"合并后的处理数据总样本数: {len(merged_processed_df)}")
        
        # 临时保存合并后的处理数据
        temp_merged_file = input_processed_path / "merged_processed_data_temp.csv"
        merged_processed_df.to_csv(temp_merged_file, index=False)
        
        # 使用合并后的特征和合并后的处理数据生成行为标签
        discover_patterns(
            merged_features_file, temp_merged_file, output_patterns_dir
        )
        
        # 删除临时文件
        temp_merged_file.unlink()
    else:
        print(f"错误: 输入类型不匹配 - 特征输入: {input_features_path.is_file() and '文件' or '目录'}, 处理后数据输入: {input_processed_path.is_file() and '文件' or '目录'}")
        print(f"请确保两个输入都是文件或都是目录")
        sys.exit(1)

    print("步骤1.3：发现行为模式完成！")


if __name__ == "__main__":
    main()
