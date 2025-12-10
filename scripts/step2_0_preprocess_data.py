#!/usr/bin/env python3
"""
步骤2.0：预处理训练数据
"""

import sys
import os

# 添加当前目录到Python路径，以便找到config模块
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

import pandas as pd
from pathlib import Path
from network_simulation.utils.logger import get_logger
from config import (
    PATTERNS_DIR,
    PROCESSED_DIR,
    PREPROCESS_DIR,
    CLEANUP_OLD_FILES,
    KEEP_LATEST_FILES
)


# 获取日志记录器
logger = get_logger(__name__)

def preprocess_data(
    input_patterns_dir: Path, input_processed_file: Path, output_preprocess_dir: Path
):
    """预处理扩散模型训练数据

    该函数对网络数据进行预处理，包括行为标签对齐、噪声剔除、特征归一化等，
    为扩散模型训练准备格式化的输入数据。

    Args:
        input_patterns_dir: 行为模式目录路径，包含valid_loss_values.json和behavior_labels_*.json文件
        input_processed_file: 处理后的数据文件路径，包含timestamp、delay、loss_rate等字段
        output_preprocess_dir: 预处理结果的输出目录路径

    Returns:
        Path: 预处理数据文件路径

    Raises:
        FileNotFoundError: 如果输入文件或目录不存在
        ValueError: 如果数据格式不符合要求

    输入输出示例：
        输入：
            input_patterns_dir: Path("data/results/patterns/")
            input_processed_file: Path("data/processed/20251203_230356_b6x-playback_processed.csv")
            output_preprocess_dir: Path("data/results/preprocessed/")
        输出：
            PosixPath('data/results/preprocessed/preprocess_data.npz')
    """
    logger.info(f"正在为扩散模型预处理数据: {input_patterns_dir}")

    # 导入训练数据预处理类
    from network_simulation.condition_generation.training_data_preprocessor import TrainingDataPreprocessor

    # 初始化训练数据预处理类
    preprocessor = TrainingDataPreprocessor()

    # 调用核心模块的预处理函数
    npz_file = preprocessor.preprocess_data(input_patterns_dir, input_processed_file, output_preprocess_dir)

    logger.info("预处理完成！")
    logger.info(f"预处理数据已保存到: {npz_file}")

    return npz_file


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

    解析命令行参数，调用preprocess_data函数预处理扩散模型训练数据。

    命令行参数：
        python scripts/step2_0_preprocess_data.py [input_patterns_dir] [input_processed_file_or_dir] [output_preprocess_dir]

    参数说明：
        input_patterns_dir: 行为模式目录路径，包含行为标签和合法丢包值 (默认: data/results/patterns)
        input_processed_file_or_dir: 处理后的数据文件或目录路径 (默认: data/processed)
        output_preprocess_dir: 预处理结果的输出目录路径 (默认: data/results/preprocess)
    """
    import argparse

    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="预处理扩散模型训练数据")
    parser.add_argument(
        "input_patterns",
        nargs="?",
        type=Path,
        default=PATTERNS_DIR,
        help="行为模式目录路径，包含行为标签和合法丢包值 (默认: data/results/patterns)",
    )
    parser.add_argument(
        "input_processed",
        nargs="?",
        type=Path,
        default=PROCESSED_DIR,
        help="处理后的数据文件或目录路径 (默认: data/processed)",
    )
    parser.add_argument(
        "output_preprocess",
        nargs="?",
        type=Path,
        default=PREPROCESS_DIR,
        help="预处理结果的输出目录路径 (默认: data/results/preprocess)",
    )

    args = parser.parse_args()

    input_patterns_dir = args.input_patterns
    input_processed_path = args.input_processed
    output_preprocess_dir = args.output_preprocess

    # 确保输出目录存在
    output_preprocess_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧文件
    cleanup_old_files(output_preprocess_dir, "preprocess_data.npz")
    cleanup_old_files(output_preprocess_dir, "merged_processed_data.csv")

    # 预处理数据
    if input_processed_path.is_file():
        # 如果输入是一个文件，处理单个文件
        preprocess_data(input_patterns_dir, input_processed_path, output_preprocess_dir)
    elif input_processed_path.is_dir():
        # 如果输入是一个文件夹，合并所有处理后的文件
        processed_files = list(input_processed_path.glob("*.csv"))
        if not processed_files:
            logger.warning(f"在 {input_processed_path} 中未找到 .csv 文件")
            sys.exit(1)

        # 合并所有处理后的文件
        all_processed_df = []
        for processed_file in processed_files:
            df = pd.read_csv(processed_file, parse_dates=["timestamp"])
            all_processed_df.append(df)

        # 合并为一个DataFrame
        merged_processed_df = pd.concat(all_processed_df, ignore_index=True)
        logger.info(
            f"合并了 {len(processed_files)} 个处理后的文件，总样本数: {len(merged_processed_df)}"
        )

        # 保存合并后的文件
        merged_file_path = output_preprocess_dir / "merged_processed_data.csv"
        merged_processed_df.to_csv(merged_file_path, index=False)

        logger.info("\n正在处理行为模式和合并后的数据...")
        # 预处理合并后的数据
        preprocess_data(input_patterns_dir, merged_file_path, output_preprocess_dir)
    else:
        logger.error(f"输入 {input_processed_path} 不是文件或目录")
        sys.exit(1)

    logger.info("步骤2.0：预处理训练数据完成！")


if __name__ == "__main__":
    main()
