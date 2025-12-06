#!/usr/bin/env python3
"""
步骤1.2：提取特征
"""

import sys
import os

sys.path.append(os.path.abspath("src"))

import pandas as pd
from pathlib import Path
from network_simulation.feature_extraction.feature_extractor import FeatureExtractor
from network_simulation.utils.logger import get_logger


# 获取日志记录器
logger = get_logger(__name__)

def extract_features(input_file: Path, output_dir: Path):
    """从处理后的数据中提取网络特征

    该函数使用滑动窗口从处理后的网络数据中提取特征，包括延迟、丢包率等统计特征。

    Args:
        input_file: 处理后的数据文件路径，包含timestamp、delay、loss_rate等字段
        output_dir: 特征数据的输出目录路径

    Returns:
        tuple: 包含特征文件路径和合法丢包值文件路径的元组

    Raises:
        FileNotFoundError: 如果输入文件不存在
        ValueError: 如果数据格式不符合要求

    输入输出示例：
        输入：
            input_file: Path("data/processed/20251203_230356_b6x-playback_processed.csv")
            output_dir: Path("data/results/features/")
        输出：
            (PosixPath('data/results/features/20251203_230356_b6x-playback_processed_features.csv'),
             PosixPath('data/results/features/valid_loss_values.json'))
    """
    logger.info(f"正在从 {input_file} 提取特征")

    # 加载处理后的数据
    df = pd.read_csv(input_file, parse_dates=["timestamp"])

    # 初始化特征提取器
    feature_extractor = FeatureExtractor()

    # 提取网络特征
    features_df = feature_extractor.extract(df)

    # 添加文件标识
    features_df["file_id"] = input_file.stem

    # 检查滑动窗口完整性
    total_samples = len(df)
    window_samples = feature_extractor.window_samples
    slide_samples = feature_extractor.slide_samples
    num_windows = (total_samples - window_samples) // slide_samples + 1

    logger.info(f"  文件: {input_file.name}")
    logger.info(f"  总样本数: {total_samples}")
    logger.info(f"  窗口样本数: {window_samples}")
    logger.info(f"  滑动步长: {slide_samples}")
    logger.info(f"  窗口数量: {num_windows}")
    logger.info(f"  特征数据形状: {features_df.shape}")
    logger.info(f"  期望窗口数: {num_windows}")

    # 验证滑动窗口数量是否匹配
    if len(features_df) != num_windows:
        logger.warning(f"  警告: 期望 {num_windows} 个窗口，但实际得到 {len(features_df)} 个")

    # 保存特征数据
    features_output_file = output_dir / f"{input_file.stem}_features.csv"
    features_df.to_csv(features_output_file, index=False)

    # 保存合法丢包值
    valid_loss_values_output = output_dir / "valid_loss_values.json"
    import json

    with open(valid_loss_values_output, "w") as f:
        json.dump(feature_extractor.valid_loss_values, f, indent=2)

    logger.info(f"特征数据已保存到: {features_output_file}")
    logger.info(f"合法丢包值已保存到: {valid_loss_values_output}")
    logger.info(f"合法丢包值: {feature_extractor.valid_loss_values}")

    return features_output_file, valid_loss_values_output


def main():
    """主函数入口

    解析命令行参数，调用extract_features函数从处理后的网络数据中提取特征。

    命令行参数：
        python scripts/step1_2_extract_features.py <input_processed_file_or_dir> <output_features_dir>

    参数说明：
        input_processed_file_or_dir: 处理后的数据文件或目录路径
        output_features_dir: 特征数据的输出目录路径
    """
    if len(sys.argv) < 3:
        logger.error(
        "用法: python scripts/step1_2_extract_features.py <input_processed_file_or_dir> <output_features_dir>"
    )
    sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 提取特征
    all_features_dfs = []
    final_valid_loss_values = []

    if input_path.is_file():
        # 如果输入是文件，处理单个文件
        features_file, valid_loss_file = extract_features(input_path, output_dir)
        # 合并特征文件
        all_features_dfs.append(pd.read_csv(features_file))
        with open(valid_loss_file, "r") as f:
            import json

            final_valid_loss_values = json.load(f)
    elif input_path.is_dir():
        # 如果输入是文件夹，处理所有.csv文件
        processed_files = list(input_path.glob("*.csv"))
        if not processed_files:
            logger.warning(f"在 {input_path} 中未找到 .csv 文件")
        sys.exit(1)

        # 首先合并所有处理后的文件
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
        merged_processed_file = output_dir / "merged_processed_data_temp.csv"
        merged_processed_df.to_csv(merged_processed_file, index=False)

        # 对合并后的文件提取特征
        logger.info("\n正在对合并后的文件提取特征...")
        features_file, valid_loss_file = extract_features(merged_processed_file, output_dir)

        # 收集特征文件
        all_features_dfs.append(pd.read_csv(features_file))
        with open(valid_loss_file, "r") as f:
            import json
            final_valid_loss_values = json.load(f)

        # 删除临时文件
        merged_processed_file.unlink()

        # 保存合并后的特征文件（重命名）
        merged_features_file = output_dir / "merged_features.csv"
        import shutil
        shutil.copy2(features_file, merged_features_file)

        # 保存合并后的合法丢包值
        merged_valid_loss_file = output_dir / "merged_valid_loss_values.json"
        with open(merged_valid_loss_file, "w") as f:
            json.dump(final_valid_loss_values, f, indent=2)

        logger.info(f"\n已将合并数据的特征保存到 {merged_features_file}")
        logger.info(f"合并后的合法丢包值: {final_valid_loss_values}")
    else:
        logger.error(f"输入 {input_path} 不是文件或目录")
        sys.exit(1)

    logger.info("步骤1.2：提取特征完成！")


if __name__ == "__main__":
    main()
