#!/usr/bin/env python3
"""
端到端网络行为模式发现与生成管道
"""

import sys
import os
import shutil
from pathlib import Path

# 添加当前目录到Python路径，以便导入其他脚本和network_simulation模块
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

# 导入日志模块
from network_simulation.utils.logger import get_logger

# 导入必要的管道类
from network_simulation.pipeline.behavior_discovery_pipeline import (
    BehaviorDiscoveryPipeline,
)
from network_simulation.pipeline.diffusion_model_pipeline import DiffusionModelPipeline

# step2需要的函数将通过DiffusionModelPipeline调用

# 初始化日志记录器
logger = get_logger(__name__)


def clean_directory(directory: Path):
    """清理目录中的所有文件和子目录

    该函数清理指定目录中的所有内容，如果目录不存在则创建它。

    Args:
        directory: 要清理的目录路径
    """
    if directory.exists():
        logger.info(f"正在清理目录: {directory}")
        for item in directory.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        logger.info(f"目录不存在，正在创建: {directory}")
        directory.mkdir(parents=True, exist_ok=True)


def main():
    """主函数入口

    解析命令行参数，执行完整的端到端网络行为模式发现与生成管道。

    命令行参数：
        python scripts/e2e_pipeline.py <input_raw_file_or_dir>

    参数说明：
        input_raw_file_or_dir: 原始网络数据文件或目录路径
        --steps: 指定要运行的步骤，例如 "1.1,1.2,2.0,2.1"，默认运行所有步骤
        --behavior-only: 只执行行为识别部分，跳过训练扩散模型和生成样本步骤
        --train-only: 只执行训练和生成部分，跳过行为识别步骤
    """
    # 解析命令行参数
    import argparse

    parser = argparse.ArgumentParser(description="端到端网络行为模式发现与生成管道")
    parser.add_argument("input_path", help="原始网络数据文件或目录路径")
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="指定要运行的步骤，例如 '1.1,1.2,2.0,2.1'，默认运行所有步骤",
    )
    parser.add_argument(
        "--behavior-only",
        action="store_true",
        help="只执行行为识别部分，跳过训练扩散模型和生成样本步骤",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="只执行训练和生成部分，跳过行为识别步骤",
    )

    args = parser.parse_args()
    input_path = Path(args.input_path)

    # 解析步骤参数
    selected_steps = None
    if args.steps:
        selected_steps = set(args.steps.split(","))

    # 检查输入路径是否存在
    if not input_path.exists():
        logger.error(f"输入路径 {input_path} 不存在")
        sys.exit(1)

    # 固定输出目录结构
    base_output_dir = Path("data/results/e2e_pipeline")

    # 子目录结构
    processed_dir = base_output_dir / "processed"  # 处理后的数据
    features_dir = base_output_dir / "features"  # 提取的特征
    patterns_dir = base_output_dir / "patterns"  # 发现的行为模式
    preprocess_dir = base_output_dir / "preprocess"  # 预处理的训练数据
    train_dir = base_output_dir / "train_results"  # 训练结果
    generation_dir = base_output_dir / "generation"  # 生成的样本
    visualization_dir = base_output_dir / "visualization"  # 可视化结果
    evaluation_dir = base_output_dir / "evaluation"  # 评估结果

    # 重新创建目录结构
    processed_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    patterns_dir.mkdir(parents=True, exist_ok=True)
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    generation_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("端到端网络行为模式发现与生成管道")
    logger.info(f"输入路径: {input_path}")
    logger.info(f"输出目录: {base_output_dir}")
    logger.info(
        f"参数: behavior-only={args.behavior_only}, train-only={args.train_only}"
    )
    logger.info("=" * 70)

    # 初始化行为发现流程和扩散模型流程
    behavior_pipeline = BehaviorDiscoveryPipeline()
    diffusion_pipeline = DiffusionModelPipeline()

    # 处理后的合并文件路径
    merged_processed_file = None

    # 根据train-only参数决定是否执行行为识别步骤
    if not args.train_only:
        # 清理输出目录（只在完整流程时清理）
        clean_directory(base_output_dir)
        # 重新创建目录结构
        processed_dir.mkdir(parents=True, exist_ok=True)
        features_dir.mkdir(parents=True, exist_ok=True)
        patterns_dir.mkdir(parents=True, exist_ok=True)
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        train_dir.mkdir(parents=True, exist_ok=True)
        generation_dir.mkdir(parents=True, exist_ok=True)
        visualization_dir.mkdir(parents=True, exist_ok=True)
        evaluation_dir.mkdir(parents=True, exist_ok=True)

        # 步骤1.1: 处理原始数据
        if selected_steps is None or "1.1" in selected_steps:
            logger.info("\n步骤1.1: 处理原始数据")
            from network_simulation.data_processing.data_processor import DataProcessor

            data_processor = DataProcessor()
            data_processor.process_raw_data(input_path, processed_dir)
            logger.info("步骤1.1: 处理原始数据完成")

        # 获取合并后的处理数据文件路径
        processed_files = list(processed_dir.glob("*.csv"))
        if processed_files:
            # 如果只有一个处理文件，直接使用；否则合并
            if len(processed_files) == 1:
                merged_processed_file = processed_files[0]
            else:
                # 合并所有处理后的文件
                import pandas as pd

                all_processed_df = []
                for processed_file in processed_files:
                    df = pd.read_csv(processed_file, parse_dates=["timestamp"])
                    all_processed_df.append(df)
                merged_processed_df = pd.concat(all_processed_df, ignore_index=True)
                merged_processed_file = processed_dir / "merged_processed_data.csv"
                merged_processed_df.to_csv(merged_processed_file, index=False)

        # 步骤1.2: 提取特征
        if selected_steps is None or "1.2" in selected_steps:
            logger.info("\n步骤1.2: 提取特征")
            behavior_pipeline.run_step1_2(processed_dir, features_dir)
            logger.info("步骤1.2: 提取特征完成")

        # 步骤1.3: 发现行为模式
        if selected_steps is None or "1.3" in selected_steps:
            logger.info("\n步骤1.3: 发现行为模式")
            behavior_pipeline.run_step1_3(features_dir, processed_dir, patterns_dir)
            logger.info("步骤1.3: 发现行为模式完成")

        # 步骤1.4: 评估行为发现效果
        if selected_steps is None or "1.4" in selected_steps:
            logger.info("\n步骤1.4: 评估行为发现效果")
            # 创建评估结果目录
            pattern_eval_dir = base_output_dir / "pattern_evaluation"
            pattern_eval_dir.mkdir(parents=True, exist_ok=True)
            try:
                behavior_pipeline.run_step1_4(
                    features_dir, patterns_dir, pattern_eval_dir
                )
                logger.info("步骤1.4: 评估行为发现效果完成")
            except Exception as e:
                logger.warning(
                    f"步骤1.4: 评估行为发现效果失败，将跳过此步骤。错误信息: {str(e)}"
                )
    else:
        # 只执行训练时，检查必要的目录是否存在
        if not processed_dir.exists() or not patterns_dir.exists():
            logger.error(
                "train-only模式需要processed_dir和patterns_dir目录存在，请先运行行为识别步骤"
            )
            sys.exit(1)
        # 获取合并后的处理数据文件路径
        processed_files = list(processed_dir.glob("*.csv"))
        if processed_files:
            if len(processed_files) == 1:
                merged_processed_file = processed_files[0]
            else:
                merged_processed_file = processed_dir / "merged_processed_data.csv"
        else:
            logger.error("在processed_dir目录中未找到处理后的数据文件")
            sys.exit(1)

    # 如果不是只执行行为识别，则执行步骤2（训练扩散模型和生成样本）
    if not args.behavior_only:
        # 确保merged_processed_file存在
        if not merged_processed_file or not merged_processed_file.exists():
            logger.error("执行step2需要处理后的数据文件，请确保step1.1已正确执行")
            sys.exit(1)

        # 步骤2.0: 预处理数据
        if selected_steps is None or "2.0" in selected_steps:
            logger.info("\n步骤2.0: 预处理训练数据")
            diffusion_pipeline.run_step2_0(
                patterns_dir, merged_processed_file, preprocess_dir
            )
            logger.info("步骤2.0: 预处理训练数据完成")

        # 步骤2.1: 训练条件扩散模型
        if selected_steps is None or "2.1" in selected_steps:
            logger.info("\n步骤2.1: 训练条件扩散模型")
            diffusion_pipeline.run_step2_1(preprocess_dir, train_dir)
            logger.info("步骤2.1: 训练条件扩散模型完成")

        # 步骤2.2: 生成样本
        if selected_steps is None or "2.2" in selected_steps:
            logger.info("\n步骤2.2: 生成网络状态样本")
            diffusion_pipeline.run_step2_2(
                patterns_dir,
                merged_processed_file,
                train_dir,
                generation_dir,
                preprocess_dir,
            )
            logger.info("步骤2.2: 生成网络状态样本完成")

        # 步骤2.3: 可视化结果对比
        if selected_steps is None or "2.3" in selected_steps:
            logger.info("\n步骤2.3: 可视化结果对比")
            diffusion_pipeline.run_step2_3(generation_dir, visualization_dir)
            logger.info("步骤2.3: 可视化结果对比完成")

        # 步骤2.4: 评估生成样本质量
        if selected_steps is None or "2.4" in selected_steps:
            logger.info("\n步骤2.4: 评估生成样本质量")
            diffusion_pipeline.run_step2_4(generation_dir, train_dir, evaluation_dir)
            logger.info("步骤2.4: 评估生成样本质量完成")
    else:
        logger.info("\n已跳过步骤2（训练扩散模型和生成样本），只执行了行为识别部分")

    logger.info("\n" + "=" * 70)
    logger.info("端到端管道执行完成!")
    logger.info(f"所有结果保存在: {base_output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
