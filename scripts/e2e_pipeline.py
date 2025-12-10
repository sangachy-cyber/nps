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

# 导入各个脚本的核心函数
from step1_1_process_raw import process_raw_data as process_raw_data_func
from step1_2_extract_features import extract_features as extract_features_func
from step1_3_discover_patterns import discover_patterns as discover_patterns_func
from step1_4_evaluate_patterns import evaluate_patterns as evaluate_patterns_func

# step2需要的函数将在需要时动态导入

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
        --behavior-only: 只执行行为识别部分，跳过训练扩散模型和生成样本步骤
    """
    # 解析命令行参数
    import argparse

    parser = argparse.ArgumentParser(description="端到端网络行为模式发现与生成管道")
    parser.add_argument("input_path", help="原始网络数据文件或目录路径")
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
    logger.info(f"参数: behavior-only={args.behavior_only}, train-only={args.train_only}")
    logger.info("=" * 70)

    # 初始化必要的变量
    merged_processed_file = None
    merged_features_file = None

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
        logger.info("\n步骤1.1: 处理原始数据")
        if input_path.is_file():
            process_raw_data_func(input_path, processed_dir)
        elif input_path.is_dir():
            raw_files = list(input_path.glob("*.txt"))
            if not raw_files:
                logger.error(f"警告: 在 {input_path} 中未找到 .txt 文件")
                sys.exit(1)
            for raw_file in raw_files:
                process_raw_data_func(raw_file, processed_dir)
        logger.info("步骤1.1: 处理原始数据完成")

        # 步骤1.2: 提取特征
        logger.info("\n步骤1.2: 提取特征")
        # 合并所有处理后的文件
        processed_files = list(processed_dir.glob("*.csv"))
        if not processed_files:
            logger.error(f"在 {processed_dir} 中未找到 .csv 文件")
            sys.exit(1)

        # 如果processed_dir中只有一个文件，直接处理它
        if len(processed_files) == 1:
            extract_features_func(processed_files[0], features_dir)
        else:
            # 如果有多个文件，我们需要先合并它们，然后处理合并后的文件
            import pandas as pd

            all_processed_df = []
            for processed_file in processed_files:
                df = pd.read_csv(processed_file, parse_dates=["timestamp"])
                # 添加文件来源列
                df["file_source"] = processed_file.stem
                all_processed_df.append(df)

            # 合并为一个DataFrame
            merged_processed_df = pd.concat(all_processed_df, ignore_index=True)
            logger.info(
                f"合并了 {len(processed_files)} 个处理后的文件，总样本数: {len(merged_processed_df)}"
            )

            # 保存合并后的文件
            merged_processed_file = processed_dir / "merged_processed_data.csv"
            merged_processed_df.to_csv(merged_processed_file, index=False)

            # 对合并后的文件提取特征
            logger.info("\n正在对合并后的文件提取特征...")
            extract_features_func(merged_processed_file, features_dir)

            # 保留文件，后面步骤1.3还需要使用
        logger.info("步骤1.2: 提取特征完成")

        # 步骤1.3: 发现行为模式
        logger.info("\n步骤1.3: 发现行为模式")
        # 查找合并后的特征文件
        merged_features_file = features_dir / "merged_features.csv"
        if not merged_features_file.exists():
            # 如果没有合并后的特征文件，使用第一个特征文件
            merged_features_file = next(features_dir.glob("*.csv"))
        # 合并所有处理后的文件用于发现模式
        processed_files = list(processed_dir.glob("*.csv"))
        import pandas as pd
        all_processed_df = []
        for processed_file in processed_files:
            df = pd.read_csv(processed_file, parse_dates=["timestamp"])
            # 添加文件来源列
            df["file_source"] = processed_file.stem
            all_processed_df.append(df)

        merged_processed_df = pd.concat(all_processed_df, ignore_index=True)
        # 保存合并后的处理数据
        merged_processed_file = processed_dir / "merged_processed_data.csv"
        merged_processed_df.to_csv(merged_processed_file, index=False)
        discover_patterns_func(merged_features_file, merged_processed_file, patterns_dir)
        logger.info("步骤1.3: 发现行为模式完成")

        # 步骤1.4: 评估行为发现效果
        logger.info("\n步骤1.4: 评估行为发现效果")
        # 创建评估结果目录
        pattern_eval_dir = base_output_dir / "pattern_evaluation"
        pattern_eval_dir.mkdir(parents=True, exist_ok=True)
        try:
            evaluate_patterns_func(merged_features_file, patterns_dir, pattern_eval_dir)
            logger.info("步骤1.4: 评估行为发现效果完成")
        except Exception as e:
            logger.warning(f"步骤1.4: 评估行为发现效果失败，将跳过此步骤。错误信息: {str(e)}")
    else:
        # 只执行训练时，直接获取现有的文件
        logger.info("\n跳过行为识别步骤，直接执行训练和生成")

        # 检查必要的文件是否存在
        required_files = []

        # 检查处理文件
        processed_files = list(processed_dir.glob("*.csv"))
        if not processed_files:
            required_files.append(f"{processed_dir}/*.csv")
        else:
            merged_processed_file = processed_dir / "merged_processed_data.csv"
            if not merged_processed_file.exists():
                # 如果没有合并文件，使用第一个文件
                merged_processed_file = processed_files[0]
            logger.info(f"使用现有的处理文件: {merged_processed_file}")

        # 检查特征文件
        features_files = list(features_dir.glob("*.csv"))
        if not features_files:
            required_files.append(f"{features_dir}/*.csv")
        else:
            merged_features_file = features_files[0]
            logger.info(f"使用现有的特征文件: {merged_features_file}")

        # 检查行为模式文件
        window_features_file = patterns_dir / "window_features_rule.npy"
        if not window_features_file.exists():
            required_files.append(f"{window_features_file}")

        # 如果缺少必要文件，报错并退出
        if required_files:
            logger.error("train-only模式下缺少必要的文件，这些文件需要先通过完整流程生成:")
            for file in required_files:
                logger.error(f"  - {file}")
            logger.error("请先运行完整流程: python scripts/e2e_pipeline.py data/raw")
            sys.exit(1)

    # 如果不是只执行行为识别，则执行步骤2（训练扩散模型和生成样本）
    if not args.behavior_only:
        # 动态导入step2需要的函数
        from step2_0_preprocess_data import preprocess_data as preprocess_data_func
        from step2_1_train_model import train_model as train_model_func
        from step2_2_generate_samples import generate_samples as generate_samples_func
        from network_simulation.visualization.results_visualizer import ResultsVisualizer
        from step2_4_evaluate_generation import main as evaluate_generation_main

        # 步骤2.0: 预处理数据
        logger.info("\n步骤2.0: 预处理训练数据")
        preprocess_data_func(patterns_dir, merged_processed_file, preprocess_dir)
        logger.info("步骤2.0: 预处理训练数据完成")

        # 步骤2.1: 训练条件扩散模型
        logger.info("\n步骤2.1: 训练条件扩散模型")
        train_model_func(preprocess_dir, train_dir)
        logger.info("步骤2.1: 训练条件扩散模型完成")

        # 步骤2.2: 生成样本
        logger.info("\n步骤2.2: 生成网络状态样本")
        model_path = train_dir / "diffusion_model_final.pth"
        generate_samples_func(
            merged_processed_file, patterns_dir, model_path, generation_dir
        )
        logger.info("步骤2.2: 生成网络状态样本完成")

        # 步骤2.3: 可视化结果对比
        logger.info("\n步骤2.3: 可视化结果对比")
        # 查找生成的样本文件
        original_files = list(generation_dir.glob("original_sample_*.csv"))
        generated_files = list(generation_dir.glob("generated_sample_*.csv"))
        if original_files and generated_files:
            # 排序文件，确保一一对应
            original_files.sort()
            generated_files.sort()
            for i, (original_file, generated_file) in enumerate(zip(original_files, generated_files)):
                # 为每组样本创建独立的输出目录
                group_info = original_file.stem.split("_")[3:]
                group_dir_name = "_" + "_".join(group_info) if group_info else ""
                group_output_dir = visualization_dir / f"group{group_dir_name}"
                group_output_dir.mkdir(parents=True, exist_ok=True)
                # 第一张图片作为主要可视化，生成综合报告所需的图片
                is_main_visualization = (i == 0)
                # 创建ResultsVisualizer实例并调用visualize_results方法
                visualizer = ResultsVisualizer(group_output_dir)
                visualizer.visualize_results(original_file, generated_file, is_main_visualization)
        logger.info("步骤2.3: 可视化结果对比完成")

        # 步骤2.4: 评估生成样本质量
        logger.info("\n步骤2.4: 评估生成样本质量")
        # 使用evaluate_generation.py的main函数，它会处理所有样本
        # 保存原始sys.argv，然后修改它以调用evaluate_generation_main
        original_argv = sys.argv.copy()
        sys.argv = [
            "step2_4_evaluate_generation.py",
            str(generation_dir),
            str(evaluation_dir),
        ]
        evaluate_generation_main()
        # 恢复原始sys.argv
        sys.argv = original_argv
        logger.info("步骤2.4: 评估生成样本质量完成")
    else:
        logger.info("\n已跳过步骤2（训练扩散模型和生成样本），只执行了行为识别部分")

    logger.info("\n" + "=" * 70)
    logger.info("端到端管道执行完成!")
    logger.info(f"所有结果保存在: {base_output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
