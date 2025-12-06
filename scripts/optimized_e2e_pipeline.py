#!/usr/bin/env python3
"""
优化后的端到端网络行为模式发现与生成管道
减少文件I/O操作，提高整体性能
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict

# 添加src目录到Python路径
sys.path.append(os.path.abspath("src"))

import pandas as pd
import json

from network_simulation.feature_extraction.feature_extractor import FeatureExtractor
from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
from network_simulation.utils.logger import get_logger

# 初始化日志记录器
logger = get_logger(__name__)

# 移除未使用的导入，只保留需要的模块


def process_raw_data(input_file: Path) -> pd.DataFrame:
    """处理单个原始网络数据文件，返回处理后的数据"""
    logger.info(f"正在处理原始数据文件: {input_file}")

    # 读取原始文件内容
    try:
        with open(input_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"原始数据文件不存在: {input_file}")
    except PermissionError:
        raise PermissionError(f"没有读取文件的权限: {input_file}")

    # 跳过前12行元数据和分隔线，只保留数据行
    if len(lines) < 13:
        raise ValueError(f"文件格式不正确，缺少足够的元数据行: {input_file}")

    data_lines = lines[12:]

    # 提取开始时间
    try:
        start_time_line = [line for line in lines if "Start Time" in line][0]
        start_time_str = start_time_line.split(": ", 1)[1].strip()
        start_time = pd.Timestamp(start_time_str)
    except (IndexError, ValueError) as e:
        raise ValueError(f"无法提取开始时间: {e}")

    # 提取时间间隔（秒）
    try:
        interval_line = [line for line in lines if "Interval(sec)" in line][0]
        interval_sec = float(interval_line.split(": ", 1)[1].strip())
    except (IndexError, ValueError) as e:
        raise ValueError(f"无法提取时间间隔: {e}")

    # 解析数据行
    data = []
    for i, line in enumerate(data_lines):
        line = line.strip()
        if not line:
            continue

        # 分割数据行，按逗号分隔
        parts = line.split(",")
        if len(parts) < 6:
            continue

        try:
            # 提取Delay1、Loss1和Bandwidth1
            delay = float(parts[0])
            loss_rate = float(parts[1]) / 100  # 转换为0-1范围
            bandwidth1 = float(parts[2])
        except ValueError:
            logger.warning(f"警告: 第{i}行的数据格式不正确，跳过该行")
            continue

        # 当遇到时延大于2000ms的数据时，忽略它和它之后的所有数据
        if delay > 2000:
            logger.info(f"在第{i}行发现时延大于2000ms，停止处理")
            break

        # 当Bandwidth1为0时，设置loss_rate为1.0（100%丢包）
        if bandwidth1 == 0:
            loss_rate = 1.0

        # 计算该数据点的时间戳
        timestamp = start_time + pd.Timedelta(seconds=interval_sec * i)

        data.append({"timestamp": timestamp, "delay": delay, "loss_rate": loss_rate})

    if not data:
        raise ValueError(f"没有提取到有效数据: {input_file}")

    # 创建DataFrame
    df = pd.DataFrame(data)
    logger.info(f"处理后数据形状: {df.shape}")
    logger.info(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    logger.info(f"时延范围: {df['delay'].min():.2f} 到 {df['delay'].max():.2f} ms")
    logger.info(f"丢包率范围: {df['loss_rate'].min():.4f} 到 {df['loss_rate'].max():.4f}")

    return df


def setup_directories() -> Tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    """设置输出目录结构"""
    # 从配置文件加载目录结构
    try:
        from src.network_simulation.config import (
            RESULTS_DIR, MODELS_DIR
        )
        # 使用配置文件中的目录结构
        base_output_dir = RESULTS_DIR / "optimized_e2e_pipeline"
        processed_dir = base_output_dir / "processed"  # 处理后的数据
        features_dir = base_output_dir / "features"  # 提取的特征
        patterns_dir = base_output_dir / "patterns"  # 发现的行为模式
        preprocess_dir = base_output_dir / "preprocess"  # 预处理的训练数据
        train_dir = MODELS_DIR / "diffusion_model"  # 训练结果
        generation_dir = base_output_dir / "generated"  # 生成的样本
        visualization_dir = base_output_dir / "visualization"  # 可视化结果
    except ImportError:
        # 导入失败时使用默认目录结构
        base_output_dir = Path("data/results/optimized_e2e_pipeline")
        processed_dir = base_output_dir / "processed"
        features_dir = base_output_dir / "features"
        patterns_dir = base_output_dir / "patterns"
        preprocess_dir = base_output_dir / "preprocess"
        train_dir = Path("data/models/diffusion_model")
        generation_dir = base_output_dir / "generated"
        visualization_dir = base_output_dir / "visualization"

    # 创建输出目录
    processed_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    patterns_dir.mkdir(parents=True, exist_ok=True)
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    generation_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    return base_output_dir, processed_dir, features_dir, patterns_dir, preprocess_dir, train_dir, generation_dir, visualization_dir


def process_input_data(input_path: Path) -> pd.DataFrame:
    """处理输入数据，返回合并后的处理数据"""
    all_processed_dfs = []

    if input_path.is_file():
        # 处理单个文件
        processed_df = process_raw_data(input_path)
        all_processed_dfs.append(processed_df)
    elif input_path.is_dir():
        # 处理目录下所有.txt文件
        raw_files = list(input_path.glob("*.txt"))
        if not raw_files:
            logger.error(f"警告: 在 {input_path} 中未找到 .txt 文件")
            sys.exit(1)

        for raw_file in raw_files:
            logger.info(f"\n正在处理 {raw_file}...")
            processed_df = process_raw_data(raw_file)
            all_processed_dfs.append(processed_df)

    # 合并所有处理后的数据
    merged_processed_df = pd.concat(all_processed_dfs, ignore_index=True)
    logger.info(f"\n合并了 {len(all_processed_dfs)} 个文件，总样本数: {len(merged_processed_df)}")

    return merged_processed_df


def extract_and_save_features(merged_processed_df: pd.DataFrame, features_dir: Path) -> Tuple[pd.DataFrame, List[float]]:
    """提取特征并保存"""
    logger.info("开始提取特征")
    feature_extractor = FeatureExtractor()
    features_df = feature_extractor.extract(merged_processed_df)
    logger.info(f"特征提取完成，特征数据形状: {features_df.shape}")

    # 保存特征数据（可选，用于调试和可视化）
    features_file = features_dir / "merged_features.csv"
    features_df.to_csv(features_file, index=False)
    logger.info(f"特征数据已保存到: {features_file}")

    # 保存合法丢包值
    valid_loss_values = feature_extractor.valid_loss_values
    valid_loss_file = features_dir / "valid_loss_values.json"
    with open(valid_loss_file, "w") as f:
        json.dump(valid_loss_values, f, indent=2)
    logger.info(f"合法丢包值已保存到: {valid_loss_file}")

    return features_df, valid_loss_values


def discover_and_save_patterns(features_df: pd.DataFrame, merged_processed_df: pd.DataFrame, patterns_dir: Path) -> Dict:
    """发现行为模式并保存"""
    logger.info("开始发现行为模式")
    pattern_identifier = PatternIdentifier(method="hdbscan")
    patterns = pattern_identifier.identify(features_df, merged_processed_df)
    logger.info(f"行为模式发现完成，发现 {patterns['metrics']['num_clusters']} 种行为模式")

    # 保存行为模式结果
    labels_file = patterns_dir / f"behavior_labels_{patterns['method']}.json"
    with open(labels_file, "w") as f:
        json.dump(
            {
                "method": patterns["method"],
                "labels": patterns["labels"],
                "metrics": patterns["metrics"],
                "cluster_stats": patterns["cluster_stats"],
            },
            f,
            indent=2,
        )
    logger.info(f"行为标签已保存到: {labels_file}")

    transition_file = patterns_dir / f"behavior_transition_graph_{patterns['method']}.json"
    with open(transition_file, "w") as f:
        json.dump(
            {
                "transition_matrix": patterns["transition_matrix"],
                "num_clusters": patterns["metrics"]["num_clusters"],
                "method": patterns["method"],
            },
            f,
            indent=2,
        )
    logger.info(f"行为转移图已保存到: {transition_file}")

    return patterns


def run_subprocess(cmd: List[str]) -> None:
    """运行子进程并检查结果"""
    logger.debug(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"错误: 执行命令 {' '.join(cmd)} 失败")
        logger.error(f"标准输出: {result.stdout}")
        logger.error(f"标准错误: {result.stderr}")
        sys.exit(1)
    logger.debug(f"命令执行成功，输出: {result.stdout}")


# 确保src目录在Python路径中
import sys
import os
if os.path.abspath('src') not in sys.path:
    sys.path.append(os.path.abspath('src'))

# 确保scripts目录在Python路径中，以便直接导入脚本
if os.path.abspath('scripts') not in sys.path:
    sys.path.append(os.path.abspath('scripts'))

# 导入各个脚本的核心函数
from step1_4_evaluate_patterns import evaluate_patterns as evaluate_patterns_func
from step2_0_preprocess_data import preprocess_data as preprocess_data_func
from step2_1_train_model import train_model as train_model_func
from step2_2_generate_samples import generate_samples as generate_samples_func
from step2_3_visualize_results import visualize_results as visualize_results_func
from step2_4_evaluate_generation import main as evaluate_generation_main


def main():
    """主函数入口"""
    if len(sys.argv) < 2:
        logger.error("用法: python scripts/optimized_e2e_pipeline.py <input_raw_file_or_dir>")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    # 检查输入路径是否存在
    if not input_path.exists():
        logger.error(f"错误: 输入路径 {input_path} 不存在")
        sys.exit(1)

    # 设置输出目录结构
    base_output_dir, processed_dir, features_dir, patterns_dir, preprocess_dir, train_dir, generation_dir, visualization_dir = setup_directories()
    evaluation_dir = base_output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("优化后的端到端网络行为模式发现与生成管道")
    logger.info(f"输入路径: {input_path}")
    logger.info(f"输出目录: {base_output_dir}")
    logger.info("=" * 70)

    # 步骤1：处理原始数据
    logger.info("\n步骤1：处理原始数据")
    merged_processed_df = process_input_data(input_path)
    
    # 保存合并后的处理数据，用于后续步骤
    merged_processed_file = processed_dir / "merged_processed_data.csv"
    merged_processed_df.to_csv(merged_processed_file, index=False)

    # 步骤2：提取特征
    logger.info("\n步骤2：提取特征")
    features_df, valid_loss_values = extract_and_save_features(merged_processed_df, features_dir)

    # 步骤3：发现行为模式
    logger.info("\n步骤3：发现行为模式")
    discover_and_save_patterns(features_df, merged_processed_df, patterns_dir)

    # 步骤4：评估行为发现效果
    logger.info("\n步骤4：评估行为发现效果")
    # 查找合并后的特征文件
    merged_features_file = features_dir / "merged_features.csv"
    if not merged_features_file.exists():
        # 如果没有合并后的特征文件，使用第一个特征文件
        merged_features_file = next(features_dir.glob("*.csv"))
    # 创建评估结果目录
    pattern_eval_dir = base_output_dir / "pattern_evaluation"
    pattern_eval_dir.mkdir(parents=True, exist_ok=True)
    evaluate_patterns_func(merged_features_file, patterns_dir, pattern_eval_dir)
    logger.info("步骤4：评估行为发现效果完成")

    # 步骤5：预处理数据
    logger.info("\n步骤5：预处理数据")
    preprocess_data_func(patterns_dir, merged_processed_file, preprocess_dir)
    logger.info("步骤5：预处理数据完成")

    # 步骤6：训练模型
    logger.info("\n步骤6：训练条件扩散模型")
    train_model_func(preprocess_dir, train_dir)
    logger.info("步骤6：训练条件扩散模型完成")

    # 步骤7：生成样本
    logger.info("\n步骤7：生成网络状态样本")
    # 查找模型文件
    model_path = train_dir / "diffusion_model_final.pth"
    if not model_path.exists():
        # 尝试查找最佳模型
        model_path = train_dir / "diffusion_model_best.pth"
        if not model_path.exists():
            logger.error(f"模型文件不存在")
            sys.exit(1)
    generate_samples_func(merged_processed_file, patterns_dir, model_path, generation_dir)
    logger.info("步骤7：生成网络状态样本完成")

    # 步骤8：可视化结果
    logger.info("\n步骤8：可视化结果对比")
    # 查找生成的样本文件
    original_files = list(generation_dir.glob("original_sample_*.csv"))
    generated_files = list(generation_dir.glob("generated_sample_*.csv"))
    if original_files and generated_files:
        # 排序文件，确保一一对应
        original_files.sort()
        generated_files.sort()
        for original_file, generated_file in zip(original_files, generated_files):
            # 为每组样本创建独立的输出目录
            group_info = original_file.stem.split("_")[3:]
            group_dir_name = "_" + "_".join(group_info) if group_info else ""
            group_output_dir = visualization_dir / f"group{group_dir_name}"
            group_output_dir.mkdir(parents=True, exist_ok=True)
            visualize_results_func(original_file, generated_file, group_output_dir)
    logger.info("步骤8：可视化结果对比完成")

    # 步骤9：评估生成样本质量
    logger.info("\n步骤9：评估生成样本质量")
    # 使用evaluate_generation.py的main函数，它会处理所有样本
    # 保存原始sys.argv，然后修改它以调用evaluate_generation_main
    original_argv = sys.argv.copy()
    sys.argv = ["step2_4_evaluate_generation.py", str(generation_dir), str(evaluation_dir)]
    evaluate_generation_main()
    # 恢复原始sys.argv
    sys.argv = original_argv
    logger.info("步骤9：评估生成样本质量完成")

    logger.info("\n" + "=" * 70)
    logger.info("优化后的端到端管道执行完成!")
    logger.info(f"所有结果保存在: {base_output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
