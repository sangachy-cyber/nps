#!/usr/bin/env python3
"""
步骤1.4：评估行为发现效果

该脚本用于评估网络行为发现的效果，包括聚类质量、转移质量和行为分离等方面，并生成HTML格式的评估报告。
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
from network_simulation.evaluation.evaluator import Evaluator
from network_simulation.utils.logger import get_logger
from config import (
    FEATURES_DIR,
    PATTERNS_DIR,
    EVALUATION_DIR,
    CLEANUP_OLD_FILES,
    KEEP_LATEST_FILES
)


# 获取日志记录器
logger = get_logger(__name__)


def load_behavior_data(patterns_dir: Path):
    """加载行为发现结果

    Args:
        patterns_dir: 行为模式结果目录路径

    Returns:
        tuple: (行为标签数据, 转移矩阵数据)
    """
    # 加载行为标签
    labels_file = patterns_dir / "behavior_labels_hdbscan.json"
    if not labels_file.exists():
        logger.error(f"未找到行为标签文件: {labels_file}")
        sys.exit(1)
    
    with open(labels_file, "r") as f:
        labels_data = json.load(f)
    
    # 加载转移矩阵
    transition_file = patterns_dir / "behavior_transition_graph_hdbscan.json"
    if not transition_file.exists():
        logger.error(f"未找到转移矩阵文件: {transition_file}")
        sys.exit(1)
    
    with open(transition_file, "r") as f:
        transition_data = json.load(f)
    
    return labels_data, transition_data


def load_feature_data(features_file: Path):
    """加载特征数据

    Args:
        features_file: 特征数据文件路径

    Returns:
        pd.DataFrame: 特征数据DataFrame
    """
    if not features_file.exists():
        logger.error(f"未找到特征数据文件: {features_file}")
        sys.exit(1)
    
    return pd.read_csv(features_file)


def evaluate_patterns(input_features_file: Path, input_patterns_dir: Path, output_eval_dir: Path):
    """评估行为发现效果

    该函数使用现有的Evaluator类评估行为发现的效果，包括聚类质量、转移质量和行为分离等方面，并生成HTML格式的评估报告。

    Args:
        input_features_file: 特征数据文件路径
        input_patterns_dir: 行为模式结果目录路径
        output_eval_dir: 评估结果输出目录路径

    Returns:
        Path: 评估结果输出目录路径
    """
    logger.info(f"开始评估行为发现效果")
    logger.info(f"输入特征文件: {input_features_file}")
    logger.info(f"输入模式结果目录: {input_patterns_dir}")
    logger.info(f"输出评估结果目录: {output_eval_dir}")
    
    # 加载特征数据
    features_df = load_feature_data(input_features_file)
    logger.info(f"特征数据加载完成，共 {len(features_df)} 行")
    
    # 加载行为发现结果
    labels_data, transition_data = load_behavior_data(input_patterns_dir)
    logger.info(f"行为发现结果加载完成")
    
    # 初始化评估器
    evaluator = Evaluator()
    
    # 准备评估数据
    # 只使用数值类型的特征，排除非数值列（如时间戳）
    X = features_df.select_dtypes(include=[np.number]).values
    labels = np.array(labels_data["labels"])
    transition_matrix = np.array(transition_data["transition_matrix"])
    
    logger.info(f"特征数据形状: {X.shape}")
    logger.info(f"行为标签形状: {labels.shape}")
    logger.info(f"转移矩阵形状: {transition_matrix.shape}")
    
    # 评估聚类质量
    logger.info("评估聚类质量")
    clustering_quality = evaluator.evaluate_clustering_quality(X, labels)
    logger.debug(f"聚类质量评估结果: {clustering_quality}")
    
    # 评估转移质量
    logger.info("评估转移质量")
    transition_quality = evaluator.evaluate_transition_quality(transition_matrix)
    logger.debug(f"转移质量评估结果: {transition_quality}")
    
    # 评估行为分离
    logger.info("评估行为分离")
    behavior_separation = evaluator.evaluate_behavior_separation(X, labels)
    logger.debug(f"行为分离评估结果: {behavior_separation}")
    
    # 整合评估结果
    evaluation_results = {
        "clustering_quality": clustering_quality,
        "transition_quality": transition_quality,
        "behavior_separation": behavior_separation,
        "transition_matrix": transition_matrix.tolist()
    }
    
    # 生成可视化图表
    logger.info("生成可视化图表")
    evaluator.generate_visualizations(X, labels, transition_matrix, output_eval_dir)
    
    # 保存评估结果
    logger.info("保存评估结果")
    evaluator.save(evaluation_results, output_eval_dir)
    
    logger.info(f"行为发现评估完成，结果已保存到: {output_eval_dir}")
    return output_eval_dir



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

    解析命令行参数，调用evaluate_patterns函数评估行为发现效果。
    """
    import argparse
    
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="评估网络行为发现效果")
    parser.add_argument(
        "input_features",
        nargs="?",
        type=Path,
        default=FEATURES_DIR,
        help="特征数据文件或目录路径 (默认: data/results/features)",
    )
    parser.add_argument(
        "input_patterns",
        nargs="?",
        type=Path,
        default=PATTERNS_DIR,
        help="行为模式结果目录路径 (默认: data/results/patterns)",
    )
    parser.add_argument(
        "output_evaluation",
        nargs="?",
        type=Path,
        default=EVALUATION_DIR,
        help="评估结果输出目录路径 (默认: data/results/evaluation)",
    )

    args = parser.parse_args()

    input_features_path = args.input_features
    input_patterns_dir = args.input_patterns
    output_eval_dir = args.output_evaluation

    # 确保输出目录存在
    output_eval_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧文件
    cleanup_old_files(output_eval_dir, "evaluation_results.json")
    cleanup_old_files(output_eval_dir, "evaluation_summary.txt")
    cleanup_old_files(output_eval_dir, "comprehensive_evaluation_report.*")

    # 评估行为模式
    if input_features_path.is_file():
        # 如果输入是文件，直接处理
        evaluate_patterns(
            input_features_path, input_patterns_dir, output_eval_dir
        )
    elif input_features_path.is_dir():
        # 如果输入是目录，使用合并后的特征文件
        merged_features_file = input_features_path / "merged_features.csv"
        if not merged_features_file.exists():
            logger.error(f"在 {input_features_path} 中未找到 merged_features.csv")
            logger.error("请先运行特征提取脚本生成合并特征文件")
            sys.exit(1)
        
        evaluate_patterns(
            merged_features_file, input_patterns_dir, output_eval_dir
        )
    else:
        logger.error(f"输入特征路径既不是文件也不是目录: {input_features_path}")
        sys.exit(1)

    logger.info("步骤1.4：评估行为发现效果完成！")



if __name__ == "__main__":
    main()