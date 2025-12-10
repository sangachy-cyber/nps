#!/usr/bin/env python3
"""
测试归一化前后的值，验证归一化是否正常工作
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

from network_simulation.condition_generation.normalization import Normalizer
from network_simulation.utils.logger import get_logger

# 获取日志记录器
logger = get_logger(__name__)

def test_normalization_values(input_file: Path):
    """
    测试归一化前后的值，验证归一化是否正常工作

    Args:
        input_file: 输入数据文件路径
    """
    logger.info(f"测试归一化前后的值，输入文件: {input_file}")

    # 加载原始数据
    df = pd.read_csv(input_file)
    logger.info(f"原始数据形状: {df.shape}")
    logger.info(f"原始数据列名: {df.columns.tolist()}")

    # 只使用前1000行数据进行测试
    df = df.head(1000)
    logger.info(f"使用前1000行数据进行测试，形状: {df.shape}")

    # 提取延迟和丢包率数据
    delay_values = df["delay"].values
    loss_rate_values = df["loss_rate"].values

    # 定义合法丢包值
    valid_loss_values = [0.0, 0.5, 1.0]

    # 初始化归一化器
    normalizer = Normalizer(valid_loss_values)

    # 打印原始数据统计
    logger.info("\n=== 原始数据统计 ===")
    logger.info(f"延迟 - 最小值: {np.min(delay_values):.4f}, 最大值: {np.max(delay_values):.4f}, 平均值: {np.mean(delay_values):.4f}, 标准差: {np.std(delay_values):.4f}")
    logger.info(f"丢包率 - 最小值: {np.min(loss_rate_values):.4f}, 最大值: {np.max(loss_rate_values):.4f}, 平均值: {np.mean(loss_rate_values):.4f}, 标准差: {np.std(loss_rate_values):.4f}")

    # 归一化数据
    logger.info("\n=== 开始归一化 ===")
    delay_norm, loss_norm = normalizer.normalize(delay_values, loss_rate_values)

    # 打印归一化后数据统计
    logger.info("\n=== 归一化后数据统计 ===")
    logger.info(f"归一化延迟 - 最小值: {np.min(delay_norm):.4f}, 最大值: {np.max(delay_norm):.4f}, 平均值: {np.mean(delay_norm):.4f}, 标准差: {np.std(delay_norm):.4f}")
    logger.info(f"归一化丢包率 - 最小值: {np.min(loss_norm):.4f}, 最大值: {np.max(loss_norm):.4f}, 平均值: {np.mean(loss_norm):.4f}, 标准差: {np.std(loss_norm):.4f}")

    # 反归一化数据
    logger.info("\n=== 开始反归一化 ===")
    delay_denorm, loss_denorm = normalizer.denormalize(delay_norm, loss_norm)

    # 打印反归一化后数据统计
    logger.info("\n=== 反归一化后数据统计 ===")
    logger.info(f"反归一化延迟 - 最小值: {np.min(delay_denorm):.4f}, 最大值: {np.max(delay_denorm):.4f}, 平均值: {np.mean(delay_denorm):.4f}, 标准差: {np.std(delay_denorm):.4f}")
    logger.info(f"反归一化丢包率 - 最小值: {np.min(loss_denorm):.4f}, 最大值: {np.max(loss_denorm):.4f}, 平均值: {np.mean(loss_denorm):.4f}, 标准差: {np.std(loss_denorm):.4f}")

    # 打印归一化前后的前10个值进行对比
    logger.info("\n=== 归一化前后值对比 (前10个) ===")
    for i in range(10):
        logger.info(f"样本 {i+1}: ")
        logger.info(f"  原始延迟: {delay_values[i]:.4f} -> 归一化: {delay_norm[i]:.4f} -> 反归一化: {delay_denorm[i]:.4f}")
        logger.info(f"  原始丢包率: {loss_rate_values[i]:.4f} -> 归一化: {loss_norm[i]:.4f} -> 反归一化: {loss_denorm[i]:.4f}")

    # 计算归一化和反归一化的误差
    delay_error = np.abs(delay_values - delay_denorm)
    loss_error = np.abs(loss_rate_values - loss_denorm)

    logger.info("\n=== 归一化误差统计 ===")
    logger.info(f"延迟误差 - 最小值: {np.min(delay_error):.6f}, 最大值: {np.max(delay_error):.6f}, 平均值: {np.mean(delay_error):.6f}, 标准差: {np.std(delay_error):.6f}")
    logger.info(f"丢包率误差 - 最小值: {np.min(loss_error):.6f}, 最大值: {np.max(loss_error):.6f}, 平均值: {np.mean(loss_error):.6f}, 标准差: {np.std(loss_error):.6f}")

    # 检查是否所有丢包率值都在合法范围内
    unique_loss_values = np.unique(loss_denorm)
    logger.info("\n=== 反归一化后丢包率唯一值 ===")
    logger.info(f"唯一值: {unique_loss_values}")
    logger.info(f"是否都在合法范围内: {all(loss in valid_loss_values for loss in unique_loss_values)}")

    logger.info("\n=== 测试完成 ===")

if __name__ == "__main__":
    import argparse

    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="测试归一化前后的值")
    parser.add_argument(
        "input_file",
        type=Path,
        default=Path("data/results/e2e_pipeline/processed/20251203_230356_b6x-playback_processed.csv"),
        help="输入数据文件路径 (默认: data/results/e2e_pipeline/processed/20251203_230356_b6x-playback_processed.csv)",
    )

    args = parser.parse_args()
    test_normalization_values(args.input_file)
