#!/usr/bin/env python3
"""
测试特征方向过滤逻辑
"""

import numpy as np
from pathlib import Path
from src.network_simulation.visualization.feature_space_visualizer import FeatureSpaceVisualizer

# 创建测试数据
def create_test_data():
    # 生成500个样本，每个样本有10个特征
    # 特征0-4与上行相关（以1结尾），特征5-9与下行相关（以2结尾）
    np.random.seed(42)
    X = np.random.randn(500, 10)

    # 生成标签，前250个样本为类别0，后250个样本为类别1
    labels = np.zeros(500, dtype=int)
    labels[250:] = 1

    # 特征名称，包含上下行信息
    feature_names = [
        "feat_delay1_std",
        "feat_delay1_mean",
        "feat_loss_rate1_std",
        "feat_loss_rate1_mean",
        "feat_ratio1",
        "feat_delay2_std",
        "feat_delay2_mean",
        "feat_loss_rate2_std",
        "feat_loss_rate2_mean",
        "feat_ratio2"
    ]

    return X, labels, feature_names

# 测试特征方向过滤逻辑
def test_feature_direction_filtering():
    """测试特征方向过滤逻辑"""
    # 创建测试数据
    X, labels, feature_names = create_test_data()

    # 创建输出目录
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    # 创建可视化器
    visualizer = FeatureSpaceVisualizer(output_dir)

    # 测试上行特征过滤
    print("测试上行特征过滤...")
    visualizer.generate_behavior_separation_plots(
        X, labels, output_dir, feature_names, direction="up"
    )
    print("上行特征过滤测试完成")

    # 测试下行特征过滤
    print("测试下行特征过滤...")
    visualizer.generate_behavior_separation_plots(
        X, labels, output_dir, feature_names, direction="down"
    )
    print("下行特征过滤测试完成")

    # 检查生成的文件
    up_files = list(output_dir.glob("up_*"))
    down_files = list(output_dir.glob("down_*"))

    print(f"生成的上行文件: {[f.name for f in up_files]}")
    print(f"生成的下行文件: {[f.name for f in down_files]}")

    # 验证生成了正确的文件
    assert len(up_files) > 0, "没有生成上行文件"
    assert len(down_files) > 0, "没有生成下行文件"

    print("特征方向过滤逻辑测试成功！")

if __name__ == "__main__":
    test_feature_direction_filtering()
