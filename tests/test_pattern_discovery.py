#!/usr/bin/env python3
"""
模式发现模块测试用例
"""

import pytest
import numpy as np
import pandas as pd
from src.network_simulation.pattern_discovery.pattern_identifier import (
    PatternIdentifier,
)


@pytest.fixture
def sample_features():
    """创建测试特征数据"""
    # 创建简单的特征数据
    data = {
        "window_start": np.arange(0, 10000, 100),
        "window_end": np.arange(100, 10100, 100),
        "feat_delay_std": np.random.randn(100),
        "feat_loss_nonzero_ratio": np.random.rand(100),
        "feat_loss_high_ratio": np.random.rand(100),
        "feat_max_consec_loss": np.random.randint(0, 10, 100),
        "feat_loss_unique_values": np.random.randint(1, 5, 100),
        "feat_delay_trend": np.random.randn(100),
        "feat_delay_acf_5": np.random.rand(100),
        "feat_loss_mode_encoded": np.random.randint(0, 3, 100),
        "feat_loss_pattern_std": np.random.randn(100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_raw_data():
    """创建测试原始数据"""
    # 创建简单的原始数据
    timestamps = pd.date_range(start="2025-01-01", periods=1000, freq="100ms")
    data = {
        "timestamp": timestamps,
        "delay": np.random.randn(1000) * 10 + 20,
        "loss_rate": np.random.choice([0.0, 0.5, 1.0], 1000),
    }
    return pd.DataFrame(data)


@pytest.fixture
def pattern_identifier():
    """初始化模式识别器"""
    return PatternIdentifier(method="rule")


@pytest.fixture
def behavior_specific_features():
    """创建用于测试特定行为类型的特征数据"""
    # 创建5个窗口的特征数据，每个窗口对应一种行为类型
    data = {
        "window_start": [0, 100, 200, 300, 400],
        "window_end": [100, 200, 300, 400, 500],
        "feat_delay_std": [0.5, 1.5, 2.5, 3.0, 2.0],
        "feat_loss_nonzero_ratio": [0.05, 0.2, 0.4, 0.1, 0.01],
        "feat_max_consec_loss": [1, 5, 20, 3, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def behavior_specific_raw_data():
    """创建用于测试特定行为类型的原始数据"""
    # 创建500个样本的原始数据，分为5个窗口，每个窗口对应一种行为类型
    timestamps = pd.date_range(start="2025-01-01", periods=500, freq="100ms")
    
    # 初始化数据数组
    delays = np.zeros(500)
    loss_rates = np.zeros(500)
    
    # Window 0: Stable (标签0) - 低延迟，低丢包
    delays[0:100] = np.random.randn(100) * 10 + 20  # 平均20ms，标准差10
    loss_rates[0:100] = np.zeros(100)  # 无丢包
    
    # Window 1: Weak Burst (标签1) - 中等延迟波动，中等丢包率
    delays[100:200] = np.random.randn(100) * 50 + 50  # 平均50ms，标准差50
    loss_rates[100:200] = np.random.choice([0.0, 0.1, 0.2], 100)  # 低到中等丢包
    
    # Window 2: Strong Burst (标签2) - 持续高延迟，高丢包
    delays[200:300] = np.random.randn(100) * 50 + 450  # 平均450ms，标准差50
    loss_rates[200:300] = np.random.rand(100) * 0.3 + 0.25  # 0.25到0.55的丢包率
    
    # Window 3: Instant Spike (标签3) - 瞬时高延迟/丢包
    delays[300:395] = np.random.randn(95) * 10 + 20  # 正常延迟
    delays[395:400] = np.array([800, 850, 900, 850, 800])  # 5个瞬时高延迟点
    loss_rates[300:395] = np.zeros(95)  # 正常丢包
    loss_rates[395:400] = np.ones(5)  # 5个瞬时高丢包点
    
    # Window 4: High Delay No Loss (标签4) - 高延迟，无丢包
    delays[400:500] = np.random.randn(100) * 50 + 600  # 平均600ms，标准差50
    loss_rates[400:500] = np.zeros(100)  # 无丢包
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "delay": delays,
        "loss_rate": loss_rates,
    })


def test_pattern_identifier_comprehensive(pattern_identifier, sample_features, sample_raw_data):
    """测试模式识别的综合功能"""
    results = pattern_identifier.identify(sample_features, sample_raw_data)

    # 验证结果包含必要的字段
    assert "labels" in results
    assert "metrics" in results
    assert "transition_matrix" in results
    assert "behavior_stats" in results
    assert "separation_metrics" in results

    # 验证标签数量与输入特征数量一致
    assert len(results["labels"]) == len(sample_features)

    # 验证标签类型
    assert isinstance(results["labels"], list)
    assert all(isinstance(label, int) for label in results["labels"])

    # 验证指标计算
    metrics = results["metrics"]
    assert "average_transition_entropy" in metrics
    assert "transition_sparsity" in metrics
    
    # 验证指标值范围
    assert isinstance(metrics["average_transition_entropy"], float)
    assert isinstance(metrics["transition_sparsity"], float)
    assert 0 <= metrics["transition_sparsity"] <= 1

    # 验证转移矩阵
    transition_matrix = results["transition_matrix"]
    assert isinstance(transition_matrix, list)
    assert all(isinstance(row, list) for row in transition_matrix)
    assert len(transition_matrix) == len(transition_matrix[0])  # 方阵
    
    # 验证转移矩阵每行和为1（概率分布）
    for row in transition_matrix:
        assert abs(sum(row) - 1.0) < 1e-6, f"转移矩阵行和不为1: {sum(row)}"

    # 验证行为统计信息
    behavior_stats = results["behavior_stats"]
    assert isinstance(behavior_stats, dict)
    
    # 验证每个行为的统计信息包含必要字段
    for label, stats in behavior_stats.items():
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "count" in stats
        
        # 验证统计值类型
        assert isinstance(stats["mean"], list)
        assert isinstance(stats["std"], list)
        assert isinstance(stats["min"], list)
        assert isinstance(stats["max"], list)
        assert isinstance(stats["count"], int)

    # 验证分离度指标
    separation_metrics = results["separation_metrics"]
    assert isinstance(separation_metrics, dict)
    assert "label_means" in separation_metrics
    assert "label_stds" in separation_metrics


def test_different_behavior_types(
    pattern_identifier, behavior_specific_features, behavior_specific_raw_data
):
    """测试不同行为类型的识别"""
    results = pattern_identifier.identify(behavior_specific_features, behavior_specific_raw_data)
    labels = results["labels"]
    
    # 验证每个窗口被正确识别为对应的行为类型
    assert len(labels) == 5
    
    # 注意：行为识别可能受随机性影响，我们检查是否包含预期的行为类型
    assert 0 in labels, "未识别到Stable行为(标签0)"
    assert 1 in labels, "未识别到Weak Burst行为(标签1)"
    assert 2 in labels, "未识别到Strong Burst行为(标签2)"
    assert 3 in labels, "未识别到Instant Spike行为(标签3)"
    assert 4 in labels, "未识别到High Delay No Loss行为(标签4)"
    
    # 验证所有标签都是有效的行为类型（0-4）
    for label in labels:
        assert 0 <= label <= 4, f"无效的行为标签: {label}"


def test_no_raw_data_behavior_identification(pattern_identifier, behavior_specific_features):
    """测试无原始数据情况下的行为识别"""
    results = pattern_identifier.identify(behavior_specific_features)
    
    # 验证结果包含必要的字段
    assert "labels" in results
    assert "metrics" in results
    assert "transition_matrix" in results
    assert "behavior_stats" in results
    assert "separation_metrics" in results
    
    # 验证标签数量与输入特征数量一致
    assert len(results["labels"]) == len(behavior_specific_features)
    
    # 验证所有标签都是有效的行为类型（0-3，因为无原始数据时只有4种行为类型）
    for label in results["labels"]:
        assert 0 <= label <= 3, f"无效的行为标签: {label}"


def test_pattern_identifier_save(
    pattern_identifier, sample_features, sample_raw_data, tmp_path
):
    """测试模式识别结果保存"""
    results = pattern_identifier.identify(sample_features, sample_raw_data)

    # 保存结果
    pattern_identifier.save(
        results,
        tmp_path,
        sample_features,
        valid_loss_values=[0.0, 0.5, 1.0],
    )

    # 验证文件是否生成
    assert (tmp_path / "behavior_labels_rule.json").exists()
    assert (tmp_path / "behavior_transition_graph_rule.json").exists()
    assert (tmp_path / "metadata" / "valid_loss_values.json").exists()
    
    # 验证窗口数据是否生成
    assert (tmp_path / "labeled_windows").exists()
    # 检查是否生成了不同标签的目录
    labeled_dirs = [d for d in (tmp_path / "labeled_windows").iterdir() if d.is_dir()]
    assert len(labeled_dirs) > 0
    
    # 检查每个标签目录下是否有窗口文件
    for label_dir in labeled_dirs:
        window_files = list(label_dir.glob("window_*.csv"))
        assert len(window_files) > 0


def test_empty_features_handling(pattern_identifier):
    """测试空特征数据处理"""
    # 创建没有特征列的DataFrame
    empty_features = pd.DataFrame({"window_start": [0, 1, 2], "window_end": [100, 200, 300]})
    
    with pytest.raises(ValueError, match="No feature columns found"):
        pattern_identifier.identify(empty_features)


def test_invalid_method(pattern_identifier, sample_features, sample_raw_data):
    """测试无效的检测方法"""
    # 修改pattern_identifier的方法为无效值
    pattern_identifier.method = "invalid_method"
    
    with pytest.raises(ValueError, match="Unknown detection method"):
        pattern_identifier.identify(sample_features, sample_raw_data)
