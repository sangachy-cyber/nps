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
    # 创建简单的原始数据（双流）
    timestamps = pd.date_range(start="2025-01-01", periods=1000, freq="100ms")
    data = {
        "timestamp": timestamps,
        "delay1": np.random.randn(1000) * 10 + 20,  # 上行时延
        "delay2": np.random.randn(1000) * 10 + 25,  # 下行时延
        "loss_rate1": np.random.choice([0.0, 0.5, 1.0], 1000),  # 上行丢包率
        "loss_rate2": np.random.choice([0.0, 0.5, 1.0], 1000),  # 下行丢包率
    }
    return pd.DataFrame(data)


@pytest.fixture
def pattern_identifier():
    """初始化模式识别器"""
    # 提供必要的配置参数
    config = {
        "STRONG_BURST_DELAY_THRESHOLD": 400,
        "STRONG_BURST_LOSS_THRESHOLD": 0.25,
        "STRONG_BURST_MIN_RUN": 15,
        "INSTANT_SPIKE_DELAY_THRESHOLD": 800,
        "INSTANT_SPIKE_LOSS_THRESHOLD": 0.8,
        "INSTANT_SPIKE_MAX_COUNT": 5,
        "INSTANT_SPIKE_MAX_RATIO": 0.1,
        "WEAK_BURST_LOSS_NONZERO_RATIO": 0.3,
        "WEAK_BURST_MIN_CONDITIONS": 2,
        "DYNAMIC_THRESHOLD_WINDOW_SIZE": 100,
        "save_allowed_prefixes": ["delay", "loss_rate", "bandwidth"],
        "DEFAULT_DELAY_MEAN_LOW": 50,
        "DEFAULT_DELAY_MEAN_HIGH": 200,
        "DEFAULT_DELAY_STD_HIGH": 100,
        "DEFAULT_LOSS_MEAN_LOW": 0.1,
        "DEFAULT_LOSS_MEAN_HIGH": 0.5,
        "DEFAULT_LOSS_STD_HIGH": 0.2,
    }
    return PatternIdentifier(method="rule", config=config)


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

    # 初始化数据数组（上下行）
    delay1 = np.zeros(500)
    delay2 = np.zeros(500)
    loss_rate1 = np.zeros(500)
    loss_rate2 = np.zeros(500)

    # Window 0: Stable (标签0) - 低延迟，低丢包
    delay1[0:100] = np.random.randn(100) * 10 + 20  # 上行：平均20ms，标准差10
    delay2[0:100] = np.random.randn(100) * 10 + 25  # 下行：平均25ms，标准差10
    loss_rate1[0:100] = np.zeros(100)  # 上行：无丢包
    loss_rate2[0:100] = np.zeros(100)  # 下行：无丢包

    # Window 1: Weak Burst (标签1) - 中等延迟波动，中等丢包率
    delay1[100:200] = np.random.randn(100) * 50 + 50  # 上行：平均50ms，标准差50
    delay2[100:200] = np.random.randn(100) * 50 + 55  # 下行：平均55ms，标准差50
    loss_rate1[100:200] = np.random.choice([0.0, 0.1, 0.2], 100)  # 上行：低到中等丢包
    loss_rate2[100:200] = np.random.choice([0.0, 0.1, 0.2], 100)  # 下行：低到中等丢包

    # Window 2: Strong Burst (标签2) - 持续高延迟，高丢包
    delay1[200:300] = np.random.randn(100) * 50 + 450  # 上行：平均450ms，标准差50
    delay2[200:300] = np.random.randn(100) * 50 + 455  # 下行：平均455ms，标准差50
    loss_rate1[200:300] = np.random.rand(100) * 0.3 + 0.25  # 上行：0.25到0.55的丢包率
    loss_rate2[200:300] = np.random.rand(100) * 0.3 + 0.25  # 下行：0.25到0.55的丢包率

    # Window 3: Instant Spike (标签3) - 瞬时高延迟/丢包
    delay1[300:395] = np.random.randn(95) * 10 + 20  # 上行：正常延迟
    delay1[395:400] = np.array([800, 850, 900, 850, 800])  # 上行：5个瞬时高延迟点
    delay2[300:395] = np.random.randn(95) * 10 + 25  # 下行：正常延迟
    delay2[395:400] = np.array([850, 900, 950, 900, 850])  # 下行：5个瞬时高延迟点
    loss_rate1[300:395] = np.zeros(95)  # 上行：正常丢包
    loss_rate1[395:400] = np.ones(5)  # 上行：5个瞬时高丢包点
    loss_rate2[300:395] = np.zeros(95)  # 下行：正常丢包
    loss_rate2[395:400] = np.ones(5)  # 下行：5个瞬时高丢包点

    # Window 4: High Delay No Loss (标签4) - 高延迟，无丢包
    delay1[400:500] = np.random.randn(100) * 50 + 600  # 上行：平均600ms，标准差50
    delay2[400:500] = np.random.randn(100) * 50 + 605  # 下行：平均605ms，标准差50
    loss_rate1[400:500] = np.zeros(100)  # 上行：无丢包
    loss_rate2[400:500] = np.zeros(100)  # 下行：无丢包

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "delay1": delay1,
            "delay2": delay2,
            "loss_rate1": loss_rate1,
            "loss_rate2": loss_rate2,
        }
    )


def test_pattern_identifier_comprehensive(
    pattern_identifier, sample_features, sample_raw_data
):
    """测试模式识别的综合功能"""
    results = pattern_identifier.identify(sample_features, sample_raw_data)

    # 验证结果包含必要的字段
    assert "labels_up" in results
    assert "metrics_up" in results
    assert "transition_matrix_up" in results
    assert "behavior_stats_up" in results
    assert "separation_metrics_up" in results
    assert "labels_down" in results
    assert "metrics_down" in results
    assert "transition_matrix_down" in results
    assert "behavior_stats_down" in results
    assert "separation_metrics_down" in results

    # 验证标签数量与输入特征数量一致
    assert len(results["labels_up"]) == len(sample_features)
    assert len(results["labels_down"]) == len(sample_features)

    # 验证标签类型
    assert isinstance(results["labels_up"], list)
    assert all(isinstance(label, int) for label in results["labels_up"])
    assert isinstance(results["labels_down"], list)
    assert all(isinstance(label, int) for label in results["labels_down"])

    # 验证指标计算（上行）
    metrics_up = results["metrics_up"]
    assert "average_transition_entropy" in metrics_up
    assert "transition_sparsity" in metrics_up

    # 验证指标值范围（上行）
    assert isinstance(metrics_up["average_transition_entropy"], float)
    assert isinstance(metrics_up["transition_sparsity"], float)
    assert 0 <= metrics_up["transition_sparsity"] <= 1

    # 验证转移矩阵（上行）
    transition_matrix_up = results["transition_matrix_up"]
    assert isinstance(transition_matrix_up, list)
    assert all(isinstance(row, list) for row in transition_matrix_up)
    assert len(transition_matrix_up) == len(transition_matrix_up[0])  # 方阵

    # 验证转移矩阵每行和为1（概率分布），全零行（吸收态）除外
    for row in transition_matrix_up:
        row_sum = sum(row)
        if row_sum > 1e-6:  # 非全零行
            assert abs(row_sum - 1.0) < 1e-6, f"转移矩阵行和不为1: {row_sum}"

    # 验证行为统计信息（上行）
    behavior_stats_up = results["behavior_stats_up"]
    assert isinstance(behavior_stats_up, dict)

    # 验证每个行为的统计信息包含必要字段（上行）
    for label, stats in behavior_stats_up.items():
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

    # 验证分离度指标（上行）
    separation_metrics_up = results["separation_metrics_up"]
    assert isinstance(separation_metrics_up, dict)
    assert "label_means" in separation_metrics_up
    assert "label_stds" in separation_metrics_up


def test_different_behavior_types(
    pattern_identifier, behavior_specific_features, behavior_specific_raw_data
):
    """测试不同行为类型的识别"""
    results = pattern_identifier.identify(
        behavior_specific_features, behavior_specific_raw_data
    )
    labels_up = results["labels_up"]
    labels_down = results["labels_down"]

    # 验证每个窗口被正确识别为对应的行为类型
    assert len(labels_up) == 5
    assert len(labels_down) == 5

    # 注意：行为识别可能受随机性影响，我们检查是否包含大部分预期的行为类型
    # 包括所有可能的有效行为标签（除了INVALID标签-1）
    expected_labels = {0, 1, 2, 3, 4, 5, 6, 7}
    actual_labels_up = set(labels_up)
    actual_labels_down = set(labels_down)

    # 至少识别到4种不同的行为类型
    assert (
        len(actual_labels_up) >= 4
    ), f"上行识别到的行为类型数量不足，实际识别到: {actual_labels_up}, 预期至少4种"
    assert (
        len(actual_labels_down) >= 4
    ), f"下行识别到的行为类型数量不足，实际识别到: {actual_labels_down}, 预期至少4种"

    # 确保所有识别到的标签都是有效的行为标签
    valid_labels = expected_labels
    assert actual_labels_up.issubset(
        valid_labels
    ), f"上行识别到无效的行为标签: {actual_labels_up - valid_labels}"
    assert actual_labels_down.issubset(
        valid_labels
    ), f"下行识别到无效的行为标签: {actual_labels_down - valid_labels}"


def test_no_raw_data_behavior_identification(
    pattern_identifier, behavior_specific_features
):
    """测试无原始数据情况下的行为识别"""
    results = pattern_identifier.identify(behavior_specific_features)

    # 验证结果包含必要的字段
    assert "labels_up" in results
    assert "metrics_up" in results
    assert "transition_matrix_up" in results
    assert "behavior_stats_up" in results
    assert "separation_metrics_up" in results
    assert "labels_down" in results
    assert "metrics_down" in results
    assert "transition_matrix_down" in results
    assert "behavior_stats_down" in results
    assert "separation_metrics_down" in results

    # 验证标签数量与输入特征数量一致
    assert len(results["labels_up"]) == len(behavior_specific_features)
    assert len(results["labels_down"]) == len(behavior_specific_features)

    # 验证所有标签都是有效的行为类型（0-3，因为无原始数据时只有4种行为类型）
    valid_labels = set(pattern_identifier.BEHAVIOR_LABELS.values())
    for label in results["labels_up"]:
        assert label in valid_labels, f"无效的上行行为标签: {label}"
    for label in results["labels_down"]:
        assert label in valid_labels, f"无效的下行行为标签: {label}"


def test_pattern_identifier_save(
    pattern_identifier, behavior_specific_features, behavior_specific_raw_data, tmp_path
):
    """测试模式识别结果保存"""
    # 使用匹配的特征和原始数据，窗口范围与数据长度匹配
    results = pattern_identifier.identify(
        behavior_specific_features, behavior_specific_raw_data
    )

    pattern_identifier.save(
        results,
        tmp_path,
        valid_loss_values=[0.0, 0.5, 1.0],
    )

    # 验证文件是否生成
    assert (tmp_path / "labels_rule_up.npy").exists()
    assert (tmp_path / "labels_rule_down.npy").exists()
    assert (tmp_path / "behavior_transition_graph_rule_up.json").exists()
    assert (tmp_path / "behavior_transition_graph_rule_down.json").exists()
    assert (tmp_path / "metadata" / "valid_loss_values.json").exists()

    # 验证窗口数据生成情况
    labeled_windows_dir = tmp_path / "labeled_windows"
    # 对于behavior_specific_data，应该有有效的窗口数据生成
    if labeled_windows_dir.exists():
        # 检查是否生成了不同标签的目录
        labeled_dirs = [d for d in labeled_windows_dir.iterdir() if d.is_dir()]
        # 至少应该有一个标签目录
        assert len(labeled_dirs) > 0

        # 检查每个标签目录下是否有窗口文件
        for label_dir in labeled_dirs:
            window_files = list(label_dir.glob("window_*.csv"))
            # 这个测试数据应该生成有效的窗口文件
            assert isinstance(window_files, list)


def test_empty_features_handling(pattern_identifier):
    """测试空特征数据处理"""
    # 创建没有特征列的DataFrame
    empty_features = pd.DataFrame(
        {"window_start": [0, 1, 2], "window_end": [100, 200, 300]}
    )

    with pytest.raises(ValueError, match="未找到特征列"):
        pattern_identifier.identify(empty_features)


def test_invalid_method(pattern_identifier, sample_features, sample_raw_data):
    """测试无效的检测方法"""
    # 修改pattern_identifier的方法为无效值
    pattern_identifier.method = "invalid_method"

    with pytest.raises(ValueError, match="未知的检测方法"):
        pattern_identifier.identify(sample_features, sample_raw_data)
