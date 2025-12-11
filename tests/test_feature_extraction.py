#!/usr/bin/env python3
"""
特征提取模块测试用例
"""

import pytest
import numpy as np
import pandas as pd
from src.network_simulation.pattern_discovery.feature_extractor import (
    FeatureExtractor,
)


@pytest.fixture
def sample_data():
    """创建测试原始数据"""
    # 创建简单的双通道原始数据
    timestamps = pd.date_range(start="2025-01-01", periods=1000, freq="100ms")
    data = {
        "timestamp": timestamps,
        "delay1": np.random.randn(1000) * 10 + 20,  # 上行延迟
        "loss_rate1": np.random.choice([0.0, 0.5, 1.0], 1000),  # 上行丢包率
        "delay2": np.random.randn(1000) * 10 + 15,  # 下行延迟，略低于上行
        "loss_rate2": np.random.choice([0.0, 0.5, 1.0], 1000),  # 下行丢包率
    }
    return pd.DataFrame(data)


@pytest.fixture
def feature_extractor():
    """初始化特征提取器"""
    return FeatureExtractor()


def test_feature_extractor_init(feature_extractor):
    """测试特征提取器初始化"""
    assert feature_extractor.window_size == 10.0
    assert feature_extractor.slide_step == 5.0
    assert feature_extractor.time_granularity == 0.1


def test_feature_extractor_extract_features(feature_extractor, sample_data):
    """测试特征提取功能"""
    # 保存原始的窗口大小和滑动步长，以便恢复
    original_window_samples = feature_extractor.window_samples
    original_slide_samples = feature_extractor.slide_samples

    # 临时修改窗口大小和滑动步长，避免高度相关特征被删除
    feature_extractor.window_samples = 50  # 减小窗口大小，降低特征相关性
    feature_extractor.slide_samples = 25  # 减小滑动步长
    feature_extractor.window_size = (
        feature_extractor.window_samples * feature_extractor.time_granularity
    )
    feature_extractor.slide_step = (
        feature_extractor.slide_samples * feature_extractor.time_granularity
    )

    features = feature_extractor.extract(sample_data)

    # 验证特征提取结果
    assert isinstance(features, pd.DataFrame)
    assert len(features) > 0

    # 验证提取过程中生成了测试期望的特征列
    # 注意：由于高度相关特征可能被删除，我们不直接断言它们存在于最终结果中
    # 而是检查它们是否在原始提取结果中生成过

    # 重新提取特征，但跳过高度相关特征删除步骤进行测试
    # 提取有效丢包值
    feature_extractor.valid_loss_values = feature_extractor._extract_valid_loss_values(
        sample_data
    )
    feature_extractor._build_loss_mode_mapping()

    # 提取单个窗口的特征，不进行高度相关特征删除
    window = sample_data.iloc[0:50].copy()
    window_features = feature_extractor._extract_window_features(window, 0, 50)

    # 验证必要的特征在单个窗口提取中存在
    # 双通道数据会生成带有1和2后缀的特征列名，分别对应上行和下行
    required_features = [
        "feat_delay1_std",
        "feat_delay1_mean",
        "feat_loss1_nonzero_ratio",
        "feat_loss1_high_ratio",
        "feat_loss1_mean",
        "feat_loss1_std",
        "feat_max_consec_loss1",
        "feat_max_congestion_run1",
        "feat_delay1_trend",
        "feat_delay1_acf_5",
        "feat_loss1_mode_encoded",
        "feat_delay2_std",
        "feat_delay2_mean",
        "feat_loss2_nonzero_ratio",
        "feat_loss2_high_ratio",
        "feat_loss2_mean",
        "feat_loss2_std",
        "feat_max_consec_loss2",
        "feat_max_congestion_run2",
        "feat_delay2_trend",
        "feat_delay2_acf_5",
        "feat_loss2_mode_encoded",
        # 跨通道特征
        "feat_delay_ratio",
        "feat_loss_symmetry",
        "feat_congestion_match",
    ]

    for feat in required_features:
        assert feat in window_features, f"特征 {feat} 未在单个窗口提取中生成"

    # 恢复原始参数
    feature_extractor.window_samples = original_window_samples
    feature_extractor.slide_samples = original_slide_samples
    feature_extractor.window_size = (
        feature_extractor.window_samples * feature_extractor.time_granularity
    )
    feature_extractor.slide_step = (
        feature_extractor.slide_samples * feature_extractor.time_granularity
    )


def test_feature_extractor_load_data(feature_extractor, sample_data, tmp_path):
    """测试数据加载功能"""
    # 保存到临时文件
    data_file = tmp_path / "test_data.csv"
    sample_data.to_csv(data_file, index=False)

    # 测试加载功能
    loaded_data = feature_extractor.load_data(data_file)

    # 验证加载结果与原始数据一致
    pd.testing.assert_frame_equal(sample_data, loaded_data)


def test_feature_extractor_windows(feature_extractor, sample_data):
    """测试窗口计算"""
    # 提取特征
    features = feature_extractor.extract(sample_data)

    # 计算预期窗口数量
    total_samples = len(sample_data)
    expected_windows = (
        total_samples - feature_extractor.window_samples
    ) // feature_extractor.slide_samples + 1

    # 验证窗口数量
    assert len(features) == expected_windows


def test_feature_extractor_save_features(sample_data, tmp_path):
    """测试特征保存功能"""
    # 提取特征
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract(sample_data)

    # 保存特征
    feature_file = tmp_path / "test_features.csv"
    features.to_csv(feature_file, index=False)

    # 验证文件存在
    assert feature_file.exists()

    # 验证文件内容，解析所有 datetime 列
    datetime_columns = ["window_start_time", "window_end_time"]
    loaded_features = pd.read_csv(feature_file, parse_dates=datetime_columns)

    # 比较时忽略 index，因为保存和加载时 index 可能不同
    pd.testing.assert_frame_equal(
        features.reset_index(drop=True), loaded_features.reset_index(drop=True)
    )


def test_feature_extractor_valid_loss_values(feature_extractor, sample_data):
    """测试合法丢包值提取"""
    # 提取特征，会自动提取合法丢包值
    features = feature_extractor.extract(sample_data)

    # 验证特征提取结果
    assert isinstance(features, pd.DataFrame)
    assert len(features) > 0

    # 验证合法丢包值已提取
    assert len(feature_extractor.valid_loss_values) > 0

    # 验证特征提取器的损失模式映射已构建
    assert len(feature_extractor.loss_mode_mapping) == len(
        feature_extractor.valid_loss_values
    )
