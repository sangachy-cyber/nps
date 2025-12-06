#!/usr/bin/env python3
"""
特征提取模块测试用例
"""

import pytest
import numpy as np
import pandas as pd
from src.network_simulation.feature_extraction.feature_extractor import (
    FeatureExtractor,
)


@pytest.fixture
def sample_data():
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
    features = feature_extractor.extract(sample_data)

    # 验证特征提取结果
    assert isinstance(features, pd.DataFrame)
    assert len(features) > 0

    # 验证必要的特征列存在
    required_features = [
        "feat_delay_std",
        "feat_loss_nonzero_ratio",
        "feat_loss_high_ratio",
        "feat_max_consec_loss",
        "feat_loss_unique_values",
        "feat_delay_trend",
        "feat_delay_acf_5",
        "feat_loss_mode_encoded",
        "feat_loss_pattern_std",
    ]

    for feat in required_features:
        assert feat in features.columns


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
