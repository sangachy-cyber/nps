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
    return PatternIdentifier(method="hdbscan")


@pytest.fixture
def pattern_identifier_gmm():
    """初始化GMM模式识别器"""
    return PatternIdentifier(method="gmm")


@pytest.fixture
def pattern_identifier_kmeans():
    """初始化KMeans模式识别器"""
    return PatternIdentifier(method="kmeans")


def test_pattern_identifier_hdbscan(
    pattern_identifier, sample_features, sample_raw_data
):
    """测试HDBSCAN模式识别"""
    results = pattern_identifier.identify(sample_features, sample_raw_data)

    # 验证结果包含必要的字段
    assert "labels" in results
    assert "metrics" in results
    assert "transition_matrix" in results
    assert "cluster_stats" in results
    assert "separation_metrics" in results

    # 验证标签数量与输入特征数量一致
    assert len(results["labels"]) == len(sample_features)

    # 验证标签类型
    assert isinstance(results["labels"], list)


def test_pattern_identifier_gmm(
    pattern_identifier_gmm, sample_features, sample_raw_data
):
    """测试GMM模式识别"""
    results = pattern_identifier_gmm.identify(sample_features, sample_raw_data)

    # 验证结果包含必要的字段
    assert "labels" in results
    assert "metrics" in results
    assert "transition_matrix" in results
    assert "cluster_stats" in results
    assert "separation_metrics" in results

    # 验证标签数量与输入特征数量一致
    assert len(results["labels"]) == len(sample_features)


def test_pattern_identifier_kmeans(
    pattern_identifier_kmeans, sample_features, sample_raw_data
):
    """测试KMeans模式识别"""
    results = pattern_identifier_kmeans.identify(sample_features, sample_raw_data)

    # 验证结果包含必要的字段
    assert "labels" in results
    assert "metrics" in results
    assert "transition_matrix" in results
    assert "cluster_stats" in results
    assert "separation_metrics" in results

    # 验证标签数量与输入特征数量一致
    assert len(results["labels"]) == len(sample_features)


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
    assert (tmp_path / "behavior_labels_hdbscan.json").exists()
    assert (tmp_path / "behavior_hdbscan_model.pkl").exists()
    assert (tmp_path / "behavior_transition_graph_hdbscan.json").exists()
    assert (tmp_path / "metadata" / "valid_loss_values.json").exists()
    assert (tmp_path / "clustered_data").exists()


def test_pattern_identifier_metrics(
    pattern_identifier, sample_features, sample_raw_data
):
    """测试模式识别指标计算"""
    results = pattern_identifier.identify(sample_features, sample_raw_data)

    # 验证指标计算
    metrics = results["metrics"]
    assert "silhouette_score" in metrics
    assert "calinski_harabasz_score" in metrics
    assert "num_clusters" in metrics
    assert "cluster_sizes" in metrics
    assert "average_transition_entropy" in metrics
    assert "transition_sparsity" in metrics


def test_pattern_identifier_transition_matrix(
    pattern_identifier, sample_features, sample_raw_data
):
    """测试转移矩阵计算"""
    results = pattern_identifier.identify(sample_features, sample_raw_data)

    transition_matrix = results["transition_matrix"]

    # 验证转移矩阵是二维列表
    assert isinstance(transition_matrix, list)
    assert all(isinstance(row, list) for row in transition_matrix)

    # 验证转移矩阵是方阵
    assert len(transition_matrix) == len(transition_matrix[0])
