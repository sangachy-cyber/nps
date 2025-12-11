#!/usr/bin/env python3
"""
评估可视化器测试用例
"""

import pytest
import pandas as pd
import numpy as np
from src.network_simulation.visualization.visualizer import Visualizer


@pytest.fixture
def visualizer(tmp_path):
    """初始化可视化器"""
    return Visualizer(tmp_path)


@pytest.fixture
def sample_generated_data():
    """创建测试生成数据"""
    timestamps = pd.date_range(start="2025-01-01", periods=1000, freq="100ms")
    data = {
        "timestamp": timestamps,
        "delay1": np.random.randn(1000) * 10 + 20,  # 上行：均值20，标准差10
        "delay2": np.random.randn(1000) * 10 + 25,  # 下行：均值25，标准差10
        "loss_rate1": np.random.choice([0.0, 0.5, 1.0], 1000),  # 上行丢包率
        "loss_rate2": np.random.choice([0.0, 0.5, 1.0], 1000),  # 下行丢包率
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_features():
    """创建测试特征数据"""
    n_samples = 100
    n_features = 6
    return np.random.rand(n_samples, n_features)


@pytest.fixture
def sample_labels():
    """创建测试标签数据"""
    n_samples = 100
    return np.random.randint(0, 3, n_samples)


@pytest.fixture
def sample_transition_matrix():
    """创建测试转移矩阵"""
    # 3x3的转移矩阵
    return np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])


@pytest.fixture
def sample_features_df():
    """创建测试特征数据框"""
    n_samples = 100
    data = {
        "feat_delay_std": np.random.rand(n_samples),
        "feat_loss_burst_ratio": np.random.rand(n_samples),
        "feat_burst_duration": np.random.rand(n_samples),
        "feat_burst_intensity": np.random.rand(n_samples),
        "feat_delay_trend": np.random.randn(n_samples),
        "feat_delay_acf_5": np.random.rand(n_samples),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_evaluation_results():
    """创建测试评估结果"""
    return {
        "metrics": {},
        "transition_matrix": np.array(
            [[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]]
        ),
    }


@pytest.fixture
def tmp_output_dir(tmp_path):
    """创建临时输出目录"""
    output_dir = tmp_path / "visualization_results"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def test_visualizer_initialization(visualizer):
    """测试可视化器初始化"""
    assert visualizer is not None
    assert visualizer.base_visualizer is not None
    assert visualizer.feature_visualizer is not None
    assert visualizer.behavior_visualizer is not None
    assert visualizer.report_generator is not None


def test_visualize_evaluation_results(visualizer, sample_evaluation_results):
    """测试可视化评估结果"""
    # 可视化评估结果
    result = visualizer.visualize_evaluation_results(sample_evaluation_results)

    # 验证返回结果是HTML片段
    assert isinstance(result, str)
    assert result.startswith("<h2>")


def test_visualize_feature_analysis(
    visualizer, sample_features_df, sample_labels, tmp_output_dir
):
    """测试可视化特征分析"""
    # 使用generate_all_visualizations方法来生成特征分析可视化
    X = sample_features_df.values
    labels = sample_labels
    transition_matrix = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
    visualizer.generate_all_visualizations(X, labels, transition_matrix, tmp_output_dir)
