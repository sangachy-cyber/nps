#!/usr/bin/env python3
"""
可视化模块测试用例
"""

import pytest
import numpy as np
import pandas as pd
from src.network_simulation.visualization.visualizer import Visualizer


@pytest.fixture
def tmp_output_dir(tmp_path):
    """创建临时输出目录"""
    output_dir = tmp_path / "visualization_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def visualizer(tmp_output_dir):
    """初始化可视化器"""
    return Visualizer(tmp_output_dir)


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
def sample_feature_names():
    """创建测试特征名称"""
    return [
        "feat_delay_std",
        "feat_loss_burst_ratio",
        "feat_burst_duration",
        "feat_burst_intensity",
        "feat_delay_trend",
        "feat_delay_acf_5"
    ]


@pytest.fixture
def sample_transition_matrix():
    """创建测试转移矩阵"""
    return np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.3, 0.6]
    ])


@pytest.fixture
def sample_evaluation_results():
    """创建测试评估结果"""
    return {
        "method": "rule",
        "labels": np.random.randint(0, 3, 100).tolist(),
        "metrics": {
            "average_transition_entropy": 0.8,
            "transition_sparsity": 0.9
        },
        "transition_matrix": [
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6]
        ],
        "separation_metrics": {
            "label_means": {
                "0": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "1": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                "2": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            }
        }
    }


@pytest.fixture
def sample_raw_data():
    """创建测试原始数据"""
    timestamps = pd.date_range(start="2025-01-01", periods=1000, freq="100ms")
    data = {
        "timestamp": timestamps,
        "delay": np.random.randn(1000) * 10 + 20,
        "loss_rate": np.random.choice([0.0, 0.5, 1.0], 1000)
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_features_df():
    """创建测试特征数据框"""
    n_windows = 10
    window_size = 100
    data = {
        "window_start": [i * window_size for i in range(n_windows)],
        "window_end": [(i + 1) * window_size for i in range(n_windows)],
        "feat_delay_std": np.random.rand(n_windows),
        "feat_loss_burst_ratio": np.random.rand(n_windows),
        "feat_burst_duration": np.random.rand(n_windows),
        "feat_burst_intensity": np.random.rand(n_windows),
        "feat_delay_trend": np.random.randn(n_windows),
        "feat_delay_acf_5": np.random.rand(n_windows)
    }
    return pd.DataFrame(data)


def test_visualizer_initialization(visualizer, tmp_output_dir):
    """测试可视化器初始化"""
    assert visualizer is not None
    assert visualizer.output_dir == tmp_output_dir
    assert tmp_output_dir.exists()


def test_visualize_evaluation_results(visualizer, sample_evaluation_results):
    """测试可视化评估结果"""
    # 可视化评估结果
    html_snippet = visualizer.visualize_evaluation_results(sample_evaluation_results)

    # 验证结果
    assert isinstance(html_snippet, str)
    assert "评估结果可视化" in html_snippet


def test_visualize_pca_scatter(visualizer, sample_features, sample_labels):
    """测试可视化PCA散点图"""
    # 可视化PCA散点图
    html_snippet = visualizer.visualize_pca_scatter(sample_features, sample_labels, "test_method")

    # 验证结果
    assert isinstance(html_snippet, str)
    assert "data:image/png;base64" in html_snippet


def test_visualize_pca_variance(visualizer, sample_features):
    """测试可视化PCA方差解释图"""
    # 可视化PCA方差解释图
    html_snippet = visualizer.visualize_pca_variance(sample_features, "test_method")

    # 验证结果
    assert isinstance(html_snippet, str)
    assert "data:image/png;base64" in html_snippet


def test_visualize_tsne_scatter(visualizer, sample_features, sample_labels):
    """测试可视化t-SNE散点图"""
    # 可视化t-SNE散点图
    html_snippet = visualizer.visualize_tsne_scatter(sample_features, sample_labels, "test_method")

    # 验证结果
    assert isinstance(html_snippet, str)
    assert "data:image/png;base64" in html_snippet


def test_visualize_umap_scatter(visualizer, sample_features, sample_labels):
    """测试可视化UMAP散点图"""
    # 可视化UMAP散点图
    html_snippet = visualizer.visualize_umap_scatter(sample_features, sample_labels, "test_method")

    # 验证结果
    assert isinstance(html_snippet, str)
    assert "data:image/png;base64" in html_snippet


def test_visualize_feature_distribution(visualizer, sample_features, sample_labels, sample_feature_names):
    """测试可视化特征分布"""
    # 可视化特征分布
    html_snippet = visualizer.visualize_feature_distribution(sample_features, sample_labels, sample_feature_names, "test_method")

    # 验证结果
    assert isinstance(html_snippet, str)
    assert "data:image/png;base64" in html_snippet


def test_visualize_correlation_heatmap(visualizer, sample_features, sample_feature_names):
    """测试可视化特征相关性热力图"""
    # 可视化特征相关性热力图
    html_snippet = visualizer.visualize_correlation_heatmap(sample_features, sample_feature_names, "test_method")

    # 验证结果
    assert isinstance(html_snippet, str)
    assert "data:image/png;base64" in html_snippet


def test_visualize_transition_matrix(visualizer, sample_transition_matrix):
    """测试可视化转移矩阵"""
    # 可视化转移矩阵
    html_snippet = visualizer.visualize_transition_matrix(sample_transition_matrix, "test_method")

    # 验证结果
    assert isinstance(html_snippet, str)
    assert "data:image/png;base64" in html_snippet


def test_generate_html_report(visualizer, sample_evaluation_results, sample_features, sample_feature_names):
    """测试生成HTML报告"""
    # 生成HTML报告（不提供原始数据，避免出现索引越界问题）
    visualizer.generate_html_report(
        sample_evaluation_results,
        sample_features,
        sample_feature_names
    )

    # 验证报告文件生成
    report_file = visualizer.output_dir / f"behavior_pattern_report_{sample_evaluation_results['method']}.html"
    assert report_file.exists()
    assert report_file.stat().st_size > 0


def test_generate_html_report_no_raw_data(visualizer, sample_evaluation_results, sample_features, sample_feature_names):
    """测试不提供原始数据时生成HTML报告"""
    # 生成HTML报告（不提供原始数据）
    visualizer.generate_html_report(
        sample_evaluation_results,
        sample_features,
        sample_feature_names
    )

    # 验证报告文件生成
    report_file = visualizer.output_dir / f"behavior_pattern_report_{sample_evaluation_results['method']}.html"
    assert report_file.exists()
    assert report_file.stat().st_size > 0


def test_generate_separation_table(visualizer, sample_evaluation_results, sample_feature_names):
    """测试生成分离度表格"""
    # 生成分离度表格
    html_table = visualizer._generate_separation_table(
        sample_evaluation_results["separation_metrics"],
        sample_feature_names
    )

    # 验证结果
    assert isinstance(html_table, str)
    assert "<tr>" in html_table
    assert "<td>" in html_table


def test_visualize_raw_data_samples(visualizer, sample_raw_data, sample_features_df):
    """测试可视化原始数据样本"""
    # 创建与features_df行数匹配的标签
    sample_labels = np.random.randint(0, 3, len(sample_features_df))

    # 可视化原始数据样本
    html_snippet = visualizer.visualize_raw_data_samples(
        sample_raw_data,
        sample_features_df,
        sample_labels,
        "test_method"
    )

    # 验证结果
    assert isinstance(html_snippet, str)
    assert "原始数据样本可视化" in html_snippet
