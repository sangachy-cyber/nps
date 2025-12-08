#!/usr/bin/env python3
"""
评估器测试用例
"""

import pytest
import pandas as pd
import numpy as np
from src.network_simulation.evaluation.evaluator import Evaluator


@pytest.fixture
def evaluator():
    """初始化评估器"""
    return Evaluator()


@pytest.fixture
def sample_generated_data():
    """创建测试生成数据"""
    timestamps = pd.date_range(start="2025-01-01", periods=1000, freq="100ms")
    data = {
        "timestamp": timestamps,
        "delay": np.random.randn(1000) * 10 + 20,  # 均值20，标准差10
        "loss_rate": np.random.choice([0.0, 0.5, 1.0], 1000),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(tmp_path, sample_generated_data):
    """创建测试CSV文件"""
    csv_file = tmp_path / "sample_generated.csv"
    sample_generated_data.to_csv(csv_file, index=False)
    return csv_file


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
    return np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.3, 0.6]
    ])


@pytest.fixture
def tmp_output_dir(tmp_path):
    """创建临时输出目录"""
    output_dir = tmp_path / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def test_evaluator_initialization(evaluator):
    """测试评估器初始化"""
    assert evaluator is not None
    assert evaluator.time_granularity == 0.1  # 默认100ms
    assert isinstance(evaluator.feature_columns, list)
    assert len(evaluator.feature_columns) == 6


def test_load_data(evaluator, sample_csv_file, sample_generated_data):
    """测试加载数据功能"""
    # 加载数据
    df = evaluator.load_data(sample_csv_file)

    # 验证数据加载结果
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_generated_data)
    assert list(df.columns) == ["timestamp", "delay", "loss_rate"]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert pd.api.types.is_float_dtype(df["delay"])
    assert pd.api.types.is_float_dtype(df["loss_rate"])


def test_evaluate(evaluator, sample_generated_data):
    """测试评估功能"""
    # 评估数据
    results = evaluator.evaluate(sample_generated_data)

    # 验证评估结果
    assert isinstance(results, dict)
    assert "statistical_fidelity" in results
    assert "indistinguishability" in results
    assert "dynamic_rationality" in results

    # 验证统计保真度评估结果
    assert isinstance(results["statistical_fidelity"], dict)
    assert "delay_mean" in results["statistical_fidelity"]
    assert "delay_std" in results["statistical_fidelity"]
    assert "loss_rate_mean" in results["statistical_fidelity"]
    assert "loss_rate_std" in results["statistical_fidelity"]

    # 验证不可区分性评估结果
    assert isinstance(results["indistinguishability"], dict)
    assert "discriminator_auc" in results["indistinguishability"]
    assert "tstr_keep_rate" in results["indistinguishability"]

    # 验证动态合理性评估结果
    assert isinstance(results["dynamic_rationality"], dict)
    assert "delay_acf_5" in results["dynamic_rationality"]
    assert "burst_statistics" in results["dynamic_rationality"]

    # 验证突发统计结果
    burst_stats = results["dynamic_rationality"]["burst_statistics"]
    assert "num_bursts" in burst_stats
    assert "avg_burst_duration" in burst_stats
    assert "burst_frequency" in burst_stats





def test_evaluate_transition_quality(evaluator, sample_transition_matrix):
    """测试转移质量评估"""
    # 评估转移质量
    results = evaluator.evaluate_transition_quality(sample_transition_matrix)

    # 验证评估结果
    assert isinstance(results, dict)
    assert "average_transition_entropy" in results
    assert "transition_sparsity" in results

    # 验证数值范围
    assert results["average_transition_entropy"] >= 0
    assert 0 <= results["transition_sparsity"] <= 1


def test_evaluate_behavior_separation(evaluator, sample_features, sample_labels):
    """测试行为分离评估"""
    # 评估行为分离
    results = evaluator.evaluate_behavior_separation(sample_features, sample_labels)

    # 验证评估结果
    assert isinstance(results, dict)
    assert "behavior_stats" in results
    assert "separation_metrics" in results
    assert "behavior_similarity" in results

    # 验证行为统计结果
    assert isinstance(results["behavior_stats"], dict)
    assert len(results["behavior_stats"]) > 0

    # 验证分离指标
    separation = results["separation_metrics"]
    assert "avg_inter_cluster_distance" in separation
    assert "avg_intra_cluster_distance" in separation
    assert "separation_index" in separation

    # 验证行为相似性
    similarity = results["behavior_similarity"]
    assert "similarity_matrix" in similarity
    assert "average_similarity" in similarity
    assert "min_similarity" in similarity
    assert "max_similarity" in similarity


def test_calculate_acf(evaluator):
    """测试自相关函数计算"""
    # 创建测试数据
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # 计算自相关
    acf = evaluator._calculate_acf(data, lag=2)

    # 验证结果
    assert isinstance(acf, float)
    assert -1 <= acf <= 1


def test_calculate_burst_statistics(evaluator):
    """测试突发统计计算"""
    # 创建测试丢包率数据
    loss_rates = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])

    # 计算突发统计
    burst_stats = evaluator._calculate_burst_statistics(loss_rates)

    # 验证结果
    assert isinstance(burst_stats, dict)
    assert "num_bursts" in burst_stats
    assert "avg_burst_duration" in burst_stats
    assert "burst_frequency" in burst_stats

    # 验证具体数值
    assert burst_stats["num_bursts"] == 3  # 3个突发
    assert burst_stats["avg_burst_duration"] == (2 + 1 + 3) * 0.1 / 3  # 平均突发持续时间


def test_save_evaluation_results(evaluator, sample_generated_data, tmp_output_dir):
    """测试保存评估结果"""
    # 评估数据
    results = evaluator.evaluate(sample_generated_data)

    # 保存结果
    evaluator.save(results, tmp_output_dir)

    # 验证文件存在
    json_path = tmp_output_dir / "evaluation_results.json"
    summary_path = tmp_output_dir / "evaluation_summary.txt"
    markdown_path = tmp_output_dir / "comprehensive_evaluation_report.md"
    html_path = tmp_output_dir / "comprehensive_evaluation_report.html"

    assert json_path.exists()
    assert summary_path.exists()
    assert markdown_path.exists()
    assert html_path.exists()

    # 验证文件内容不为空
    assert json_path.stat().st_size > 0
    assert summary_path.stat().st_size > 0
    assert markdown_path.stat().st_size > 0
    assert html_path.stat().st_size > 0


def test_evaluate_with_empty_data(evaluator):
    """测试评估空数据"""
    # 创建空数据框
    empty_df = pd.DataFrame(columns=["timestamp", "delay", "loss_rate"])

    # 评估空数据
    results = evaluator.evaluate(empty_df)

    # 验证评估结果
    assert isinstance(results, dict)
    assert "statistical_fidelity" in results
    assert "indistinguishability" in results
    assert "dynamic_rationality" in results
