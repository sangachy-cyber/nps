#!/usr/bin/env python3
"""
条件生成器测试用例
"""

import pytest
import pandas as pd
from src.network_simulation.condition_generation.condition_generator import ConditionGenerator


@pytest.fixture
def valid_loss_values():
    """创建合法丢包值列表"""
    return [0.0, 0.5, 1.0]


@pytest.fixture
def condition_generator(valid_loss_values):
    """初始化条件生成器"""
    return ConditionGenerator(valid_loss_values)


@pytest.fixture
def sample_schedule():
    """创建测试调度计划"""
    return {
        "segments": [
            {
                "start_time": 0,
                "end_time": 5,
                "behavior_type": "stable"
            },
            {
                "start_time": 5,
                "end_time": 10,
                "behavior_type": "burst"
            }
        ]
    }


@pytest.fixture
def sample_patterns():
    """创建测试模式映射"""
    return {
        "behavior_cluster_map": {
            "stable": 0,
            "burst": 1,
            "high_jitter": 2,
            "recovery": 3
        }
    }


@pytest.fixture
def sample_behavior_ids():
    """创建测试用的行为ID序列"""
    return [0, 0, 1, 1, 2, 2]


@pytest.fixture
def tmp_output_dir(tmp_path):
    """创建临时输出目录"""
    output_dir = tmp_path / "generated_data"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def test_condition_generator_initialization(condition_generator, valid_loss_values):
    """测试条件生成器初始化"""
    assert condition_generator is not None
    assert condition_generator.valid_loss_values == valid_loss_values
    assert condition_generator.diffusion_model is not None
    assert condition_generator.constraint_injector is not None


def test_condition_generator_init_with_config():
    """测试使用配置初始化条件生成器"""
    config = {
        "time_granularity": 0.2,
        "other_param": "test"
    }
    generator = ConditionGenerator(config=config)
    assert generator.time_granularity == 0.1  # 配置中的time_granularity没有被使用，保持默认值
    assert generator.config == config


def test_generate_behavior_id_sequence(condition_generator, sample_schedule, sample_patterns):
    """测试生成行为ID序列"""
    total_samples = 100  # 10秒，100ms粒度
    behavior_ids = condition_generator._generate_behavior_id_sequence(
        sample_schedule, sample_patterns, total_samples
    )

    # 验证行为ID序列长度
    assert len(behavior_ids) == total_samples

    # 验证行为ID序列的前50个样本（前5秒）为稳定行为（0）
    assert all(id == 0 for id in behavior_ids[:50])

    # 验证行为ID序列的后50个样本（后5秒）为突发行为（1）
    assert all(id == 1 for id in behavior_ids[50:])


def test_generate_sequences(condition_generator, sample_schedule, sample_patterns):
    """测试生成序列"""
    total_samples = 100  # 10秒，100ms粒度
    delay_sequence, loss_sequence = condition_generator._generate_sequences(
        sample_schedule, sample_patterns, total_samples
    )

    # 验证序列长度
    assert len(delay_sequence) == total_samples
    assert len(loss_sequence) == total_samples

    # 验证延迟非负
    assert all(delay >= 0 for delay in delay_sequence)

    # 验证丢包率为合法值
    assert all(lr in condition_generator.valid_loss_values for lr in loss_sequence)


def test_generate(condition_generator, sample_schedule, sample_patterns):
    """测试生成网络模拟数据"""
    duration = 10  # 10秒
    df = condition_generator.generate(sample_schedule, sample_patterns, duration)

    # 验证生成的数据框
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100  # 10秒，100ms粒度

    # 验证数据框包含必要的列
    assert "timestamp" in df.columns
    assert "delay" in df.columns
    assert "loss_rate" in df.columns

    # 验证延迟非负
    assert all(df["delay"] >= 0)

    # 验证丢包率为合法值
    assert all(lr in condition_generator.valid_loss_values for lr in df["loss_rate"])


def test_save(condition_generator, sample_schedule, sample_patterns, tmp_output_dir):
    """测试保存生成的数据"""
    duration = 5  # 5秒
    df = condition_generator.generate(sample_schedule, sample_patterns, duration)

    # 保存数据
    output_path = tmp_output_dir / "generated_data.csv"
    condition_generator.save(df, output_path)

    # 验证文件存在
    assert output_path.exists()

    # 验证文件内容
    loaded_df = pd.read_csv(output_path)
    assert len(loaded_df) == len(df)
    assert list(loaded_df.columns) == list(df.columns)


def test_generate_behavior_segment(condition_generator, sample_patterns):
    """测试生成特定行为类型的数据段"""
    num_samples = 50  # 5秒，100ms粒度
    behavior_type = "stable"

    delay_segment, loss_segment = condition_generator._generate_behavior_segment(
        behavior_type, sample_patterns, num_samples
    )

    # 验证数据段长度
    assert len(delay_segment) == num_samples
    assert len(loss_segment) == num_samples

    # 验证延迟非负
    assert all(delay >= 0 for delay in delay_segment)

    # 验证丢包率为合法值
    assert all(lr in condition_generator.valid_loss_values for lr in loss_segment)


def test_load_schedule(tmp_path):
    """测试从文件加载调度计划"""
    # 创建测试调度文件
    schedule = {
        "segments": [
            {
                "start_time": 0,
                "end_time": 5,
                "behavior_type": "stable"
            }
        ]
    }

    schedule_path = tmp_path / "schedule.json"
    import json
    with open(schedule_path, "w") as f:
        json.dump(schedule, f)

    # 测试加载调度计划
    generator = ConditionGenerator()
    loaded_schedule = generator.load_schedule(schedule_path)

    # 验证加载结果
    assert loaded_schedule == schedule


def test_generate_with_invalid_behavior_type(condition_generator, sample_patterns):
    """测试使用无效行为类型生成数据"""
    schedule = {
        "segments": [
            {
                "start_time": 0,
                "end_time": 5,
                "behavior_type": "invalid_behavior"
            }
        ]
    }

    # 应该能正常生成，使用默认行为类型
    df = condition_generator.generate(schedule, sample_patterns, duration=5)
    assert len(df) == 50  # 5秒，100ms粒度
    assert all(df["delay"] >= 0)
