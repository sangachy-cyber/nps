#!/usr/bin/env python3
"""
条件生成器测试用例
"""

import pytest
from src.network_simulation.condition_generation.condition_generator import (
    ConditionGenerator,
)


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
            {"start_time": 0, "end_time": 5, "behavior_type": "stable"},
            {"start_time": 5, "end_time": 10, "behavior_type": "burst"},
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
            "recovery": 3,
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
    assert condition_generator.valid_loss_values_up == valid_loss_values
    assert condition_generator.valid_loss_values_down is not None
    assert condition_generator.constraint_injector is not None


def test_condition_generator_init_with_config():
    """测试使用配置初始化条件生成器"""
    config = {"time_granularity": 0.2, "other_param": "test"}
    generator = ConditionGenerator(config=config)
    assert generator.time_granularity == 0.2  # 配置中的time_granularity被正确使用
    assert generator.config == config


def test_load_schedule(tmp_path):
    """测试从文件加载调度计划"""
    # 创建测试调度文件
    schedule = {
        "segments": [{"start_time": 0, "end_time": 5, "behavior_type": "stable"}]
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
