#!/usr/bin/env python3
"""
约束注入模块测试用例
"""

import pytest
import numpy as np
from src.network_simulation.condition_generation.constraint_injector import (
    ConstraintInjector,
)


@pytest.fixture
def valid_loss_values():
    """创建合法丢包值列表"""
    return [0.0, 0.5, 1.0]


@pytest.fixture
def constraint_injector(valid_loss_values):
    """初始化约束注入器"""
    return ConstraintInjector(valid_loss_values)


@pytest.fixture
def sample_delay():
    """创建测试延迟数据"""
    return np.random.randn(1000) * 10 + 20


@pytest.fixture
def sample_loss_rate():
    """创建测试丢包率数据"""
    return np.random.choice([0.0, 0.25, 0.5, 0.75, 1.0], 1000)


def test_constraint_injector_init(constraint_injector, valid_loss_values):
    """测试约束注入器初始化"""
    assert constraint_injector.valid_loss_values == valid_loss_values


def test_constraint_injector_process_loss_rate(constraint_injector, sample_loss_rate):
    """测试丢包率处理功能"""
    processed_loss_rate = constraint_injector.process_loss_rate(sample_loss_rate)

    # 验证处理后的丢包率都在合法值列表中
    for lr in processed_loss_rate:
        assert lr in constraint_injector.valid_loss_values


def test_constraint_injector_validate_sequence(
    constraint_injector, sample_delay, sample_loss_rate
):
    """测试序列验证功能"""
    validation = constraint_injector.validate_sequence(sample_delay, sample_loss_rate)

    # 验证验证结果包含必要的字段
    assert "delay_valid" in validation
    assert "loss_rate_valid" in validation
    assert "all_valid" in validation

    # 测试负延迟应该被标记为无效
    negative_delay = sample_delay.copy()
    negative_delay[0] = -1.0  # 设置一个负延迟
    validation_negative = constraint_injector.validate_sequence(
        negative_delay, sample_loss_rate
    )
    assert not validation_negative["delay_valid"]

    # 测试所有非负延迟应该被标记为有效
    positive_delay = sample_delay.copy()
    positive_delay[positive_delay < 0] = 0.0  # 将所有负延迟设置为0
    validation_positive = constraint_injector.validate_sequence(
        positive_delay, sample_loss_rate
    )
    assert validation_positive["delay_valid"]


def test_constraint_injector_fix_sequence(constraint_injector):
    """测试序列修复功能"""
    # 创建无效的延迟和丢包率数据
    invalid_delay = np.array([-1.0, 0.5, -2.0, 1.0, -0.5])
    invalid_loss_rate = np.array([0.25, 0.75, 0.1, 0.9, 0.3])

    # 修复延迟
    fixed_delay = np.clip(invalid_delay, a_min=0, a_max=None)
    assert all(delay >= 0 for delay in fixed_delay)

    # 修复丢包率
    fixed_loss_rate = constraint_injector.process_loss_rate(invalid_loss_rate)
    for lr in fixed_loss_rate:
        assert lr in constraint_injector.valid_loss_values


def test_constraint_injector_edge_cases(constraint_injector):
    """测试边界情况"""
    # 空序列测试
    empty_delay = np.array([])
    empty_loss_rate = np.array([])
    validation = constraint_injector.validate_sequence(empty_delay, empty_loss_rate)
    assert validation["all_valid"]

    # 单个元素
    single_delay = np.array([10.0])
    single_loss_rate = np.array([0.01])
    validation = constraint_injector.validate_sequence(single_delay, single_loss_rate)
    assert validation["all_valid"]

    # 全零数据
    zero_delay = np.array([0.0] * 100)
    zero_loss_rate = np.array([0.0] * 100)
    validation = constraint_injector.validate_sequence(zero_delay, zero_loss_rate)
    assert validation["all_valid"]
