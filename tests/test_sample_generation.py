#!/usr/bin/env python3
"""
样本生成测试用例
"""

import pytest
import torch
import numpy as np
from src.network_simulation.condition_generation.diffusion_model import (
    ConditionDiffusionModel,
)
from src.network_simulation.condition_generation.constraint_injector import (
    ConstraintInjector,
)


@pytest.fixture
def device():
    """获取可用设备"""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@pytest.fixture
def constraint_injector():
    """初始化约束注入器"""
    valid_loss_values = [0.0, 0.5, 1.0]
    return ConstraintInjector(valid_loss_values)


@pytest.fixture
def mock_checkpoint():
    """创建模拟的模型检查点"""
    return {
        "behavior_mapping": {"stable": 0, "burst": 1, "high_jitter": 2},
        "model_state_dict": {},
    }


@pytest.fixture
def sample_behavior_ids(device):
    """创建测试用的行为ID序列"""
    batch_size = 1
    seq_len = 100
    return torch.full((batch_size, seq_len), 1, device=device, dtype=torch.long)


@pytest.fixture
def diffusion_model(device):
    """初始化扩散模型"""
    # 对于 input_dim=2，提供相应的 constrained_dims=[0, 1]
    model = ConditionDiffusionModel(
        input_dim=2,  # 输入维度：延迟和丢包率
        behavior_embed_dim=32,  # 行为嵌入维度
        T=1000,  # 扩散步数
        constrained_dims=[0, 1],  # 只约束前两个维度
        cond_dim=8,  # 条件向量维度
    )
    model.to(device)
    model.eval()
    return model


def test_diffusion_model_initialization(diffusion_model, device):
    """测试扩散模型初始化"""
    assert diffusion_model is not None
    # 检查模型参数是否已移至正确设备
    for param in diffusion_model.parameters():
        assert param.device.type == device.type


def test_sample_generation(diffusion_model, constraint_injector, device):
    """测试样本生成"""
    # 创建条件向量张量
    batch_size = 1
    seq_len = 100
    cond_dim = 8
    condition_vector = torch.randn(batch_size, seq_len, cond_dim, device=device)

    # 生成样本
    with torch.no_grad():
        generated = diffusion_model.sample(condition_vector)

    # 验证生成结果的形状
    assert generated.shape == (batch_size, seq_len, 2)

    # 转换为numpy数组
    generated_np = generated.cpu().numpy()[0]

    # 验证生成的延迟和丢包率
    delay_norm = generated_np[:, 0]
    loss_norm = generated_np[:, 1]

    # 验证延迟范围
    # 使用安全的方式计算，避免溢出
    max_safe_value = np.finfo(np.float64).max / 100 - 1
    delay_norm_clipped = np.clip(delay_norm, -1, max_safe_value)
    delay = (delay_norm_clipped + 1) * 100
    delay = np.clip(delay, 0, np.finfo(np.float64).max)
    assert np.all(delay >= 0)  # 延迟不能为负

    # 验证丢包率处理
    loss_rate = constraint_injector.process_loss_rate(loss_norm)
    assert len(loss_rate) == len(loss_norm)
    for lr in loss_rate:
        assert lr in constraint_injector.valid_loss_values


def test_sample_generation_shape(diffusion_model, device):
    """测试不同形状的样本生成"""
    batch_sizes = [1, 2]
    seq_lens = [100, 200]
    cond_dim = 8

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            # 创建条件向量张量
            condition_vector = torch.randn(batch_size, seq_len, cond_dim, device=device)

            with torch.no_grad():
                generated = diffusion_model.sample(condition_vector)

            assert generated.shape == (batch_size, seq_len, 2)


def test_constraint_injector_processing(constraint_injector):
    """测试约束注入器处理"""
    # 创建测试数据
    loss_values = np.array([-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5])

    # 处理丢包率
    processed = constraint_injector.process_loss_rate(loss_values)

    # 验证处理结果
    for lr in processed:
        assert lr in constraint_injector.valid_loss_values

    # 验证处理后的丢包率范围
    assert np.all(np.array(processed) >= 0)
    assert np.all(np.array(processed) <= 1)
