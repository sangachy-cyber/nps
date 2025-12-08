#!/usr/bin/env python3
"""
扩散模型测试用例
"""

import pytest
import torch
from src.network_simulation.condition_generation.diffusion_model import (
    ConditionDiffusionModel,
    UNet,
    UNetBlock,
    Time2Vec,
    CausalDilatedConv,
)


@pytest.fixture
def device():
    """获取可用设备"""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@pytest.fixture
def diffusion_model(device):
    """初始化扩散模型"""
    model = ConditionDiffusionModel(
        input_dim=2,
        num_behaviors=5,
        behavior_embed_dim=32,
        T=1000,
    )
    model.to(device)
    model.eval()
    return model


@pytest.fixture
def sample_input(device):
    """创建测试样本输入"""
    batch_size = 1
    seq_len = 600
    input_dim = 2

    # 随机输入数据
    x = torch.randn(batch_size, seq_len, input_dim, device=device)

    # 随机时间步
    t = torch.randint(0, 1000, (batch_size, seq_len, 1), device=device)

    # 随机行为ID
    behavior_ids = torch.randint(0, 5, (batch_size, seq_len), device=device)

    return x, t, behavior_ids


def test_time2vec_forward():
    """测试 Time2Vec 前向传播"""
    t2v = Time2Vec(32)
    x = torch.randn(1, 10, 1)
    output = t2v(x)
    assert output.shape == (1, 10, 32)


def test_causal_dilated_conv_forward():
    """测试因果膨胀卷积前向传播"""
    conv = CausalDilatedConv(
        in_channels=64, out_channels=128, kernel_size=3, dilation=1
    )
    x = torch.randn(1, 64, 100)
    output = conv(x)
    assert output.shape == (1, 128, 100)


def test_unet_block_forward():
    """测试 UNetBlock 前向传播"""
    unet_block = UNetBlock(in_channels=64, out_channels=128, kernel_size=3, dilation=1)
    x = torch.randn(1, 64, 100)
    output = unet_block(x)
    assert output.shape == (1, 128, 100)


def test_unet_forward(sample_input, device):
    """测试 UNet 前向传播"""
    x, t, behavior_embed = sample_input
    behavior_embed = torch.randn(1, 600, 32, device=device)  # 模拟行为嵌入

    # 将 x 转换为 1D 卷积所需的形状 (batch, channels, seq_len)
    # 注意：这里注释掉未使用的转换，如需使用请取消注释
    # x_conv = x.transpose(1, 2)  # (1, 2, 600)

    unet = UNet(input_dim=2, behavior_embed_dim=32)
    unet.to(device)

    output = unet(x, t, behavior_embed)
    assert output.shape == x.shape


def test_diffusion_model_forward(diffusion_model, sample_input):
    """测试扩散模型前向传播"""
    x, t, behavior_ids = sample_input
    output = diffusion_model(x, t, behavior_ids)
    assert output.shape == x.shape


def test_diffusion_model_sample(diffusion_model, device):
    """测试扩散模型采样功能"""
    # 创建行为ID张量
    batch_size = 1
    seq_len = 600
    behavior_ids = torch.randint(0, 5, (batch_size, seq_len), device=device)

    # 生成样本
    generated = diffusion_model.sample(behavior_ids, device)
    assert generated.shape == (batch_size, seq_len, 2)


def test_diffusion_model_repeated_sample(diffusion_model, device):
    """测试扩散模型重复采样"""
    # 创建行为ID张量
    batch_size = 1
    seq_len = 600
    behavior_ids = torch.randint(0, 5, (batch_size, seq_len), device=device)

    # 生成两次样本
    generated1 = diffusion_model.sample(behavior_ids, device)
    generated2 = diffusion_model.sample(behavior_ids, device)

    # 确保两次生成的样本不同（随机性测试）
    assert not torch.allclose(generated1, generated2)
    assert generated1.shape == generated2.shape
