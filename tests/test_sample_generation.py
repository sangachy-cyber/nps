#!/usr/bin/env python3
"""
测试样本生成脚本
"""

import sys
import os

sys.path.append(os.path.abspath("src"))

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from network_simulation.condition_generation.diffusion_model import (
    ConditionDiffusionModel,
)
from network_simulation.condition_generation.constraint_injector import (
    ConstraintInjector,
)


def test_sample_generation():
    """测试样本生成"""
    print("开始测试样本生成...")

    # 设置设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    model_path = Path(
        "/Users/xiaotuanzi/PycharmProjects/NPS/data/results/e2e_pipeline/train_results/diffusion_model_final.pth"
    )
    print(f"正在加载模型: {model_path}")

    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    behavior_mapping = checkpoint["behavior_mapping"]
    num_behaviors = len(behavior_mapping)

    # 初始化模型
    model = ConditionDiffusionModel(
        input_dim=2,  # 输入维度：延迟和丢包率
        num_behaviors=num_behaviors,
        behavior_embed_dim=32,  # 行为嵌入维度
        T=1000,  # 扩散步数
    )

    # 加载模型状态
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("模型加载完成！")

    # 创建一个简单的行为ID序列
    batch_size = 1
    seq_len = 1000  # 生成1000个样本，而不是6000个
    behavior_ids = torch.full((batch_size, seq_len), 1, device=device, dtype=torch.long)

    print(f"生成样本：batch_size={batch_size}, seq_len={seq_len}")

    # 生成样本
    with torch.no_grad():
        generated = model.sample(behavior_ids, device)

    print("样本生成完成！")
    print(f"生成样本形状: {generated.shape}")

    # 反归一化
    delay_scaler_mean_ = checkpoint["delay_scaler_mean_"]
    delay_scaler_scale_ = checkpoint["delay_scaler_scale_"]

    generated_np = generated.cpu().numpy()[0]

    # 反归一化延迟
    delay_norm = generated_np[:, 0]
    delay = (delay_norm * delay_scaler_scale_) + delay_scaler_mean_

    # 反归一化丢包率
    loss_norm = generated_np[:, 1]

    print(f"生成的延迟范围: {delay.min():.2f} - {delay.max():.2f} ms")
    print(f"生成的丢包率范围: {loss_norm.min():.4f} - {loss_norm.max():.4f}")

    print("测试完成！")


if __name__ == "__main__":
    test_sample_generation()
