#!/usr/bin/env python3
"""
归一化模块测试用例
"""

import numpy as np
import pytest
from network_simulation.condition_generation.normalization import (
    normalize_delay,
    denormalize_delay,
    normalize_loss_rate,
    denormalize_loss_rate
)


class TestNormalization:
    """
    归一化模块测试类
    """
    
    def test_delay_normalization_consistency(self):
        """
        测试延迟归一化和反归一化的一致性
        """
        # 创建测试数据
        delay_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        # 归一化
        delay_norm, delay_norm_params = normalize_delay(delay_values)
        
        # 反归一化
        delay_denorm = denormalize_delay(delay_norm, delay_norm_params)
        
        # 检查反归一化后的数据与原始数据的差异
        # 由于归一化和反归一化过程中会有一定的精度损失，我们使用相对误差来判断
        relative_error = np.abs(delay_denorm - delay_values) / delay_values
        assert np.all(relative_error < 0.1), f"延迟反归一化误差过大: {relative_error}"
    
    def test_delay_normalization_with_zero_values(self):
        """
        测试延迟归一化和反归一化处理零值的情况
        """
        # 创建包含零值的测试数据
        delay_values = np.array([0.0, 10.0, 20.0, 30.0, 0.0])
        
        # 归一化
        delay_norm, delay_norm_params = normalize_delay(delay_values)
        
        # 反归一化
        delay_denorm = denormalize_delay(delay_norm, delay_norm_params)
        
        # 检查反归一化后的数据与原始数据的差异
        relative_error = np.abs(delay_denorm - delay_values) / (delay_values + 1e-8)  # 避免除以零
        assert np.all(relative_error < 0.1), f"延迟反归一化误差过大: {relative_error}"
        
        # 检查反归一化后的数据是否为非负
        assert np.all(delay_denorm >= 0), f"反归一化后的延迟包含负值: {delay_denorm}"
    
    def test_loss_rate_normalization_consistency(self):
        """
        测试丢包率归一化和反归一化的一致性
        """
        # 创建测试数据和合法丢包值
        loss_rate_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        valid_loss_values = [0.0, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0]
        
        # 归一化
        loss_norm, loss_norm_params = normalize_loss_rate(loss_rate_values, valid_loss_values)
        
        # 反归一化
        loss_denorm = denormalize_loss_rate(loss_norm, loss_norm_params)
        
        # 检查反归一化后的数据与原始数据是否完全一致
        # 由于丢包率归一化是映射到离散的合法值，所以应该完全一致
        assert np.all(loss_denorm == loss_rate_values), f"丢包率反归一化结果不一致: 原始 {loss_rate_values}, 反归一化 {loss_denorm}"
    
    def test_loss_rate_normalization_with_unknown_values(self):
        """
        测试丢包率归一化处理未知值的情况
        """
        # 创建包含未知值的测试数据
        loss_rate_values = np.array([0.1, 0.3, 0.9, 1.1])
        valid_loss_values = [0.0, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0]
        
        # 归一化
        loss_norm, loss_norm_params = normalize_loss_rate(loss_rate_values, valid_loss_values)
        
        # 反归一化
        loss_denorm = denormalize_loss_rate(loss_norm, loss_norm_params)
        
        # 检查反归一化后的数据是否为合法值
        assert np.all(np.isin(loss_denorm, valid_loss_values)), f"反归一化后的丢包率包含非法值: {loss_denorm}"
        
        # 检查反归一化后的数据是否在[0, 1]范围内
        assert np.all(loss_denorm >= 0), f"反归一化后的丢包率包含负值: {loss_denorm}"
        assert np.all(loss_denorm <= 1), f"反归一化后的丢包率超出1: {loss_denorm}"
    
    def test_loss_rate_normalization_with_single_valid_value(self):
        """
        测试丢包率归一化处理只有一个有效值的情况
        """
        # 创建测试数据和只有一个合法丢包值的情况
        loss_rate_values = np.array([0.0, 0.0, 0.0, 0.0])
        valid_loss_values = [0.0]
        
        # 归一化
        loss_norm, loss_norm_params = normalize_loss_rate(loss_rate_values, valid_loss_values)
        
        # 反归一化
        loss_denorm = denormalize_loss_rate(loss_norm, loss_norm_params)
        
        # 检查反归一化后的数据是否为合法值
        assert np.all(loss_denorm == valid_loss_values[0]), f"丢包率反归一化结果不一致: 原始 {loss_rate_values}, 反归一化 {loss_denorm}"
    
    def test_delay_normalization_with_constant_values(self):
        """
        测试延迟归一化处理常量值的情况
        """
        # 创建全部为同一个值的测试数据
        delay_values = np.array([20.0, 20.0, 20.0, 20.0])
        
        # 归一化
        delay_norm, delay_norm_params = normalize_delay(delay_values)
        
        # 反归一化
        delay_denorm = denormalize_delay(delay_norm, delay_norm_params)
        
        # 检查反归一化后的数据是否与原始数据一致
        relative_error = np.abs(delay_denorm - delay_values) / delay_values
        assert np.all(relative_error < 0.1), f"延迟反归一化误差过大: {relative_error}"
    
    def test_loss_rate_normalization_with_edge_values(self):
        """
        测试丢包率归一化处理边界值的情况
        """
        # 创建包含边界值的测试数据
        loss_rate_values = np.array([0.0, 1.0, 0.0, 1.0])
        valid_loss_values = [0.0, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0]
        
        # 归一化
        loss_norm, loss_norm_params = normalize_loss_rate(loss_rate_values, valid_loss_values)
        
        # 反归一化
        loss_denorm = denormalize_loss_rate(loss_norm, loss_norm_params)
        
        # 检查反归一化后的数据是否与原始数据一致
        assert np.all(loss_denorm == loss_rate_values), f"丢包率反归一化结果不一致: 原始 {loss_rate_values}, 反归一化 {loss_denorm}"
