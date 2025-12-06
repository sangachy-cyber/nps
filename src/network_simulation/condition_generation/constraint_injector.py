#!/usr/bin/env python3
"""
物理约束注入机制
负责网络物理约束的验证和注入
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


class ConstraintInjector:
    """物理约束注入器"""

    def __init__(self, valid_loss_values: List[float]):
        self.valid_loss_values = valid_loss_values
        self.valid_loss_values_np = np.array(valid_loss_values)

    def validate_delay(self, delay: np.ndarray) -> bool:
        """验证延迟的物理合理性
        Args:
            delay: 延迟序列
        Returns:
            是否合法
        """
        return np.all(delay >= 0)

    def validate_loss_rate(self, loss_rate: np.ndarray) -> bool:
        """验证丢包率的物理合理性
        Args:
            loss_rate: 丢包率序列
        Returns:
            是否合法
        """
        # 检查是否在0-1范围内
        range_valid = np.all(loss_rate >= 0) and np.all(loss_rate <= 1)
        # 检查是否为合法值
        value_valid = np.all(np.isin(loss_rate, self.valid_loss_values_np))
        return range_valid and value_valid

    def validate_sequence(
        self, delay: np.ndarray, loss_rate: np.ndarray
    ) -> Dict[str, bool]:
        """验证完整序列的物理合理性
        Args:
            delay: 延迟序列
            loss_rate: 丢包率序列
        Returns:
            各项验证结果
        """
        return {
            "delay_valid": self.validate_delay(delay),
            "loss_rate_valid": self.validate_loss_rate(loss_rate),
            "all_valid": self.validate_delay(delay)
            and self.validate_loss_rate(loss_rate),
        }

    def process_delay(self, delay_norm: np.ndarray, scaler: object) -> np.ndarray:
        """处理延迟序列，确保物理合理性
        Args:
            delay_norm: 归一化的延迟序列
            scaler: 用于逆变换的scaler对象
        Returns:
            处理后的延迟序列
        """
        # 逆变换
        delay = scaler.inverse_transform(delay_norm.reshape(-1, 1)).flatten()
        # 确保非负
        delay = np.clip(delay, a_min=0, a_max=None)
        return delay

    def process_loss_rate(self, loss_norm: np.ndarray) -> np.ndarray:
        """处理丢包率序列，确保物理合理性
        Args:
            loss_norm: 归一化的丢包率序列（范围[-1, 1]）
        Returns:
            处理后的丢包率序列，确保为合法值
        """
        # 将[-1, 1]映射到[0, len(valid_loss_values)-1]
        idx_float = (loss_norm + 1) / 2 * (len(self.valid_loss_values) - 1)
        # 四舍五入并裁剪到有效范围
        idx = np.round(idx_float).astype(int)
        idx = np.clip(idx, a_min=0, a_max=len(self.valid_loss_values) - 1)
        # 映射到合法值
        loss_rate = self.valid_loss_values_np[idx]
        return loss_rate

    def process_sequence(
        self, delay_norm: np.ndarray, loss_norm: np.ndarray, delay_scaler: object
    ) -> Tuple[np.ndarray, np.ndarray]:
        """处理完整序列，确保物理合理性
        Args:
            delay_norm: 归一化的延迟序列
            loss_norm: 归一化的丢包率序列
            delay_scaler: 用于延迟逆变换的scaler对象
        Returns:
            处理后的延迟序列和丢包率序列
        """
        # 处理延迟
        delay = self.process_delay(delay_norm, delay_scaler)
        # 处理丢包率
        loss_rate = self.process_loss_rate(loss_norm)
        return delay, loss_rate

    def inject_constraints(self, x: torch.Tensor) -> torch.Tensor:
        """在生成过程中注入约束
        Args:
            x: 生成的序列，shape (batch_size, seq_len, 2)，其中第二维为[delay, loss_rate]
        Returns:
            注入约束后的序列
        """
        # 延迟约束：确保非负
        x[:, :, 0] = torch.clip(x[:, :, 0], min=0.0)
        # 丢包率约束：确保在[-1, 1]范围内（归一化后）
        x[:, :, 1] = torch.clip(x[:, :, 1], min=-1.0, max=1.0)
        return x

    def calculate_constraint_violations(
        self, delay: np.ndarray, loss_rate: np.ndarray
    ) -> Dict[str, float]:
        """计算约束违反程度
        Args:
            delay: 延迟序列
            loss_rate: 丢包率序列
        Returns:
            各项约束违反程度
        """
        # 延迟违反：负数延迟的绝对值之和
        delay_violation = np.sum(np.abs(np.minimum(delay, 0)))

        # 丢包率范围违反：超出[0, 1]范围的绝对值之和
        loss_range_violation = np.sum(np.abs(np.minimum(loss_rate, 0))) + np.sum(
            np.abs(np.maximum(loss_rate - 1, 0))
        )

        # 丢包率值违反：与最近合法值的距离之和
        loss_value_violation = 0.0
        for lr in loss_rate:
            min_dist = np.min(np.abs(self.valid_loss_values_np - lr))
            loss_value_violation += min_dist

        return {
            "delay_violation": delay_violation,
            "loss_range_violation": loss_range_violation,
            "loss_value_violation": loss_value_violation,
            "total_violation": delay_violation
            + loss_range_violation
            + loss_value_violation,
        }

    def generate_valid_sequence(
        self, generator_func, *args, max_attempts: int = 10, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成合法序列
        Args:
            generator_func: 生成函数
            *args: 生成函数的位置参数
            max_attempts: 最大尝试次数
            **kwargs: 生成函数的关键字参数
        Returns:
            合法的延迟序列和丢包率序列
        """
        for attempt in range(max_attempts):
            # 生成序列
            delay, loss_rate = generator_func(*args, **kwargs)
            # 验证序列
            validation = self.validate_sequence(delay, loss_rate)
            if validation["all_valid"]:
                return delay, loss_rate
            # 否则进行后处理
            delay = np.clip(delay, a_min=0, a_max=None)
            # 修复丢包率
            for i, lr in enumerate(loss_rate):
                min_idx = np.argmin(np.abs(self.valid_loss_values_np - lr))
                loss_rate[i] = self.valid_loss_values_np[min_idx]
            # 再次验证
            validation = self.validate_sequence(delay, loss_rate)
            if validation["all_valid"]:
                return delay, loss_rate

        # 如果多次尝试后仍不合法，返回最后一次的结果
        return delay, loss_rate
