#!/usr/bin/env python3
"""
物理约束注入机制
负责网络物理约束的验证和注入
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from network_simulation.utils.logger import get_logger

# 初始化日志记录器
logger = get_logger(__name__)


class ConstraintInjector:
    """物理约束注入器"""

    def __init__(self, valid_loss_values_up: List[float], valid_loss_values_down: List[float] = None):
        # 上下行独立的合法丢包值
        self.valid_loss_values_up = valid_loss_values_up
        self.valid_loss_values_down = valid_loss_values_down or valid_loss_values_up

        # 转换为numpy数组，提高计算效率
        self.valid_loss_values_up_np = np.array(valid_loss_values_up)
        self.valid_loss_values_down_np = np.array(valid_loss_values_down)

        # 兼容旧版本，保留valid_loss_values属性
        self.valid_loss_values = valid_loss_values_up
        self.valid_loss_values_np = self.valid_loss_values_up_np

        logger.info(f"初始化ConstraintInjector，上行合法丢包率值: {valid_loss_values_up}, 下行合法丢包率值: {valid_loss_values_down}")

    def validate_delay(self, delay: np.ndarray) -> bool:
        """验证延迟的物理合理性
        Args:
            delay: 延迟序列
        Returns:
            是否合法
        """
        return np.all(delay >= 0)

    def validate_loss_rate(self, loss_rate: np.ndarray, direction: str = 'up') -> bool:
        """验证丢包率的物理合理性
        Args:
            loss_rate: 丢包率序列
            direction: 数据方向，'up' 表示上行，'down' 表示下行
        Returns:
            是否合法
        """
        # 检查是否在0-1范围内
        range_valid = np.all(loss_rate >= 0) and np.all(loss_rate <= 1)

        # 根据方向选择合法丢包值
        valid_loss_values = self.valid_loss_values_up_np if direction == 'up' else self.valid_loss_values_down_np

        # 检查是否为合法值
        value_valid = np.all(np.isin(loss_rate, valid_loss_values))
        return range_valid and value_valid

    def validate_sequence(
        self, delay: np.ndarray, loss_rate: np.ndarray, direction: str = 'up'
    ) -> Dict[str, bool]:
        """验证完整序列的物理合理性
        Args:
            delay: 延迟序列
            loss_rate: 丢包率序列
            direction: 数据方向，'up' 表示上行，'down' 表示下行
        Returns:
            各项验证结果
        """
        return {
            "delay_valid": self.validate_delay(delay),
            "loss_rate_valid": self.validate_loss_rate(loss_rate, direction),
            "all_valid": self.validate_delay(delay)
            and self.validate_loss_rate(loss_rate, direction),
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

    def process_loss_rate(self, loss_norm: np.ndarray, direction: str = 'up') -> np.ndarray:
        """处理丢包率序列，确保物理合理性
        Args:
            loss_norm: 归一化的丢包率序列（范围[-1, 1]）
            direction: 数据方向，'up' 表示上行，'down' 表示下行
        Returns:
            处理后的丢包率序列，确保为合法值
        """
        # 根据方向选择合法丢包值
        valid_loss_values = self.valid_loss_values_up_np if direction == 'up' else self.valid_loss_values_down_np
        valid_loss_count = len(valid_loss_values)

        # 将[-1, 1]映射到[0, valid_loss_count-1]
        # 这是一个逆min-max映射：(x + 1)/2 将[-1,1]映射到[0,1]，再乘以(N-1)映射到索引范围
        idx_float = (loss_norm + 1) / 2 * (valid_loss_count - 1)

        # 四舍五入并裁剪到有效范围
        idx = np.round(idx_float).astype(int)
        idx = np.clip(idx, a_min=0, a_max=valid_loss_count - 1)

        # 映射到合法值
        loss_rate = valid_loss_values[idx]
        return loss_rate

    def process_sequence(
        self, delay_norm: np.ndarray, loss_norm: np.ndarray, delay_scaler: object, direction: str = 'up'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """处理完整序列，确保物理合理性
        Args:
            delay_norm: 归一化的延迟序列
            loss_norm: 归一化的丢包率序列
            delay_scaler: 用于逆变换的scaler对象
            direction: 数据方向，'up' 表示上行，'down' 表示下行
        Returns:
            处理后的延迟序列和丢包率序列
        """
        # 处理延迟
        delay = self.process_delay(delay_norm, delay_scaler)
        # 处理丢包率，传递方向参数
        loss_rate = self.process_loss_rate(loss_norm, direction)
        return delay, loss_rate

    def inject_constraints(self, x: torch.Tensor) -> torch.Tensor:
        """在生成过程中注入约束
        Args:
            x: 生成的序列，shape (batch_size, seq_len, input_dim)
        Returns:
            注入约束后的序列
        """
        # 输入维度检查
        if x.dim() != 3:
            raise ValueError(f"输入张量必须是3维的，当前形状: {x.shape}")

        # 输入维度：对于4D模型，input_dim应该包含延迟和丢包率
        # 假设特征顺序为 [delay1, loss1, delay2, loss2, ...]

        # 延迟约束：确保所有延迟维度（索引0和2）非负
        # 注意：这里假设前4个维度是 [delay1, loss1, delay2, loss2]
        if x.shape[2] >= 1:  # 确保至少有一个维度
            # 延迟1约束：确保非负
            x[:, :, 0] = torch.clip(x[:, :, 0], min=0.0)

            # 丢包率1约束：确保在[-1, 1]范围内（归一化后）
            if x.shape[2] >= 2:
                x[:, :, 1] = torch.clip(x[:, :, 1], min=-1.0, max=1.0)

                # 延迟2约束：确保非负
                if x.shape[2] >= 3:
                    x[:, :, 2] = torch.clip(x[:, :, 2], min=0.0)

                    # 丢包率2约束：确保在[-1, 1]范围内（归一化后）
                    if x.shape[2] >= 4:
                        x[:, :, 3] = torch.clip(x[:, :, 3], min=-1.0, max=1.0)

        return x

    def calculate_constraint_violations(
        self, delay: np.ndarray, loss_rate: np.ndarray, direction: str = 'up'
    ) -> Dict[str, float]:
        """计算约束违反程度
        Args:
            delay: 延迟序列
            loss_rate: 丢包率序列
            direction: 数据方向，'up' 表示上行，'down' 表示下行
        Returns:
            各项约束违反程度
        """
        # 延迟违反：负数延迟的绝对值之和
        delay_violation = np.sum(np.abs(np.minimum(delay, 0)))

        # 丢包率范围违反：超出[0, 1]范围的绝对值之和
        loss_range_violation = np.sum(np.abs(np.minimum(loss_rate, 0))) + np.sum(
            np.abs(np.maximum(loss_rate - 1, 0))
        )

        # 根据方向选择合法丢包值
        valid_loss_values = self.valid_loss_values_up_np if direction == 'up' else self.valid_loss_values_down_np

        # 丢包率值违反：与最近合法值的距离之和
        loss_value_violation = 0.0
        for lr in loss_rate:
            min_dist = np.min(np.abs(valid_loss_values - lr))
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
        self, generator_func, *args, max_attempts: int = 10, direction: str = 'up', **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成合法序列
        Args:
            generator_func: 生成函数
            *args: 生成函数的位置参数
            max_attempts: 最大尝试次数
            direction: 数据方向，'up' 表示上行，'down' 表示下行
            **kwargs: 生成函数的关键字参数
        Returns:
            合法的延迟序列和丢包率序列
        """
        logger.info(f"开始生成合法序列，方向: {direction}, 最大尝试次数: {max_attempts}")

        # 根据方向选择合法丢包值
        valid_loss_values = self.valid_loss_values_up_np if direction == 'up' else self.valid_loss_values_down_np

        for attempt in range(max_attempts):
            # 生成序列
            delay, loss_rate = generator_func(*args, **kwargs)
            # 验证序列
            validation = self.validate_sequence(delay, loss_rate, direction)

            if validation["all_valid"]:
                logger.info(f"尝试 {attempt+1}/{max_attempts} 成功生成合法序列")
                return delay, loss_rate

            # 否则进行后处理
            logger.debug(f"尝试 {attempt+1}/{max_attempts} 生成的序列不合法，进行后处理")
            delay = np.clip(delay, a_min=0, a_max=None)

            # 修复丢包率，使用当前方向对应的合法值
            for i, lr in enumerate(loss_rate):
                min_idx = np.argmin(np.abs(valid_loss_values - lr))
                loss_rate[i] = valid_loss_values[min_idx]

            # 再次验证
            validation = self.validate_sequence(delay, loss_rate, direction)
            if validation["all_valid"]:
                logger.info(f"尝试 {attempt+1}/{max_attempts} 后处理成功生成合法序列")
                return delay, loss_rate

        logger.warning(f"已达到最大尝试次数 {max_attempts}，返回最后一次生成的序列")
        return delay, loss_rate
