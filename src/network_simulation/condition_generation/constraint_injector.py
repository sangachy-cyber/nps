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
    """物理约束注入器

    该类负责验证和注入网络模拟数据的物理约束，确保生成的数据符合现实网络特性，
    支持上下行独立配置和验证。
    """

    def __init__(
        self,
        valid_loss_values_up: List[float] = None,
        valid_loss_values_down: List[float] = None,
        normalizer: object = None,
    ):
        """初始化物理约束注入器

        Args:
            valid_loss_values_up (List[float], optional): 上行合法丢包值列表，默认值为 [0.0, 1/3, 0.5, 2/3, 1.0]
            valid_loss_values_down (List[float], optional): 下行合法丢包值列表，默认值为上行丢包值列表
            normalizer (object, optional): 归一化器对象，用于获取归一化参数和合法丢包值

        Examples:
            >>> from network_simulation.condition_generation.constraint_injector import ConstraintInjector
            >>> # 方式1：直接指定合法丢包值
            >>> injector = ConstraintInjector(
            ...     valid_loss_values_up=[0.0, 0.1, 0.5, 1.0],
            ...     valid_loss_values_down=[0.0, 0.2, 0.7, 1.0]
            ... )
            >>> # 方式2：使用归一化器
            >>> from network_simulation.condition_generation.normalization import Normalizer
            >>> normalizer = Normalizer([0.0, 0.1, 0.5, 1.0])
            >>> injector = ConstraintInjector(normalizer=normalizer)
        """
        # 初始化normalizer和归一化参数
        self.normalizer = normalizer
        self.normalization_params = {}

        if normalizer is not None:
            # 使用提供的normalizer的归一化参数
            self.normalization_params = normalizer.normalization_params
            self.valid_loss_values_up = normalizer.valid_loss_up
            self.valid_loss_values_down = normalizer.valid_loss_down
        else:
            # 上下行独立的合法丢包值
            self.valid_loss_values_up = valid_loss_values_up or [
                0.0,
                1 / 3,
                0.5,
                2 / 3,
                1.0,
            ]
            self.valid_loss_values_down = (
                valid_loss_values_down or self.valid_loss_values_up
            )

        # 转换为numpy数组，提高计算效率
        self.valid_loss_values_up_np = np.array(self.valid_loss_values_up)
        self.valid_loss_values_down_np = np.array(self.valid_loss_values_down)

        # 兼容旧版本，保留valid_loss_values属性
        self.valid_loss_values = self.valid_loss_values_up
        self.valid_loss_values_np = self.valid_loss_values_up_np

        logger.info(
            f"初始化ConstraintInjector，上行合法丢包值: {self.valid_loss_values_up}, 下行合法丢包值: {self.valid_loss_values_down}, normalizer: {normalizer is not None}"
        )

    def validate_delay(self, delay: np.ndarray) -> bool:
        """验证延迟的物理合理性

        Args:
            delay (np.ndarray): 延迟序列，单位为毫秒

        Returns:
            bool: 延迟是否合法，所有延迟值非负则返回True，否则返回False

        Examples:
            >>> from network_simulation.condition_generation.constraint_injector import ConstraintInjector
            >>> import numpy as np
            >>> injector = ConstraintInjector()
            >>> # 合法延迟
            >>> delay_valid = np.array([10, 20, 30, 40, 50])
            >>> print(f"合法延迟验证结果: {injector.validate_delay(delay_valid)}")  # 是
            >>> # 非法延迟（包含负值）
            >>> delay_invalid = np.array([10, -5, 30, 40, -10])
            >>> print(f"非法延迟验证结果: {injector.validate_delay(delay_invalid)}")  # 否
        """
        return np.all(delay >= 0)

    def validate_loss_rate(self, loss_rate: np.ndarray, direction: str = "up") -> bool:
        """验证丢包率的物理合理性

        Args:
            loss_rate (np.ndarray): 丢包率序列，范围0-1
            direction (str, optional): 数据方向，'up' 表示上行，'down' 表示下行，默认值为'up'

        Returns:
            bool: 丢包率是否合法
                - 所有丢包值必须在0-1范围内
                - 所有丢包值必须是指定方向的合法丢包值之一

        Examples:
            >>> from network_simulation.condition_generation.constraint_injector import ConstraintInjector
            >>> import numpy as np
            >>> injector = ConstraintInjector(
            ...     valid_loss_values_up=[0.0, 0.5, 1.0],
            ...     valid_loss_values_down=[0.0, 0.1, 0.2, 1.0]
            ... )
            >>> # 合法上行丢包率
            >>> loss_up_valid = np.array([0.0, 0.5, 1.0, 0.5])
            >>> print(f"合法上行丢包率验证结果: {injector.validate_loss_rate(loss_up_valid, 'up')}")  # 是
            >>> # 非法上行丢包率（包含0.3，不在合法列表中）
            >>> loss_up_invalid = np.array([0.0, 0.3, 1.0])
            >>> print(f"非法上行丢包率验证结果: {injector.validate_loss_rate(loss_up_invalid, 'up')}")  # 否
            >>> # 合法下行丢包率
            >>> loss_down_valid = np.array([0.0, 0.1, 0.2, 1.0])
            >>> print(f"合法下行丢包率验证结果: {injector.validate_loss_rate(loss_down_valid, 'down')}")  # 是
        """
        # 检查是否在0-1范围内
        range_valid = np.all(loss_rate >= 0) and np.all(loss_rate <= 1)

        # 根据方向选择合法丢包值
        valid_loss_values = (
            self.valid_loss_values_up_np
            if direction == "up"
            else self.valid_loss_values_down_np
        )

        # 检查是否为合法值
        value_valid = np.all(np.isin(loss_rate, valid_loss_values))
        return range_valid and value_valid

    def validate_sequence(
        self, delay: np.ndarray, loss_rate: np.ndarray, direction: str = "up"
    ) -> Dict[str, bool]:
        """验证完整序列的物理合理性

        Args:
            delay (np.ndarray): 延迟序列，单位为毫秒
            loss_rate (np.ndarray): 丢包率序列，范围0-1
            direction (str, optional): 数据方向，'up' 表示上行，'down' 表示下行，默认值为'up'

        Returns:
            Dict[str, bool]: 包含各项验证结果的字典
                - delay_valid: 延迟是否合法
                - loss_rate_valid: 丢包率是否合法
                - all_valid: 整个序列是否完全合法

        Examples:
            >>> from network_simulation.condition_generation.constraint_injector import ConstraintInjector
            >>> import numpy as np
            >>> injector = ConstraintInjector(
            ...     valid_loss_values_up=[0.0, 0.5, 1.0]
            ... )
            >>> # 合法序列
            >>> delay_valid = np.array([10, 20, 30])
            >>> loss_valid = np.array([0.0, 0.5, 1.0])
            >>> result = injector.validate_sequence(delay_valid, loss_valid)
            >>> print(f"合法序列验证结果: {result}")  # {'delay_valid': True, 'loss_rate_valid': True, 'all_valid': True}
            >>> # 非法序列（包含负延迟）
            >>> delay_invalid = np.array([10, -5, 30])
            >>> result = injector.validate_sequence(delay_invalid, loss_valid)
            >>> print(f"非法延迟序列验证结果: {result}")  # {'delay_valid': False, 'loss_rate_valid': True, 'all_valid': False}
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
            delay_norm (np.ndarray): 归一化的延迟序列，范围取决于归一化方法
            scaler (object): 用于逆变换的scaler对象，需实现inverse_transform方法

        Returns:
            np.ndarray: 处理后的延迟序列，所有值均为非负

        Examples:
            >>> from network_simulation.condition_generation.constraint_injector import ConstraintInjector
            >>> from sklearn.preprocessing import MinMaxScaler
            >>> import numpy as np
            >>> injector = ConstraintInjector()
            >>> scaler = MinMaxScaler(feature_range=(0, 1))
            >>> # 假设已拟合scaler
            >>> delay_norm = np.array([0.1, 0.5, 0.8, 0.3])
            >>> delay = injector.process_delay(delay_norm, scaler)
            >>> print(f"处理后的延迟: {delay}")
        """
        # 逆变换
        delay = scaler.inverse_transform(delay_norm.reshape(-1, 1)).flatten()
        # 确保非负
        delay = np.clip(delay, a_min=0, a_max=None)
        return delay

    def process_loss_rate(
        self, loss_norm: np.ndarray, direction: str = "up"
    ) -> np.ndarray:
        """处理丢包率序列，确保物理合理性

        Args:
            loss_norm (np.ndarray): 归一化的丢包率序列，范围[-1, 1]
            direction (str, optional): 数据方向，'up' 表示上行，'down' 表示下行，默认值为'up'

        Returns:
            np.ndarray: 处理后的丢包率序列，确保所有值均为指定方向的合法丢包值

        Examples:
            >>> from network_simulation.condition_generation.constraint_injector import ConstraintInjector
            >>> import numpy as np
            >>> injector = ConstraintInjector(
            ...     valid_loss_values_up=[0.0, 0.5, 1.0]
            ... )
            >>> # 归一化的丢包率序列
            >>> loss_norm = np.array([-1.0, 0.0, 0.5, 1.0])
            >>> # 处理为合法丢包值
            >>> loss_rate = injector.process_loss_rate(loss_norm, 'up')
            >>> print(f"处理后的丢包率: {loss_rate}")  # [0.0, 0.5, 0.5, 1.0]
        """
        if self.normalizer is not None:
            # 复用Normalizer的反归一化逻辑，确保一致性
            loss_params_key = f"loss_rate_{direction}"
            if loss_params_key in self.normalization_params:
                loss_params = self.normalization_params[loss_params_key]
                return self.normalizer.denormalize_loss_rate(loss_norm, loss_params)
            else:
                logger.warning(
                    f"归一化参数中缺少{loss_params_key}，使用默认反归一化逻辑"
                )

        # 回退到默认逻辑
        # 根据方向选择合法丢包值
        valid_loss_values = (
            self.valid_loss_values_up_np
            if direction == "up"
            else self.valid_loss_values_down_np
        )
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
        self,
        delay_norm: np.ndarray,
        loss_norm: np.ndarray,
        delay_scaler: object,
        direction: str = "up",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """处理完整序列，确保物理合理性

        Args:
            delay_norm (np.ndarray): 归一化的延迟序列，范围取决于归一化方法
            loss_norm (np.ndarray): 归一化的丢包率序列，范围[-1, 1]
            delay_scaler (object): 用于逆变换的scaler对象，需实现inverse_transform方法
            direction (str, optional): 数据方向，'up' 表示上行，'down' 表示下行，默认值为'up'

        Returns:
            Tuple[np.ndarray, np.ndarray]: 处理后的延迟序列和丢包率序列
                - delay: 处理后的延迟序列，单位为毫秒，所有值非负
                - loss_rate: 处理后的丢包率序列，范围0-1，所有值均为合法丢包值

        Examples:
            >>> from network_simulation.condition_generation.constraint_injector import ConstraintInjector
            >>> from sklearn.preprocessing import MinMaxScaler
            >>> import numpy as np
            >>> injector = ConstraintInjector(
            ...     valid_loss_values_up=[0.0, 0.5, 1.0]
            ... )
            >>> scaler = MinMaxScaler(feature_range=(0, 1))
            >>> # 归一化的序列
            >>> delay_norm = np.array([0.1, 0.5, 0.8, 0.3])
            >>> loss_norm = np.array([-1.0, 0.0, 0.5, 1.0])
            >>> # 处理为合法序列
            >>> delay, loss_rate = injector.process_sequence(delay_norm, loss_norm, scaler, 'up')
            >>> print(f"处理后的延迟: {delay}")
            >>> print(f"处理后的丢包率: {loss_rate}")
        """
        # 处理延迟
        delay = self.process_delay(delay_norm, delay_scaler)
        # 处理丢包率，传递方向参数
        loss_rate = self.process_loss_rate(loss_norm, direction)
        return delay, loss_rate

    def inject_constraints(self, x: torch.Tensor) -> torch.Tensor:
        """在生成过程中注入物理约束

        该方法在生成过程中实时注入约束，确保生成的序列符合物理规则，
        支持上下行独立约束注入。

        Args:
            x (torch.Tensor): 生成的序列，shape (batch_size, seq_len, input_dim)
                - batch_size: 批次大小
                - seq_len: 序列长度
                - input_dim: 输入维度，假设特征顺序为 [delay1, loss1, delay2, loss2, ...]

        Returns:
            torch.Tensor: 注入约束后的序列，所有值均符合物理规则

        Examples:
            >>> from network_simulation.condition_generation.constraint_injector import ConstraintInjector
            >>> import torch
            >>> injector = ConstraintInjector()
            >>> # 生成一个随机序列
            >>> x = torch.randn(2, 100, 4)  # shape: (batch_size=2, seq_len=100, input_dim=4)
            >>> # 注入约束
            >>> x_constrained = injector.inject_constraints(x)
            >>> # 验证约束是否生效
            >>> print(f"延迟1最小值: {x_constrained[:, :, 0].min().item()}")  # 应该 >= 0
            >>> print(f"丢包率1最小值: {x_constrained[:, :, 1].min().item()}")  # 应该 >= -1
            >>> print(f"丢包率1最大值: {x_constrained[:, :, 1].max().item()}")  # 应该 <= 1
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
        self, delay: np.ndarray, loss_rate: np.ndarray, direction: str = "up"
    ) -> Dict[str, float]:
        """计算约束违反程度

        该方法计算序列违反物理约束的程度，用于评估生成序列的质量。

        Args:
            delay (np.ndarray): 延迟序列，单位为毫秒
            loss_rate (np.ndarray): 丢包率序列，范围0-1
            direction (str, optional): 数据方向，'up' 表示上行，'down' 表示下行，默认值为'up'

        Returns:
            Dict[str, float]: 包含各项约束违反程度的字典
                - delay_violation: 延迟违反程度，负数延迟的绝对值之和
                - loss_range_violation: 丢包率范围违反程度，超出[0, 1]范围的绝对值之和
                - loss_value_violation: 丢包率值违反程度，与最近合法值的距离之和
                - total_violation: 总违反程度，各项违反程度之和

        Examples:
            >>> from network_simulation.condition_generation.constraint_injector import ConstraintInjector
            >>> import numpy as np
            >>> injector = ConstraintInjector(
            ...     valid_loss_values_up=[0.0, 0.5, 1.0]
            ... )
            >>> # 生成一个违反约束的序列
            >>> delay = np.array([10, -5, 30, 40])  # 包含负值延迟
            >>> loss_rate = np.array([0.0, 0.3, 1.5, 0.5])  # 包含非法丢包值和超出范围的值
            >>> # 计算约束违反程度
            >>> violations = injector.calculate_constraint_violations(delay, loss_rate, 'up')
            >>> print(f"约束违反程度: {violations}")
        """
        # 延迟违反：负数延迟的绝对值之和
        delay_violation = np.sum(np.abs(np.minimum(delay, 0)))

        # 丢包率范围违反：超出[0, 1]范围的绝对值之和
        loss_range_violation = np.sum(np.abs(np.minimum(loss_rate, 0))) + np.sum(
            np.abs(np.maximum(loss_rate - 1, 0))
        )

        # 根据方向选择合法丢包值
        valid_loss_values = (
            self.valid_loss_values_up_np
            if direction == "up"
            else self.valid_loss_values_down_np
        )

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
        self,
        generator_func,
        *args,
        max_attempts: int = 10,
        direction: str = "up",
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成合法的网络模拟序列

        该方法使用指定的生成函数尝试生成合法序列，如果生成的序列不合法，
        会进行后处理，确保最终返回的序列符合物理约束。

        Args:
            generator_func (callable): 生成函数，需返回(delay, loss_rate)元组
                - 预期签名: def generator_func(*args, **kwargs) -> Tuple[np.ndarray, np.ndarray]
            *args: 传递给生成函数的位置参数
            max_attempts (int, optional): 最大尝试次数，默认值为10
            direction (str, optional): 数据方向，'up' 表示上行，'down' 表示下行，默认值为'up'
            **kwargs: 传递给生成函数的关键字参数

        Returns:
            Tuple[np.ndarray, np.ndarray]: 合法的延迟和丢包率序列
                - delay: 延迟序列，单位为毫秒，所有值非负
                - loss_rate: 丢包率序列，范围0-1，所有值均为合法丢包值

        Examples:
            >>> from network_simulation.condition_generation.constraint_injector import ConstraintInjector
            >>> import numpy as np
            >>> injector = ConstraintInjector(
            ...     valid_loss_values_up=[0.0, 0.5, 1.0]
            ... )
            >>> # 定义一个简单的生成函数
            >>> def simple_generator():
            ...     delay = np.random.randn(100) * 50 + 100  # 可能包含负值
            ...     loss_rate = np.random.rand(100)  # 可能包含非法丢包值
            ...     return delay, loss_rate
            >>> # 生成合法序列
            >>> delay, loss_rate = injector.generate_valid_sequence(simple_generator, max_attempts=5, direction='up')
            >>> print(f"生成的延迟最小值: {delay.min()}")  # 应该 >= 0
            >>> print(f"生成的丢包率值: {np.unique(loss_rate)}")  # 应该只有合法值
        """
        logger.info(
            f"开始生成合法序列，方向: {direction}, 最大尝试次数: {max_attempts}"
        )

        # 根据方向选择合法丢包值
        valid_loss_values = (
            self.valid_loss_values_up_np
            if direction == "up"
            else self.valid_loss_values_down_np
        )

        for attempt in range(max_attempts):
            # 生成序列
            delay, loss_rate = generator_func(*args, **kwargs)
            # 验证序列
            validation = self.validate_sequence(delay, loss_rate, direction)

            if validation["all_valid"]:
                logger.info(f"尝试 {attempt+1}/{max_attempts} 成功生成合法序列")
                return delay, loss_rate

            # 否则进行后处理
            logger.debug(
                f"尝试 {attempt+1}/{max_attempts} 生成的序列不合法，进行后处理"
            )
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
