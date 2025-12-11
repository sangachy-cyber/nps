#!/usr/bin/env python3
"""
丢包率相关工具函数模块
提供丢包率合法值提取、验证等功能
"""

import numpy as np
import pandas as pd
from .logger import get_logger

logger = get_logger(__name__)


def extract_valid_loss_values(df: pd.DataFrame, tolerance: float = 1e-3) -> list:
    """从数据集提取合法丢包值，支持双通道和单通道数据

    Args:
        df (pd.DataFrame): 包含网络数据的DataFrame，必须包含以下列之一：
            - 双通道：loss_rate1（上行丢包率）和 loss_rate2（下行丢包率）
            - 单通道：loss_rate
        tolerance (float, optional): 丢包率值匹配的容差，用于合并相似值，默认为1e-3

    Returns:
        list[float]: 排序后的唯一合法丢包值列表，范围为0-1

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> # 创建双通道数据
        >>> df = pd.DataFrame({
        ...     'loss_rate1': [0.0, 0.01, 0.05, 0.01, 0.1],
        ...     'loss_rate2': [0.0, 0.02, 0.05, 0.02, 0.2]
        ... })
        >>> valid_loss_values = extract_valid_loss_values(df)
        >>> print(valid_loss_values)
        [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
    """
    # 处理双通道数据
    if all(col in df.columns for col in ["loss_rate1", "loss_rate2"]):
        # 合并上下行丢包率数据
        loss_rates = np.concatenate([df["loss_rate1"].values, df["loss_rate2"].values])
    elif "loss_rate" in df.columns:
        # 处理单通道数据
        loss_rates = df["loss_rate"].values
    else:
        # 没有找到丢包率列，返回空列表
        logger.warning(
            "未找到丢包率列（loss_rate1, loss_rate2 或 loss_rate），返回空列表"
        )
        return []

    unique_values = np.unique(loss_rates)

    # 如果有NaN值，则移除
    unique_values = unique_values[~np.isnan(unique_values)]

    # 应用容差匹配合并相似值
    valid_values = []

    for val in sorted(unique_values):
        # Check if this value is close to any already in valid_values
        if not valid_values or all(
            np.abs(val - existing) > tolerance for existing in valid_values
        ):
            valid_values.append(val)

    return valid_values


def extract_directional_valid_loss_values(
    df: pd.DataFrame, tolerance: float = 1e-3
) -> tuple:
    """从数据集提取上下行独立的合法丢包值

    Args:
        df (pd.DataFrame): 包含网络数据的DataFrame，必须包含以下列之一：
            - 双通道：loss_rate1（上行丢包率）和 loss_rate2（下行丢包率）
            - 单通道：loss_rate
        tolerance (float, optional): 丢包率值匹配的容差，用于合并相似值，默认为1e-3

    Returns:
        tuple[list[float], list[float], list[float]]:
            - 上行合法丢包值列表：排序后的唯一合法上行丢包值
            - 下行合法丢包值列表：排序后的唯一合法下行丢包值
            - 合并的合法丢包值列表：排序后的唯一合法丢包值（上下行合并）

    Examples:
        >>> import pandas as pd
        >>> # 创建双通道数据
        >>> df = pd.DataFrame({
        ...     'loss_rate1': [0.0, 0.01, 0.05, 0.01, 0.1],
        ...     'loss_rate2': [0.0, 0.02, 0.05, 0.02, 0.2]
        ... })
        >>> up_loss, down_loss, merged_loss = extract_directional_valid_loss_values(df)
        >>> print("上行丢包值:", up_loss)
        上行丢包值: [0.0, 0.01, 0.05, 0.1]
        >>> print("下行丢包值:", down_loss)
        下行丢包值: [0.0, 0.02, 0.05, 0.2]
        >>> print("合并丢包值:", merged_loss)
        合并丢包值: [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
    """
    valid_loss_values_up = None
    valid_loss_values_down = None
    valid_loss_values = None

    # 检查是否包含上下行丢包率数据
    has_up_down_loss = all(col in df.columns for col in ["loss_rate1", "loss_rate2"])

    if has_up_down_loss:
        # 分离上下行丢包率数据
        loss_rates_up = df["loss_rate1"].values
        loss_rates_down = df["loss_rate2"].values

        # 推导上下行独立的合法丢包值
        valid_loss_values_up = extract_valid_loss_values(
            pd.DataFrame({"loss_rate": loss_rates_up}), tolerance
        )
        valid_loss_values_down = extract_valid_loss_values(
            pd.DataFrame({"loss_rate": loss_rates_down}), tolerance
        )

        # 合并的合法丢包值用于兼容模型训练
        loss_rates = np.concatenate([loss_rates_up, loss_rates_down])
        valid_loss_values = extract_valid_loss_values(
            pd.DataFrame({"loss_rate": loss_rates}), tolerance
        )
    else:
        logger.warning("原始数据中未找到丢包率列，无法自动推导合法丢包值")
        valid_loss_values = None
        valid_loss_values_up = None
        valid_loss_values_down = None

    return valid_loss_values_up, valid_loss_values_down, valid_loss_values


def calculate_max_consecutive_true(arr: np.ndarray) -> int:
    """计算布尔数组中连续True值的最大长度

    Args:
        arr (np.ndarray): 布尔数组，用于表示某种状态的连续情况

    Returns:
        int: 连续True值的最大长度

    Examples:
        >>> import numpy as np
        >>> # 示例1：包含连续True值的数组
        >>> arr1 = np.array([True, True, False, True, True, True, False])
        >>> print(calculate_max_consecutive_true(arr1))
        3

        >>> # 示例2：空数组
        >>> arr2 = np.array([])
        >>> print(calculate_max_consecutive_true(arr2))
        0

        >>> # 示例3：没有True值的数组
        >>> arr3 = np.array([False, False, False])
        >>> print(calculate_max_consecutive_true(arr3))
        0
    """
    if len(arr) == 0:
        return 0

    # 使用itertools.groupby简化实现，更高效
    from itertools import groupby

    consecutive_runs = [sum(1 for _ in group) for key, group in groupby(arr) if key]
    return max(consecutive_runs) if consecutive_runs else 0
