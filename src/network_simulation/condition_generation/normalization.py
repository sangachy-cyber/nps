#!/usr/bin/env python3
"""
归一化处理模块
提供统一的归一化和反归一化功能
"""

import numpy as np
from sklearn.preprocessing import RobustScaler
from network_simulation.utils.logger import get_logger

# 获取日志记录器
logger = get_logger(__name__)


def quantile_normalize(data: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    分位数归一化
    将数据转换为均匀分布，映射到[0, 1]范围

    Args:
        data: 原始数据，shape (n_samples,)，一维数组

    Returns:
        tuple: 包含两个元素的元组
            - 归一化后的数据，shape (n_samples,)
            - 归一化参数字典，包含sorted_data和quantiles
    """
    # 确保数据是一维的
    data = data.flatten()
    n_samples = len(data)

    # 直接计算原始数据的分位数，不进行复杂的预处理
    # 这样可以确保归一化和反归一化的映射关系对称
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]

    # 计算每个数据点的分位数（使用排名除以样本数+1，避免极端值）
    # 使用(n_samples+1)分母可以避免最大值映射到1.0
    quantiles = np.arange(1, n_samples + 1) / (n_samples + 1)

    # 创建分位数映射
    quantile_map = np.zeros(n_samples)
    for i, idx in enumerate(sorted_indices):
        quantile_map[idx] = quantiles[i]

    # 保存归一化参数
    params = {
        'sorted_data': sorted_data,
        'quantiles': quantiles
    }

    return quantile_map, params


def quantile_denormalize(normalized_data: np.ndarray, params: dict) -> np.ndarray:
    """
    分位数反归一化
    将均匀分布数据还原为原始分布

    Args:
        normalized_data: 归一化后的数据，shape (n_samples,)
        params: 归一化参数，包含sorted_data和quantiles

    Returns:
        np.ndarray: 反归一化后的数据，shape (n_samples,)
    """
    sorted_data = params['sorted_data']
    quantiles = params['quantiles']

    # 使用线性插值进行反归一化
    denormalized = np.interp(normalized_data, quantiles, sorted_data)

    return denormalized


def normalize_delay(delay_values: np.ndarray, method: str = 'robust', use_log_transform: bool = True) -> tuple[np.ndarray, dict]:
    """
    对延迟数据进行归一化处理

    Args:
        delay_values: 原始延迟数据，shape (n_samples,)
        method: 归一化方法，可选值为'robust'或'quantile'，默认为'robust'
        use_log_transform: 是否使用对数变换，默认为True

    Returns:
        tuple: 包含两个元素的元组
            - 归一化后的延迟数据，shape (n_samples,)
            - 归一化参数字典，包含归一化方法、变换参数等
    """
    logger.info(f"开始归一化延迟数据，输入形状: {delay_values.shape}, 方法: {method}, use_log_transform: {use_log_transform}")
    logger.info(f"原始延迟数据统计 - 最小值: {np.min(delay_values):.4f}, 最大值: {np.max(delay_values):.4f}, 平均值: {np.mean(delay_values):.4f}, 标准差: {np.std(delay_values):.4f}")

    # 保存原始数据的最小值，用于反归一化
    original_min = np.min(delay_values)

    # 复制原始数据，避免修改输入
    delay_values_copy = delay_values.copy()

    # 添加对数变换，减少数据偏斜
    if use_log_transform:
        # 加1避免log(0)
        delay_values_copy = np.log(delay_values_copy + 1)
        logger.info(f"对数变换后 - 最小值: {np.min(delay_values_copy):.4f}, 最大值: {np.max(delay_values_copy):.4f}, 平均值: {np.mean(delay_values_copy):.4f}, 标准差: {np.std(delay_values_copy):.4f}")

    if method == 'quantile':
        # 分位数归一化
        quantile_data, quantile_params = quantile_normalize(delay_values_copy)

        # 缩放到[-1, 1]范围
        delay_norm = quantile_data * 2 - 1

        logger.info(f"分位数归一化后 - 最小值: {np.min(delay_norm):.4f}, 最大值: {np.max(delay_norm):.4f}, 平均值: {np.mean(delay_norm):.4f}, 标准差: {np.std(delay_norm):.4f}")

        # 保存归一化参数
        normalization_params = {
            'normalization_method': 'quantile',
            'original_min': original_min,
            'quantile_params': quantile_params,
            'use_log_transform': use_log_transform
        }
    else:
        # RobustScaler归一化（原有实现）
        # 1. 使用对数变换后的数据进行RobustScaler归一化
        # 使用合理的分位数范围(25, 75)，只考虑中间50%的数据，避免极端值影响
        delay_scaler = RobustScaler(quantile_range=(25, 75))  # 使用标准四分位距范围
        delay_scaled = delay_scaler.fit_transform(delay_values_copy.reshape(-1, 1)).flatten()

        logger.info(f"RobustScaler处理后 - 最小值: {np.min(delay_scaled):.4f}, 最大值: {np.max(delay_scaled):.4f}, 平均值: {np.mean(delay_scaled):.4f}, 标准差: {np.std(delay_scaled):.4f}")
        logger.info(f"RobustScaler参数 - 中心值: {delay_scaler.center_}, 缩放因子: {delay_scaler.scale_}")

        # 3. 计算当前归一化后的最小值和最大值
        current_min = np.min(delay_scaled)
        current_max = np.max(delay_scaled)

        logger.info(f"RobustScaler输出范围 - 最小值: {current_min:.4f}, 最大值: {current_max:.4f}")

        # 4. 对RobustScaler输出进行优化处理，解决分布不均衡问题
        if current_max > current_min:
            # 对RobustScaler输出进行截断，去除极端值
            # 只保留[-5, 5]范围内的值，然后再缩放到[-1, 1]
            delay_scaled_clipped = np.clip(delay_scaled, a_min=-5, a_max=5)
            current_min_clipped = np.min(delay_scaled_clipped)
            current_max_clipped = np.max(delay_scaled_clipped)

            logger.info(f"截断后范围 - 最小值: {current_min_clipped:.4f}, 最大值: {current_max_clipped:.4f}")

            # 缩放到[-1, 1]范围
            if current_max_clipped > current_min_clipped:
                delay_norm = 2 * ((delay_scaled_clipped - current_min_clipped) / (current_max_clipped - current_min_clipped)) - 1
            else:
                delay_norm = np.zeros_like(delay_scaled)
        else:
            delay_norm = np.zeros_like(delay_scaled)
            # 设置默认的clipped值
            current_min_clipped = current_min
            current_max_clipped = current_max

        logger.info(f"最终归一化后 - 最小值: {np.min(delay_norm):.4f}, 最大值: {np.max(delay_norm):.4f}, 平均值: {np.mean(delay_norm):.4f}, 标准差: {np.std(delay_norm):.4f}")

        # 保存归一化参数，包括截断后的范围
        normalization_params = {
            'normalization_method': 'robust',
            'delay_scaler_center_': delay_scaler.center_,
            'delay_scaler_scale_': delay_scaler.scale_,
            'robust_scale_min': current_min,
            'robust_scale_max': current_max,
            'clipped_min': current_min_clipped,  # 保存截断后的最小值
            'clipped_max': current_max_clipped,  # 保存截断后的最大值
            'original_min': original_min,
            'use_log_transform': use_log_transform
        }

    logger.info(f"归一化参数: {normalization_params}")

    return delay_norm, normalization_params


def denormalize_delay(delay_norm: np.ndarray, normalization_params: dict) -> np.ndarray:
    """
    对延迟数据进行反归一化处理

    Args:
        delay_norm: 归一化后的延迟数据，shape (n_samples,)
        normalization_params: 归一化参数，包含归一化方法、变换参数等

    Returns:
        np.ndarray: 反归一化后的延迟数据，shape (n_samples,)
    """
    logger.info(f"开始反归一化延迟数据，输入形状: {delay_norm.shape}")
    logger.info(f"归一化延迟数据统计 - 最小值: {np.min(delay_norm):.4f}, 最大值: {np.max(delay_norm):.4f}, 平均值: {np.mean(delay_norm):.4f}, 标准差: {np.std(delay_norm):.4f}")
    logger.info(f"反归一化参数: {normalization_params}")

    # 获取原始数据最小值，用于确保延迟值非负
    original_min = normalization_params.get('original_min', 0.0)

    # 检查是否使用对数变换
    use_log_transform = normalization_params.get('use_log_transform', False)

    # 检查归一化方法
    normalization_method = normalization_params.get('normalization_method', 'robust')

    if normalization_method == 'quantile':
        # 分位数反归一化
        logger.info("使用分位数反归一化方法")

        # 将[-1, 1]转换回[0, 1]范围
        quantile_data = (delay_norm + 1) / 2

        # 执行分位数反归一化
        delay = quantile_denormalize(quantile_data, normalization_params['quantile_params'])

        logger.info(f"分位数反变换后 - 最小值: {np.min(delay):.4f}, 最大值: {np.max(delay):.4f}, 平均值: {np.mean(delay):.4f}, 标准差: {np.std(delay):.4f}")
    else:
        # RobustScaler反归一化（原有实现）
        logger.info("使用RobustScaler反归一化方法")

        # 从参数中提取值
        delay_scaler_center_ = normalization_params['delay_scaler_center_']
        delay_scaler_scale_ = normalization_params['delay_scaler_scale_']
        robust_scale_min = normalization_params['robust_scale_min']
        robust_scale_max = normalization_params['robust_scale_max']

        # 1. 先将模型输出的[-1, 1]范围转换回RobustScaler的输出范围
        # 优先使用截断后的范围（如果存在），否则使用原始范围
        if 'clipped_min' in normalization_params and 'clipped_max' in normalization_params:
            # 使用截断后的范围进行反变换
            clipped_min = normalization_params['clipped_min']
            clipped_max = normalization_params['clipped_max']

            logger.info(f"使用截断后的范围进行反变换: [{clipped_min:.4f}, {clipped_max:.4f}]")

            if clipped_max > clipped_min:
                delay_scaled = ((delay_norm + 1) / 2) * (clipped_max - clipped_min) + clipped_min
            else:
                delay_scaled = delay_norm
        else:
            # 兼容旧版本参数，使用原始范围
            logger.info(f"使用原始范围进行反变换: [{robust_scale_min:.4f}, {robust_scale_max:.4f}]")

            if robust_scale_max > robust_scale_min:
                delay_scaled = ((delay_norm + 1) / 2) * (robust_scale_max - robust_scale_min) + robust_scale_min
            else:
                delay_scaled = delay_norm

        logger.info(f"转换回RobustScaler范围后 - 最小值: {np.min(delay_scaled):.4f}, 最大值: {np.max(delay_scaled):.4f}, 平均值: {np.mean(delay_scaled):.4f}, 标准差: {np.std(delay_scaled):.4f}")

        # 2. 然后应用RobustScaler的反变换
        # RobustScaler的反变换公式：原始值 = (归一化值 * 四分位距) + 中位数
        delay = (delay_scaled * delay_scaler_scale_) + delay_scaler_center_

        logger.info(f"RobustScaler反变换后 - 最小值: {np.min(delay):.4f}, 最大值: {np.max(delay):.4f}, 平均值: {np.mean(delay):.4f}, 标准差: {np.std(delay):.4f}")

    # 应用指数变换，恢复原始值
    if use_log_transform:
        delay = np.exp(delay) - 1  # 指数变换，减1恢复原始值
        logger.info(f"指数变换后 - 最小值: {np.min(delay):.4f}, 最大值: {np.max(delay):.4f}, 平均值: {np.mean(delay):.4f}, 标准差: {np.std(delay):.4f}")

    # 确保延迟值非负，符合物理意义
    # 使用原始数据的最小值作为下限，确保生成的数据与原始数据在同一范围内
    delay = np.clip(delay, a_min=original_min, a_max=None)

    logger.info(f"最终反归一化后 - 最小值: {np.min(delay):.4f}, 最大值: {np.max(delay):.4f}, 平均值: {np.mean(delay):.4f}, 标准差: {np.std(delay):.4f}")

    return delay


def normalize_loss_rate(loss_rate_values: np.ndarray, valid_loss_values: list) -> tuple[np.ndarray, dict]:
    """
    对丢包率数据进行归一化处理

    Args:
        loss_rate_values: 原始丢包率数据，shape (n_samples,)
        valid_loss_values: 合法丢包值列表，包含所有可能的丢包率值

    Returns:
        tuple: 包含两个元素的元组
            - 归一化后的丢包率数据，shape (n_samples,)
            - 归一化参数字典，包含丢包率映射等信息
    """
    logger.info(f"开始归一化丢包率数据，输入形状: {loss_rate_values.shape}")
    logger.info(f"原始丢包率数据统计 - 最小值: {np.min(loss_rate_values):.4f}, 最大值: {np.max(loss_rate_values):.4f}, 平均值: {np.mean(loss_rate_values):.4f}, 标准差: {np.std(loss_rate_values):.4f}")
    logger.info(f"合法丢包值: {valid_loss_values}")

    # 构建loss_rate到index的映射
    loss_rate_mapping = {value: idx for idx, value in enumerate(valid_loss_values)}
    N_vals = len(valid_loss_values)

    logger.info(f"丢包率映射: {loss_rate_mapping}")

    # 处理不在映射中的丢包率值，将它们映射到最接近的合法值
    def get_loss_rate_index(lr):
        if lr in loss_rate_mapping:
            return loss_rate_mapping[lr]
        else:
            closest_value = min(valid_loss_values, key=lambda x: abs(x - lr))
            return loss_rate_mapping[closest_value]

    # 执行序数编码
    loss_rate_indices = np.array([get_loss_rate_index(lr) for lr in loss_rate_values])

    logger.info(f"丢包率序数编码后 - 唯一值: {np.unique(loss_rate_indices)}, 统计: {np.bincount(loss_rate_indices)}")

    # Min-Max缩放到[-1, 1]范围
    if N_vals > 1:
        loss_norm = (loss_rate_indices / (N_vals - 1)) * 2 - 1

        # 检查归一化后的丢包率是否有足够的变化
        unique_norm_loss = np.unique(loss_norm)
        logger.info(f"Min-Max缩放后 - 最小值: {np.min(loss_norm):.4f}, 最大值: {np.max(loss_norm):.4f}, 平均值: {np.mean(loss_norm):.4f}, 标准差: {np.std(loss_norm):.4f}")
        logger.info(f"归一化丢包率唯一值数量: {len(unique_norm_loss)}")

        if len(unique_norm_loss) == 1:
            # 添加均匀分布噪声，范围为[-0.1, 0.1]
            noise = np.random.uniform(-0.1, 0.1, size=loss_norm.shape)
            loss_norm = loss_norm + noise
            # 确保仍在[-1, 1]范围内
            loss_norm = np.clip(loss_norm, -1, 1)
            logger.info(f"添加噪声后 - 最小值: {np.min(loss_norm):.4f}, 最大值: {np.max(loss_norm):.4f}, 平均值: {np.mean(loss_norm):.4f}, 标准差: {np.std(loss_norm):.4f}")
    else:
        # 当只有一个有效丢包值时，直接设置为0
        loss_norm = np.zeros_like(loss_rate_indices)
        logger.info("只有一个有效丢包值，归一化结果为全零")

    # 保存归一化参数
    normalization_params = {
        'loss_rate_mapping': loss_rate_mapping,
        'valid_loss_values': valid_loss_values
    }

    logger.info(f"归一化参数: {normalization_params}")

    return loss_norm, normalization_params


def denormalize_loss_rate(loss_norm: np.ndarray, normalization_params: dict) -> np.ndarray:
    """
    对丢包率数据进行反归一化处理

    Args:
        loss_norm: 归一化后的丢包率数据，shape (n_samples,)
        normalization_params: 归一化参数，包含丢包率映射和合法丢包值列表

    Returns:
        np.ndarray: 反归一化后的丢包率数据，shape (n_samples,)
    """
    logger.info(f"开始反归一化丢包率数据，输入形状: {loss_norm.shape}")
    logger.info(f"归一化丢包率数据统计 - 最小值: {np.min(loss_norm):.4f}, 最大值: {np.max(loss_norm):.4f}, 平均值: {np.mean(loss_norm):.4f}, 标准差: {np.std(loss_norm):.4f}")
    logger.info(f"反归一化参数: {normalization_params}")

    # 从参数中提取值
    valid_loss_values = normalization_params['valid_loss_values']

    logger.info(f"合法丢包值: {valid_loss_values}")

    # 将[-1, 1]映射到[0, len(valid_loss_values)-1]
    idx_float = (loss_norm + 1) / 2 * (len(valid_loss_values) - 1)
    # 四舍五入并裁剪到有效范围
    idx = np.round(idx_float).astype(int)
    idx = np.clip(idx, a_min=0, a_max=len(valid_loss_values) - 1)

    logger.info(f"映射索引后 - 唯一值: {np.unique(idx)}, 统计: {np.bincount(idx)}")

    # 映射到合法值
    loss_rate = np.array(valid_loss_values)[idx]

    logger.info(f"反归一化后丢包率统计 - 最小值: {np.min(loss_rate):.4f}, 最大值: {np.max(loss_rate):.4f}, 平均值: {np.mean(loss_rate):.4f}, 标准差: {np.std(loss_rate):.4f}")

    return loss_rate
