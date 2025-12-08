#!/usr/bin/env python3
"""
步骤2.2：生成样本数据
"""

import sys
import os
import json

# 添加当前目录到Python路径，以便找到config模块
sys.path.append(os.path.abspath("."))
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
from network_simulation.utils.logger import get_logger
from config import (
    DEFAULT_WINDOW_SIZE,
    DEFAULT_STRIDE,
    PROCESSED_DIR,
    PATTERNS_DIR,
    DEFAULT_MODEL_DIR,
    GENERATED_DIR,
    DEFAULT_SAMPLE_LENGTH,
    DEFAULT_NUM_GROUPS,
    CLEANUP_OLD_FILES,
    KEEP_LATEST_FILES
)


# 获取日志记录器
logger = get_logger(__name__)

def load_valid_loss_values(input_patterns_dir: Path) -> list:
    """加载合法丢包值

    尝试从多个位置加载valid_loss_values.json文件，包括：
    1. input_patterns_dir/metadata/valid_loss_values.json
    2. input_patterns_dir.parent/features/valid_loss_values.json
    3. input_patterns_dir.parent/features/merged_valid_loss_values.json

    Args:
        input_patterns_dir: 行为模式目录路径

    Returns:
        list: 合法丢包值列表

    Raises:
        FileNotFoundError: 如果在所有位置都找不到valid_loss_values.json文件
    """
    # 尝试从多个位置加载valid_loss_values.json文件
    valid_loss_values_file = input_patterns_dir / "metadata" / "valid_loss_values.json"
    if not valid_loss_values_file.exists():
        # 尝试从features目录加载（优化流水线的情况）
        features_dir = input_patterns_dir.parent / "features"
        valid_loss_values_file = features_dir / "valid_loss_values.json"
        if not valid_loss_values_file.exists():
            # 尝试从features目录加载合并后的合法丢包值（原始流水线的情况）
            valid_loss_values_file = features_dir / "merged_valid_loss_values.json"
            if not valid_loss_values_file.exists():
                raise FileNotFoundError(f"在 {input_patterns_dir} 及其父目录中未找到 valid_loss_values.json 文件")

    with open(valid_loss_values_file, "r") as f:
        valid_loss_values = json.load(f)
    
    return valid_loss_values

def load_behavior_labels(input_patterns_dir: Path) -> np.ndarray:
    """加载行为标签

    从input_patterns_dir目录中查找并加载behavior_labels_*.json文件，
    返回行为标签数组。

    Args:
        input_patterns_dir: 行为模式目录路径

    Returns:
        np.ndarray: 行为标签数组
    """
    behavior_labels_file = next(input_patterns_dir.glob("behavior_labels_*.json"))
    with open(behavior_labels_file, "r") as f:
        behavior_labels_data = json.load(f)
    behavior_ids = np.array(behavior_labels_data["labels"])
    
    return behavior_ids

def expand_behavior_labels(
    behavior_ids: np.ndarray,
    processed_df: pd.DataFrame,
    window_size: int,
    stride: int
) -> np.ndarray:
    """扩展行为标签到样本级别

    将窗口级别的行为标签扩展到每个样本点，确保每个时间点都有对应的行为标签。

    Args:
        behavior_ids: 窗口级别的行为标签数组
        processed_df: 处理后的数据DataFrame
        window_size: 窗口大小
        stride: 窗口步长

    Returns:
        np.ndarray: 样本级别的行为标签数组
    """
    num_windows = len(behavior_ids)
    
    # 初始化行为标签数组，-2表示未分配
    expanded_behavior_ids = np.full(len(processed_df), -2, dtype=int)
    
    # 直接为每个时间窗口分配对应的行为标签
    for i in range(num_windows):
        window_start = i * stride
        window_end = window_start + window_size
        
        # 确保窗口不超出数据范围
        window_start = max(0, window_start)
        window_end = min(len(processed_df), window_end)
        
        # 为当前窗口内的所有时间点分配相同的行为标签
        expanded_behavior_ids[window_start:window_end] = behavior_ids[i]
    
    # 处理边界情况，确保所有时间点都有行为标签
    # 对于开头未分配的时间点，使用第一个窗口的标签
    if -2 in expanded_behavior_ids[:num_windows]:
        expanded_behavior_ids[:num_windows] = behavior_ids[0]
    
    # 对于末尾未分配的时间点，使用最后一个窗口的标签
    last_window_start = (num_windows - 1) * stride
    if -2 in expanded_behavior_ids[last_window_start:]:
        expanded_behavior_ids[last_window_start:] = behavior_ids[-1]
    
    return expanded_behavior_ids

def _calculate_dtw_distance(a, b) -> float:
    """计算两个序列之间的DTW距离

    Args:
        a: 第一个序列
        b: 第二个序列

    Returns:
        float: DTW距离
    """
    from scipy.spatial.distance import euclidean
    n = len(a)
    m = len(b)
    dtw = np.zeros((n+1, m+1))
    dtw[:, 0] = np.inf
    dtw[0, :] = np.inf
    dtw[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = euclidean([a[i-1]], [b[j-1]])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    return dtw[n, m]

def _collect_label_sequences(
    expanded_behavior_ids: np.ndarray,
    processed_df: pd.DataFrame,
    unique_non_noise_labels: list
) -> dict:
    """收集每个非噪声标签的样本序列

    Args:
        expanded_behavior_ids: 扩展后的行为标签数组
        processed_df: 处理后的数据DataFrame
        unique_non_noise_labels: 非噪声标签列表

    Returns:
        dict: 包含每个标签及其代表序列的字典
    """
    label_sequences = {}
    for label in unique_non_noise_labels:
        # 为每个标签收集一些样本序列用于DTW比较
        label_indices = np.where(expanded_behavior_ids == label)[0]
        if len(label_indices) > 0:
            # 取前100个样本作为该标签的代表序列
            sample_indices = label_indices[:100]
            # 提取这些样本的delay和loss_rate序列
            label_sequence = processed_df.iloc[sample_indices][['delay', 'loss_rate']].values.flatten()
            label_sequences[label] = label_sequence
    return label_sequences

def _find_closest_label(
    noise_sequence: np.ndarray,
    label_sequences: dict,
    unique_non_noise_labels: list
) -> int:
    """为噪声序列找到最接近的非噪声标签

    Args:
        noise_sequence: 噪声序列
        label_sequences: 包含每个标签及其代表序列的字典
        unique_non_noise_labels: 非噪声标签列表

    Returns:
        int: 最接近的非噪声标签
    """
    min_distance = float('inf')
    closest_label = unique_non_noise_labels[0]
    
    for label, sequence in label_sequences.items():
        # 确保序列长度相同
        if len(sequence) > len(noise_sequence):
            sequence = sequence[:len(noise_sequence)]
        elif len(sequence) < len(noise_sequence):
            # 填充较短的序列
            sequence = np.pad(sequence, (0, len(noise_sequence) - len(sequence)), 'constant')
        
        distance = _calculate_dtw_distance(noise_sequence, sequence)
        if distance < min_distance:
            min_distance = distance
            closest_label = label
    
    return closest_label

def fix_noise_labels(
    expanded_behavior_ids: np.ndarray,
    processed_df: pd.DataFrame
) -> np.ndarray:
    """优化噪声标签处理

    使用DTW距离最近的标签替换噪声标签，确保所有样本点都有有效的行为标签。

    Args:
        expanded_behavior_ids: 扩展后的行为标签数组，其中-1表示噪声标签
        processed_df: 处理后的数据DataFrame

    Returns:
        np.ndarray: 处理后的行为标签数组，无噪声标签
    """
    # 收集非噪声标签
    unique_non_noise_labels = [
        label for label in np.unique(expanded_behavior_ids) if label != -1
    ]
    
    if len(unique_non_noise_labels) == 0:
        # 如果没有非噪声标签，使用默认标签0
        logger.warning("所有行为标签都是噪声标签，使用默认标签0")
        return np.zeros_like(expanded_behavior_ids)
    
    # 收集每个非噪声标签的样本序列
    label_sequences = _collect_label_sequences(
        expanded_behavior_ids, processed_df, unique_non_noise_labels
    )
    
    # 初始化处理后的行为标签
    processed_behavior_ids = expanded_behavior_ids.copy()
    
    # 找出所有噪声标签的位置
    noise_indices = np.where(expanded_behavior_ids == -1)[0]
    logger.info(f"找到 {len(noise_indices)} 个噪声标签，使用DTW距离最近的标签替换")
    
    # 为每个噪声标签找到最近的非噪声标签
    for i in noise_indices:
        # 提取当前噪声标签位置附近的序列
        # 考虑到计算效率，只使用前后各10个样本
        start = max(0, i - 5)
        end = min(len(processed_df), i + 5)
        noise_sequence = processed_df.iloc[start:end][['delay', 'loss_rate']].values.flatten()
        
        # 计算与每个非噪声标签的DTW距离
        closest_label = _find_closest_label(
            noise_sequence, label_sequences, unique_non_noise_labels
        )
        
        # 替换噪声标签为最近的非噪声标签
        processed_behavior_ids[i] = closest_label
    
    logger.info("噪声标签替换完成")
    return processed_behavior_ids

def _find_mixed_sample_groups(
    valid_processed_df: pd.DataFrame,
    valid_expanded_behavior_ids: np.ndarray,
    sample_length: int,
    check_length: int = 100
) -> list:
    """寻找包含多个类别的样本组

    Args:
        valid_processed_df: 处理后的数据DataFrame
        valid_expanded_behavior_ids: 扩展后的行为标签数组
        sample_length: 样本长度
        check_length: 检查长度

    Returns:
        list: 包含混合类别样本组的列表
    """
    mixed_samples_found = []
    
    # 遍历所有可能的起始索引，寻找混合类别样本组
    for idx in range(0, len(valid_processed_df) - sample_length + 1, sample_length // 2):  # 步长减半，增加找到的概率
        # 检查更长的序列，提高多类别检测的准确性
        check_longer_length = 200
        if idx + check_longer_length <= len(valid_expanded_behavior_ids):
            current_sequence = valid_expanded_behavior_ids[idx : idx + check_longer_length]
        else:
            current_sequence = valid_expanded_behavior_ids[idx : idx + check_length]
        
        # 统计当前序列的主要行为类别（出现次数最多的行为类别）
        unique_current, counts_current = np.unique(current_sequence, return_counts=True)
        main_behavior = unique_current[np.argmax(counts_current)]
        main_behavior_ratio = np.max(counts_current) / len(current_sequence)
        
        # 检查是否包含多个类别
        if len(unique_current) >= 2:  # 至少包含2个不同类别
            mixed_samples_found.append({
                'idx': idx,
                'unique_count': len(unique_current),
                'main_behavior': main_behavior,
                'main_behavior_ratio': main_behavior_ratio,
                'unique_current': unique_current
            })
    
    # 按照包含的类别数量排序，优先选择包含更多类别的样本组
    mixed_samples_found.sort(key=lambda x: x['unique_count'], reverse=True)
    
    return mixed_samples_found

def _find_remaining_sample_groups(
    valid_processed_df: pd.DataFrame,
    valid_expanded_behavior_ids: np.ndarray,
    sample_length: int,
    check_length: int,
    start_indices: list,
    group_behavior_types: list,
    num_groups: int
) -> tuple:
    """寻找剩余样本组

    Args:
        valid_processed_df: 处理后的数据DataFrame
        valid_expanded_behavior_ids: 扩展后的行为标签数组
        sample_length: 样本长度
        check_length: 检查长度
        start_indices: 已找到的起始索引列表
        group_behavior_types: 已找到的组行为类型列表
        num_groups: 生成组数

    Returns:
        tuple: 包含起始索引列表和组行为类型列表的元组
    """
    # 遍历所有可能的起始索引，步长为sample_length，避免重叠
    for idx in range(0, len(valid_processed_df) - sample_length + 1, sample_length):
        if len(start_indices) >= num_groups:
            break
        
        # 只检查前100个行为ID，加速寻找过程
        current_sequence = valid_expanded_behavior_ids[idx : idx + check_length]
        # 统计当前序列的主要行为类别（出现次数最多的行为类别）
        unique_current, counts_current = np.unique(current_sequence, return_counts=True)
        main_behavior = unique_current[np.argmax(counts_current)]
        main_behavior_ratio = np.max(counts_current) / check_length
        
        # 检查该起始索引是否已经被选中
        if idx not in start_indices and main_behavior not in [t for t in group_behavior_types if not t.startswith('mixed_')]:
            start_indices.append(idx)
            group_behavior_types.append(main_behavior)
            logger.info(
                f"找到样本组 - 起始索引: {idx}, 主要行为类别: {main_behavior}, 比例: {main_behavior_ratio:.2f}"
            )
    
    return start_indices, group_behavior_types

def _find_additional_sample_groups(
    valid_processed_df: pd.DataFrame,
    valid_expanded_behavior_ids: np.ndarray,
    sample_length: int,
    check_length: int,
    start_indices: list,
    group_behavior_types: list,
    num_groups: int
) -> tuple:
    """使用额外策略寻找样本组

    Args:
        valid_processed_df: 处理后的数据DataFrame
        valid_expanded_behavior_ids: 扩展后的行为标签数组
        sample_length: 样本长度
        check_length: 检查长度
        start_indices: 已找到的起始索引列表
        group_behavior_types: 已找到的组行为类型列表
        num_groups: 生成组数

    Returns:
        tuple: 包含起始索引列表和组行为类型列表的元组
    """
    # 遍历所有可能的起始索引，找到不同的序列
    for idx in range(
        0, len(valid_processed_df) - sample_length + 1, sample_length // 2
    ):
        # 跳过已经选中的索引
        if idx in start_indices:
            continue
        
        # 只检查前100个行为ID，加速寻找过程
        current_sequence = valid_expanded_behavior_ids[idx : idx + check_length]
        # 统计当前序列的主要行为类别
        unique_current, counts_current = np.unique(
            current_sequence, return_counts=True
        )
        main_behavior = unique_current[np.argmax(counts_current)]
        
        # 添加到列表中
        start_indices.append(idx)
        group_behavior_types.append(main_behavior)
        logger.info(f"找到额外样本组 - 起始索引: {idx}, 主要行为类别: {main_behavior}")
        
        # 如果已经找到了足够的样本组，停止寻找
        if len(start_indices) >= num_groups:
            break
    
    return start_indices, group_behavior_types

def _find_random_sample_groups(
    valid_processed_df: pd.DataFrame,
    sample_length: int,
    start_indices: list,
    group_behavior_types: list,
    num_groups: int
) -> tuple:
    """使用随机选择补充样本组

    Args:
        valid_processed_df: 处理后的数据DataFrame
        sample_length: 样本长度
        start_indices: 已找到的起始索引列表
        group_behavior_types: 已找到的组行为类型列表
        num_groups: 生成组数

    Returns:
        tuple: 包含起始索引列表和组行为类型列表的元组
    """
    import random
    
    # 只尝试10次随机选择，避免无限循环
    max_attempts = 10
    attempts = 0
    
    while len(start_indices) < num_groups and attempts < max_attempts:
        # 随机选择一个起始索引
        random_idx = random.randint(0, len(valid_processed_df) - sample_length)
        
        # 添加到列表中
        start_indices.append(random_idx)
        group_behavior_types.append(1)  # 默认行为类别
        logger.info(f"随机选择样本组 - 起始索引: {random_idx}, 主要行为类别: 1")
        
        attempts += 1
    
    return start_indices, group_behavior_types

def select_sample_groups(
    valid_processed_df: pd.DataFrame,
    valid_expanded_behavior_ids: np.ndarray,
    sample_length: int,
    num_groups: int
) -> tuple:
    """选择样本组

    找到多组真实连续序列，每组对应不同的主要行为类别。

    Args:
        valid_processed_df: 处理后的数据DataFrame
        valid_expanded_behavior_ids: 扩展后的行为标签数组
        sample_length: 样本长度
        num_groups: 生成组数

    Returns:
        tuple: 包含起始索引列表和组行为类型列表的元组
    """
    # 统计所有行为类别的分布
    unique_behaviors, behavior_counts = np.unique(
        valid_expanded_behavior_ids, return_counts=True
    )
    behavior_distribution = dict(zip(unique_behaviors, behavior_counts))
    logger.info(f"行为类别分布: {behavior_distribution}")
    logger.info(f"参考样本涉及的行为类别: {sorted(unique_behaviors.tolist())}")
    
    # 遍历所有可能的起始索引，找到多组样本
    start_indices = []
    group_behavior_types = []
    
    # 使用更高效的方式寻找样本组
    # 只检查每个样本组的前100个行为ID，而不是全部6000个
    check_length = 100
    
    # 首先寻找包含多个类别的样本组
    logger.info("首先寻找包含多个类别的样本组...")
    mixed_samples_found = _find_mixed_sample_groups(
        valid_processed_df, valid_expanded_behavior_ids, sample_length, check_length
    )
    
    # 选择前num_groups个样本组
    for sample in mixed_samples_found:
        if len(start_indices) >= num_groups:
            break
        
        # 避免重复的主要行为类别
        if sample['main_behavior'] not in [t for t in group_behavior_types if not t.startswith('mixed_')]:
            start_indices.append(sample['idx'])
            group_behavior_types.append(f"mixed_{sample['main_behavior']}")
            logger.info(
                f"找到多样化样本组 - 起始索引: {sample['idx']}, 包含 {sample['unique_count']} 个类别, 主要行为: {sample['main_behavior']}, 比例: {sample['main_behavior_ratio']:.2f}"
            )
    
    # 如果找不到足够的混合类别样本组，使用原来的逻辑寻找样本组
    if len(start_indices) < num_groups:
        logger.info(f"只找到 {len(start_indices)} 个混合类别样本组，使用原始逻辑寻找剩余样本组...")
        start_indices, group_behavior_types = _find_remaining_sample_groups(
            valid_processed_df, valid_expanded_behavior_ids, sample_length,
            check_length, start_indices, group_behavior_types, num_groups
        )
    
    # 如果找到的样本组不足，使用额外的策略
    if len(start_indices) < num_groups:
        logger.info(f"只找到 {len(start_indices)} 组样本，使用额外策略寻找剩余样本组...")
        start_indices, group_behavior_types = _find_additional_sample_groups(
            valid_processed_df, valid_expanded_behavior_ids, sample_length,
            check_length, start_indices, group_behavior_types, num_groups
        )
    
    # 如果仍然找不到足够的样本组，使用随机选择
    if len(start_indices) < num_groups:
        logger.info(f"仍然只找到 {len(start_indices)} 组样本，使用随机选择补充...")
        start_indices, group_behavior_types = _find_random_sample_groups(
            valid_processed_df, sample_length, start_indices, group_behavior_types, num_groups
        )
    
    return start_indices, group_behavior_types

def load_model(input_model_path: Path, device: torch.device) -> tuple:
    """加载模型

    加载训练好的条件扩散模型，并返回模型实例、行为映射和归一化参数。

    Args:
        input_model_path: 训练好的扩散模型文件路径
        device: 模型运行设备

    Returns:
        tuple: 包含模型实例、行为映射和归一化参数的元组
    """
    logger.info(f"正在加载模型: {input_model_path}")
    checkpoint = torch.load(input_model_path, map_location=device, weights_only=False)
    behavior_mapping = checkpoint["behavior_mapping"]
    num_behaviors = len(behavior_mapping)
    
    # 初始化条件扩散模型
    model = ConditionDiffusionModel(
        input_dim=2,  # 输入维度：延迟和丢包率
        num_behaviors=num_behaviors,
        behavior_embed_dim=32,  # 行为嵌入维度
        T=1000,  # 扩散步数
    )
    
    # 加载模型状态并设置为评估模式（添加strict=False处理架构变化）
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    
    # 从checkpoint中加载反归一化参数
    # 检查归一化方法
    normalization_method = checkpoint.get("normalization_method", "robust")
    # 将numpy字符串数组转换为Python字符串
    if isinstance(normalization_method, np.ndarray):
        normalization_method = normalization_method.item()
    
    # 检查是否使用对数变换
    use_log_transform = checkpoint.get('use_log_transform', False)
    # 将numpy布尔值转换为Python布尔值
    if isinstance(use_log_transform, np.ndarray):
        use_log_transform = use_log_transform.item()
    
    # 打印反归一化参数，用于调试
    logger.info("\n反归一化参数：")
    logger.info(f"归一化方法: {normalization_method}")
    logger.info(f"是否使用对数变换: {use_log_transform}")
    
    # 根据归一化方法加载不同的参数
    delay_norm_params = {
        'normalization_method': normalization_method,
        'use_log_transform': use_log_transform  # 从检查点读取对数变换参数
    }
    
    if normalization_method == 'robust':
        # 加载robust归一化参数
        delay_scaler_center_ = checkpoint["delay_scaler_center_"]  # 注意：这里存储的是center_而不是mean_
        delay_scaler_scale_ = checkpoint["delay_scaler_scale_"]
        current_min = checkpoint.get("robust_scale_min", -1.0)  # 对应预处理中的current_min
        current_max = checkpoint.get("robust_scale_max", 1.0)  # 对应预处理中的current_max
        
        logger.info(f"delay_scaler_center_: {delay_scaler_center_}")
        logger.info(f"delay_scaler_scale_: {delay_scaler_scale_}")
        logger.info(f"current_min: {current_min}")
        logger.info(f"current_max: {current_max}")
        
        # 添加robust归一化参数
        delay_norm_params.update({
            'delay_scaler_center_': delay_scaler_center_,
            'delay_scaler_scale_': delay_scaler_scale_,
            'robust_scale_min': current_min,
            'robust_scale_max': current_max
        })
    else:
        # 加载分位数归一化参数
        quantile_sorted_data = checkpoint["quantile_sorted_data"]
        quantile_quantiles = checkpoint["quantile_quantiles"]
        
        logger.info(f"分位数参数 - 数据点数: {len(quantile_sorted_data)}")
        
        # 添加分位数归一化参数
        delay_norm_params.update({
            'quantile_params': {
                'sorted_data': quantile_sorted_data,
                'quantiles': quantile_quantiles
            }
        })
    
    return model, behavior_mapping, delay_norm_params

def generate_single_sample_group(
    i: int,
    start_idx: int,
    group_behavior_type: any,
    valid_processed_df: pd.DataFrame,
    valid_expanded_behavior_ids: np.ndarray,
    sample_length: int,
    behavior_mapping: dict,
    model: ConditionDiffusionModel,
    device: torch.device,
    delay_norm_params: dict,
    loss_norm_params: dict,
    constraint_injector: ConstraintInjector,
    output_generation_dir: Path
) -> tuple:
    """生成单个样本组

    为指定的起始索引生成对应的网络状态数据。

    Args:
        i: 样本组索引
        start_idx: 起始索引
        group_behavior_type: 组行为类型
        valid_processed_df: 处理后的数据DataFrame
        valid_expanded_behavior_ids: 扩展后的行为标签数组
        sample_length: 样本长度
        behavior_mapping: 行为映射字典
        model: 条件扩散模型实例
        device: 模型运行设备
        delay_norm_params: 延迟归一化参数
        loss_norm_params: 丢包率归一化参数
        constraint_injector: 约束注入器实例
        output_generation_dir: 生成样本的输出目录路径

    Returns:
        tuple: 包含原始样本文件路径和生成样本文件路径的元组，
              如果生成失败则返回(None, None)
    """
    logger.info(f"\n处理样本组 {i + 1}/{len(start_indices)}...")
    logger.info(f"起始索引: {start_idx}, 主要行为类别: {group_behavior_type}")
    
    # 选择当前组的样本
    selected_df = valid_processed_df.iloc[
        start_idx : start_idx + sample_length
    ].copy()
    selected_behavior_ids = valid_expanded_behavior_ids[
        start_idx : start_idx + sample_length
    ]
    
    # 将原始行为ID转换为模型内部使用的ID，处理不在映射中的行为ID
    model_behavior_ids = []
    # 获取模型训练时的所有原始行为ID
    model_behavior_keys = list(behavior_mapping.keys())
    
    for behavior_id in selected_behavior_ids:
        if behavior_id in behavior_mapping:
            model_behavior_ids.append(behavior_mapping[behavior_id])
        else:
            # 对于不在映射中的行为ID，将其映射到数值最接近的行为ID
            closest_behavior = min(model_behavior_keys, key=lambda x: abs(x - behavior_id))
            model_behavior_ids.append(behavior_mapping[closest_behavior])
            logger.warning(
                f"行为ID {behavior_id} 不在模型映射中，已转换为最接近的行为ID {closest_behavior}"
            )
    
    # 确保样本长度符合要求
    if len(selected_behavior_ids) < sample_length:
        logger.warning(f"样本组 {i + 1} 样本不足 {sample_length}，跳过")
        return None, None
    
    # 转换行为ID为张量，使用转换后的model_behavior_ids
    behavior_ids_tensor = (
        torch.tensor(model_behavior_ids, dtype=torch.long).unsqueeze(0).to(device)
    )
    
    # 生成样本
    logger.info("正在生成网络状态数据...")
    with torch.no_grad():
        generated = model.sample(behavior_ids_tensor, device)
    
    # 转换回numpy数组
    generated = generated.cpu().numpy()[0]
    
    # 导入归一化模块
    from network_simulation.condition_generation.normalization import denormalize_delay, denormalize_loss_rate
    
    # 反归一化延迟 - 使用统一的反归一化函数
    delay_norm = generated[:, 0]
    delay = denormalize_delay(delay_norm, delay_norm_params)
    
    # 添加适当的约束，确保生成的数据的范围与原始数据一致
    # 使用原始数据的统计特性来约束生成的延迟值
    original_min = selected_df['delay'].min()
    original_max = selected_df['delay'].max()
    original_mean = selected_df['delay'].mean()
    original_std = selected_df['delay'].std()
    
    # 打印原始数据的统计特性，用于调试
    logger.info(f"原始数据统计 - 最小值: {original_min:.4f}, 最大值: {original_max:.4f}, 平均值: {original_mean:.4f}, 标准差: {original_std:.4f}")
    
    # 调整约束逻辑：使用原始数据的3倍标准差作为约束范围，减少极端值的生成
    # 同时保留原始数据的最大值作为绝对上限
    constraint_upper = min(original_max, original_mean + 3 * original_std)
    
    # 打印约束上限，用于调试
    logger.info(f"约束上限: {constraint_upper:.4f}")
    
    # 1. 首先对原始范围外的值进行软约束
    # 对于超过约束上限但未超过原始最大值的值，进行渐进式压缩
    # 这样可以减少大量值被截断到最大值的情况
    mask = (delay > constraint_upper) & (delay <= original_max)
    if np.any(mask):
        # 渐进式压缩：随着值超过约束上限越多，压缩程度越大
        # 使用指数衰减函数压缩
        excess = delay[mask] - constraint_upper
        max_excess = original_max - constraint_upper
        if max_excess > 0:
            # 计算压缩系数，从1到0线性变化
            compression_factor = 1 - (excess / max_excess)
            # 应用压缩，将超出部分压缩到[constraint_upper, original_max]范围内
            compressed_values = constraint_upper + excess * compression_factor
            delay[mask] = compressed_values
    
    # 2. 最后限制生成的延迟在合理范围内（0到原始最大值）
    delay = np.clip(delay, a_min=0, a_max=original_max)
    
    # 3. 对生成的延迟值添加额外的平滑处理，减少极端值的影响
    # 使用简单的中值滤波去除孤立的极端值
    from scipy.signal import medfilt
    # 只对接近最大值的值进行中值滤波
    max_threshold = original_max * 0.9
    mask = delay >= max_threshold
    if np.any(mask):
        # 对整个序列进行中值滤波
        delay_smoothed = medfilt(delay, kernel_size=5)
        # 只替换接近最大值的值
        delay[mask] = delay_smoothed[mask]
    
    # 打印裁剪后的延迟统计，用于调试
    logger.info(f"裁剪后延迟统计 - 最小值: {delay.min():.4f}, 最大值: {delay.max():.4f}, 平均值: {delay.mean():.4f}, 标准差: {delay.std():.4f}")
    
    # 反归一化丢包率 - 使用统一的反归一化函数
    loss_norm = generated[:, 1]
    loss_rate = denormalize_loss_rate(loss_norm, loss_norm_params)
    
    # 添加额外处理，确保生成的丢包率分布更接近原始数据
    # 1. 计算原始数据中丢包率为0的比例
    original_zero_loss_rate = (selected_df['loss_rate'] == 0).sum() / len(selected_df)
    
    # 2. 如果原始数据中大多数丢包率为0，调整生成的丢包率分布
    if original_zero_loss_rate > 0.8:  # 如果原始数据中80%以上的丢包率为0
        # 计算需要调整为0的丢包率数量
        target_zero_count = int(len(loss_rate) * original_zero_loss_rate)
        
        # 随机选择一些位置将丢包率设置为0
        zero_indices = np.random.choice(
            len(loss_rate),
            size=target_zero_count,
            replace=False
        )
        loss_rate[zero_indices] = 0.0
    
    # 创建生成的数据Frame
    generated_df = selected_df.copy()
    generated_df["delay"] = delay
    generated_df["loss_rate"] = loss_rate
    
    # 验证生成的数据是否符合网络约束
    delay = generated_df["delay"].values
    loss_rate = generated_df["loss_rate"].values
    validation = constraint_injector.validate_sequence(delay, loss_rate)
    logger.info(f"生成数据验证结果: {validation}")
    
    # 保存当前组的生成数据
    original_file = (
        output_generation_dir
        / f"original_sample_6000_group_{i + 1}_behavior_{group_behavior_type}.csv"
    )
    generated_file = (
        output_generation_dir
        / f"generated_sample_6000_group_{i + 1}_behavior_{group_behavior_type}.csv"
    )
    
    selected_df.to_csv(original_file, index=False)
    generated_df.to_csv(generated_file, index=False)
    
    logger.info(f"样本组 {i + 1} 已保存")
    logger.info(f"原始数据: {original_file}")
    logger.info(f"生成数据: {generated_file}")
    
    return original_file, generated_file

def generate_samples(
    input_processed_file: Path,
    input_patterns_dir: Path,
    input_model_path: Path,
    output_generation_dir: Path,
):
    """生成网络状态样本数据

    该函数使用训练好的条件扩散模型，根据输入的行为标签生成网络状态样本数据，
    包括延迟和丢包率，并确保生成的数据符合网络约束条件。

    Args:
        input_processed_file: 处理后的数据文件路径，包含timestamp、delay、loss_rate等字段
        input_patterns_dir: 行为模式目录路径，包含valid_loss_values.json和behavior_labels_*.json文件
        input_model_path: 训练好的扩散模型文件路径
        output_generation_dir: 生成样本的输出目录路径

    Returns:
        tuple: 包含原始样本文件路径和生成样本文件路径的元组

    Raises:
        FileNotFoundError: 如果输入文件或目录不存在
        ValueError: 如果数据格式不符合要求

    输入输出示例：
        输入：
            input_processed_file: Path("data/processed/network_data.csv")
            input_patterns_dir: Path("data/patterns/")
            input_model_path: Path("models/diffusion_model_final.pth")
            output_generation_dir: Path("data/generated/")
        输出：
            (PosixPath('data/generated/original_sample_6000.csv'), PosixPath('data/generated/generated_sample_6000.csv'))

    数据格式：
        - 输入CSV文件：包含timestamp（datetime）、delay（float）、loss_rate（float）等字段
        - 输出CSV文件：与输入格式相同，但delay和loss_rate为生成值
        - behavior_labels.json：包含labels数组，每个元素为行为ID
        - valid_loss_values.json：包含合法丢包值数组
    """
    logger.info(f"正在使用模型生成样本: {input_model_path}")

    # 设备设置：优先使用MacBook MPS GPU，然后是CUDA，最后是CPU
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"使用设备: {device}")

    # 加载合法丢包值
    valid_loss_values = load_valid_loss_values(input_patterns_dir)
    
    # 初始化约束注入器：确保生成的丢包率符合合法范围
    constraint_injector = ConstraintInjector(valid_loss_values)

    # 加载处理后的数据：包含原始网络状态数据
    processed_df = pd.read_csv(input_processed_file, parse_dates=["timestamp"])

    # 加载行为标签
    behavior_ids = load_behavior_labels(input_patterns_dir)

    # 扩展行为标签到样本级别（使用10秒窗口，50%重叠，与预处理一致）
    window_size = DEFAULT_WINDOW_SIZE  # 10秒窗口，每个时间点100ms
    stride = DEFAULT_STRIDE  # 50%重叠
    expanded_behavior_ids = expand_behavior_labels(behavior_ids, processed_df, window_size, stride)

    # 优化噪声标签处理，使用DTW距离最近的标签替换噪声标签
    processed_behavior_ids = fix_noise_labels(expanded_behavior_ids, processed_df)

    # 保存处理后的行为标签和对应的处理后数据
    valid_expanded_behavior_ids = processed_behavior_ids
    valid_processed_df = processed_df.copy()

    # 批量样本选择：找到多组真实连续序列，每组对应不同的主要行为类别
    sample_length = DEFAULT_SAMPLE_LENGTH  # 从配置文件加载样本长度
    num_groups = DEFAULT_NUM_GROUPS  # 从配置文件加载生成组数

    # 确保valid_processed_df有足够的样本
    if len(valid_processed_df) < sample_length:
        logger.warning(f"有效样本数不足 {sample_length}，无法生成样本")
        return None, None

    # 选择样本组
    start_indices, group_behavior_types = select_sample_groups(
        valid_processed_df, valid_expanded_behavior_ids, sample_length, num_groups
    )

    # 2. 为每组样本生成对应的网络状态数据
    original_files = []
    generated_files = []

    # 加载模型
    model, behavior_mapping, delay_norm_params = load_model(input_model_path, device)
    
    # 构建丢包率归一化参数字典
    loss_norm_params = {
        'valid_loss_values': valid_loss_values
    }

    # 为每组样本生成数据
    for i, (start_idx, group_behavior_type) in enumerate(zip(start_indices, group_behavior_types)):
        # 保存当前start_indices到全局变量，以便generate_single_sample_group函数使用
        global start_indices
        original_file, generated_file = generate_single_sample_group(
            i, start_idx, group_behavior_type,
            valid_processed_df, valid_expanded_behavior_ids,
            sample_length, behavior_mapping,
            model, device, delay_norm_params,
            loss_norm_params, constraint_injector,
            output_generation_dir
        )
        
        if original_file and generated_file:
            original_files.append(original_file)
            generated_files.append(generated_file)

    return original_files, generated_files


def cleanup_old_files(directory, pattern):
    """清理目录下的旧文件，只保留最新的KEEP_LATEST_FILES个文件

    Args:
        directory: 要清理的目录
        pattern: 要清理的文件模式
    """

    if not CLEANUP_OLD_FILES:
        return

    files = list(directory.glob(pattern))
    if len(files) <= KEEP_LATEST_FILES:
        return

    # 按修改时间排序，最新的在前
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # 删除旧文件
    for file in files[KEEP_LATEST_FILES:]:
        file.unlink()
        logger.info(f"已清理旧文件: {file}")


def main():
    """主函数入口

    解析命令行参数，调用generate_samples函数生成网络状态样本数据。

    命令行参数：
        python scripts/step2_2_generate_samples.py [input_processed_file_or_dir] [input_patterns_dir] [input_model_dir] [output_generation_dir]

    参数说明：
        input_processed_file_or_dir: 处理后的数据文件或目录路径 (默认: data/processed)
        input_patterns_dir: 行为模式目录路径 (默认: data/results/patterns)
        input_model_dir: 训练好的模型目录路径 (默认: data/models/diffusion_model)
        output_generation_dir: 生成样本的输出目录路径 (默认: data/generated)
    """
    import argparse

    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="生成网络状态样本数据")
    parser.add_argument(
        "input_processed",
        nargs="?",
        type=Path,
        default=PROCESSED_DIR,
        help="处理后的数据文件或目录路径 (默认: data/processed)",
    )
    parser.add_argument(
        "input_patterns",
        nargs="?",
        type=Path,
        default=PATTERNS_DIR,
        help="行为模式目录路径 (默认: data/results/patterns)",
    )
    parser.add_argument(
        "input_model",
        nargs="?",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="训练好的模型目录路径 (默认: data/models/diffusion_model)",
    )
    parser.add_argument(
        "output_generation",
        nargs="?",
        type=Path,
        default=GENERATED_DIR,
        help="生成样本的输出目录路径 (默认: data/generated)",
    )

    args = parser.parse_args()

    input_processed_path = args.input_processed
    input_patterns_dir = args.input_patterns
    input_model_dir = args.input_model
    output_generation_dir = args.output_generation

    # 确保输出目录存在
    output_generation_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧文件
    cleanup_old_files(output_generation_dir, "original_sample_*.csv")
    cleanup_old_files(output_generation_dir, "generated_sample_*.csv")

    # 查找模型文件
    model_path = input_model_dir / "diffusion_model_final.pth"
    if not model_path.exists():
        logger.error(f"模型文件 {model_path} 不存在")
        sys.exit(1)

    # 生成样本
    if input_processed_path.is_file():
        # 如果输入是一个文件，处理单个文件
        generate_samples(
            input_processed_path, input_patterns_dir, model_path, output_generation_dir
        )
    elif input_processed_path.is_dir():
        # 如果输入是一个文件夹，选择第一个处理后的文件
        processed_files = list(input_processed_path.glob("*.csv"))
        if not processed_files:
            logger.warning(f"在 {input_processed_path} 中未找到 .csv 文件")
        sys.exit(1)

        # 使用第一个处理后的文件
        selected_processed_file = processed_files[0]

        logger.info(f"\n正在使用 {selected_processed_file} 生成样本...")
        # 生成样本，使用单个文件
        generate_samples(
            selected_processed_file,
            input_patterns_dir,
            model_path,
            output_generation_dir,
        )
    else:
        print(f"错误: 输入 {input_processed_path} 不是文件或目录")
        sys.exit(1)

    logger.info("步骤2.2：生成样本数据完成！")


if __name__ == "__main__":
    main()