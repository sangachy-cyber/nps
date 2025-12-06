#!/usr/bin/env python3
"""
步骤2.0：预处理训练数据
"""

import sys
import os

# 添加当前目录到Python路径，以便找到config模块
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.preprocessing import RobustScaler


def preprocess_data(
    input_patterns_dir: Path, input_processed_file: Path, output_preprocess_dir: Path
):
    """预处理扩散模型训练数据

    该函数对网络数据进行预处理，包括行为标签对齐、噪声剔除、特征归一化等，
    为扩散模型训练准备格式化的输入数据。

    Args:
        input_patterns_dir: 行为模式目录路径，包含valid_loss_values.json和behavior_labels_*.json文件
        input_processed_file: 处理后的数据文件路径，包含timestamp、delay、loss_rate等字段
        output_preprocess_dir: 预处理结果的输出目录路径

    Returns:
        Path: 预处理数据文件路径

    Raises:
        FileNotFoundError: 如果输入文件或目录不存在
        ValueError: 如果数据格式不符合要求

    输入输出示例：
        输入：
            input_patterns_dir: Path("data/results/patterns/")
            input_processed_file: Path("data/processed/20251203_230356_b6x-playback_processed.csv")
            output_preprocess_dir: Path("data/results/preprocessed/")
        输出：
            PosixPath('data/results/preprocessed/preprocess_data.npz')
    """
    print(f"正在为扩散模型预处理数据: {input_patterns_dir}")

    # 加载处理后的数据
    processed_df = pd.read_csv(input_processed_file, parse_dates=["timestamp"])
    total_samples = len(processed_df)
    print(f"处理后数据总样本数: {total_samples}")

    # 加载合法丢包值
    valid_loss_values_file = input_patterns_dir / "metadata" / "valid_loss_values.json"
    with open(valid_loss_values_file, "r") as f:
        valid_loss_values = json.load(f)

    # 构建loss_rate到index的映射
    loss_rate_mapping = {value: idx for idx, value in enumerate(valid_loss_values)}
    N_vals = len(valid_loss_values)
    print(f"合法丢包值: {valid_loss_values}")
    print(f"丢包率映射: {loss_rate_mapping}")

    # 导入配置
    from config import DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE
    
    # 1. 处理行为标签对齐
    # 加载行为标签（从JSON文件）
    behavior_labels_files = list(input_patterns_dir.glob("behavior_labels_*.json"))
    if not behavior_labels_files:
        raise FileNotFoundError(f"在 {input_patterns_dir} 中未找到行为标签文件")
    
    behavior_labels_file = behavior_labels_files[0]  # 使用第一个行为标签文件
    with open(behavior_labels_file, "r") as f:
        behavior_labels_data = json.load(f)
    behavior_ids = np.array(behavior_labels_data["labels"])
    
    # 获取所有原始行为标签，用于创建完整的映射
    all_original_behaviors = np.unique(behavior_ids)
    # 剔除噪声标签(-1)，因为它们已经被处理
    all_original_behaviors = all_original_behaviors[all_original_behaviors != -1]

    # 使用配置文件中的窗口参数
    window_size = DEFAULT_WINDOW_SIZE  # 从配置文件加载
    stride = DEFAULT_STRIDE  # 从配置文件加载

    # 计算合并数据的窗口数量
    total_windows = (total_samples - window_size) // stride + 1
    num_behavior_labels = len(behavior_ids)

    print(
        f"原始行为标签数量: {num_behavior_labels}, 合并数据窗口数量: {total_windows}, 窗口大小: {window_size}, 重叠步长: {stride}"
    )

    # 确保行为标签数量与窗口数量匹配
    if num_behavior_labels != total_windows:
        print(
            f"警告: 行为标签数量 {num_behavior_labels} 与窗口数量 {total_windows} 不匹配"
        )
        # 改进行为标签调整逻辑
        if num_behavior_labels > total_windows:
            # 行为标签数量过多，采用均匀采样方式保留
            step = num_behavior_labels / total_windows
            behavior_ids = behavior_ids[[int(round(i * step)) for i in range(total_windows)]]
            print(f"已均匀采样行为标签数量为 {len(behavior_ids)}")
        else:
            # 行为标签数量不足，使用插值方式填充
            # 计算需要填充的标签数量
            padding = total_windows - num_behavior_labels
            if padding > 0:
                # 计算每个现有标签需要重复的次数
                repeat_counts = [1] * num_behavior_labels
                for i in range(padding):
                    # 找出当前重复次数最少的标签，增加其重复次数
                    min_idx = repeat_counts.index(min(repeat_counts))
                    repeat_counts[min_idx] += 1
                
                # 根据重复次数扩展行为标签
                expanded_behavior_ids = []
                for label, count in zip(behavior_ids, repeat_counts):
                    expanded_behavior_ids.extend([label] * count)
                
                # 截取到需要的长度
                behavior_ids = np.array(expanded_behavior_ids[:total_windows], dtype=int)
                print(f"已使用插值方式填充行为标签数量为 {len(behavior_ids)}")

    # 更新窗口数量
    num_windows = len(behavior_ids)

    # 初始化行为标签数组，-2表示未分配
    behavior_labels = np.full(total_samples, -2, dtype=int)

    # 计算每个窗口的起始和结束位置
    window_info = []
    for i in range(num_windows):
        window_start = i * stride
        window_end = window_start + window_size
        window_info.append((window_start, window_end, behavior_ids[i]))

    # 为每个时间点分配行为标签
    for t in range(total_samples):
        # 找到包含当前时间点的所有窗口
        matching_windows = []
        for i, (start, end, behavior_id) in enumerate(window_info):
            if start <= t < end:
                matching_windows.append((i, behavior_id))

        if matching_windows:
            # 如果有多个窗口包含当前时间点，选择中间的窗口
            if len(matching_windows) > 1:
                # 选择中间的窗口
                middle_idx = len(matching_windows) // 2
                behavior_labels[t] = matching_windows[middle_idx][1]
            else:
                # 只有一个窗口包含当前时间点
                behavior_labels[t] = matching_windows[0][1]
        else:
            # 没有窗口包含当前时间点，使用最近的窗口
            # 计算与所有窗口中心的距离
            window_centers = [(start + end) // 2 for start, end, _ in window_info]
            closest_window_idx = np.argmin(np.abs(np.array(window_centers) - t))
            behavior_labels[t] = window_info[closest_window_idx][2]

    # 检查是否还有未分配的标签
    if np.any(behavior_labels == -2):
        print(f"警告: {np.sum(behavior_labels == -2)} 个样本没有分配行为标签")
        # 优化未分配标签的填充逻辑
        # 首先填充开头的未分配标签
        for t in range(total_samples):
            if behavior_labels[t] != -2:
                # 找到第一个有效标签
                first_valid_label = behavior_labels[t]
                # 填充前面的所有未分配标签
                for i in range(t):
                    behavior_labels[i] = first_valid_label
                break

        # 然后填充结尾的未分配标签
        for t in range(total_samples - 1, -1, -1):
            if behavior_labels[t] != -2:
                # 找到最后一个有效标签
                last_valid_label = behavior_labels[t]
                # 填充后面的所有未分配标签
                for i in range(t + 1, total_samples):
                    behavior_labels[i] = last_valid_label
                break

        # 最后处理中间的未分配标签
        for t in range(total_samples):
            if behavior_labels[t] == -2:
                # 找到左边最近的有效标签
                left = t - 1
                while left >= 0 and behavior_labels[left] == -2:
                    left -= 1
                left_label = (
                    behavior_labels[left] if left >= 0 else behavior_labels[t + 1]
                )

                # 找到右边最近的有效标签
                right = t + 1
                while right < total_samples and behavior_labels[right] == -2:
                    right += 1
                right_label = (
                    behavior_labels[right]
                    if right < total_samples
                    else behavior_labels[t - 1]
                )

                # 使用左右标签的平均值或随机选择一个
                behavior_labels[t] = left_label if left >= 0 else right_label

    # 统计行为标签分布
    unique_behaviors, counts = np.unique(behavior_labels, return_counts=True)
    print(f"行为标签分布:")
    for behavior, count in zip(unique_behaviors, counts):
        print(f"类别 {behavior}: {count} 个样本 ({count / total_samples * 100:.2f}%)")

    # 2. 剔除噪声标签（-1）
    non_noise_mask = behavior_labels != -1
    valid_samples = non_noise_mask.sum()
    print(f"有效样本数（排除噪声）: {valid_samples}/{total_samples}")

    # 只保留非噪声样本
    valid_df = processed_df[non_noise_mask].copy()
    valid_behavior_labels = behavior_labels[non_noise_mask]
    
    # 3. 实现样本均衡，减少类别8的占比
    # 统计各类别样本数量
    unique_labels, label_counts = np.unique(valid_behavior_labels, return_counts=True)
    label_distribution = dict(zip(unique_labels, label_counts))
    print(f"均衡前样本分布: {label_distribution}")
    
    # 找出最常见的类别（类别8）
    most_common_label = max(label_distribution, key=label_distribution.get)
    most_common_count = label_distribution[most_common_label]
    
    # 计算其他类别的平均样本数
    other_labels = [label for label in unique_labels if label != most_common_label]
    if other_labels:
        avg_other_count = int(np.mean([label_distribution[label] for label in other_labels]))
        
        # 如果最常见类别的样本数远多于其他类别的平均值，进行采样
        if most_common_count > 2 * avg_other_count:
            # 计算需要保留的最常见类别的样本数
            target_count = int(avg_other_count * 1.5)  # 保留1.5倍的平均样本数
            
            # 获取最常见类别的索引
            most_common_indices = np.where(valid_behavior_labels == most_common_label)[0]
            
            # 随机采样最常见类别
            sampled_indices = np.random.choice(most_common_indices, size=target_count, replace=False)
            
            # 获取其他类别的索引
            other_indices = np.where(np.isin(valid_behavior_labels, other_labels))[0]
            
            # 合并采样后的索引
            balanced_indices = np.concatenate([sampled_indices, other_indices])
            
            # 按原始顺序排序
            balanced_indices.sort()
            
            # 更新有效样本和行为标签
            valid_df = valid_df.iloc[balanced_indices].copy()
            valid_behavior_labels = valid_behavior_labels[balanced_indices]
            
            # 统计均衡后的样本分布
            balanced_unique_labels, balanced_label_counts = np.unique(valid_behavior_labels, return_counts=True)
            balanced_distribution = dict(zip(balanced_unique_labels, balanced_label_counts))
            print(f"均衡后样本分布: {balanced_distribution}")
            print(f"已将类别 {most_common_label} 的样本数从 {most_common_count} 减少到 {target_count}")

    # 3. 特征归一化
    # 导入归一化模块
    from network_simulation.condition_generation.normalization import normalize_delay, normalize_loss_rate
    
    # 3.1 delay归一化：使用统一的归一化函数
    print("Normalizing delay values...")
    delay_values = valid_df["delay"].values
    # 使用分位数归一化方法，并启用对数变换
    delay_norm, delay_norm_params = normalize_delay(delay_values, method='quantile', use_log_transform=True)
    
    print(
        f"Delay normalization - Min: {np.min(delay_norm):.4f}, Max: {np.max(delay_norm):.4f}, Mean: {np.mean(delay_norm):.4f}"
    )
    
    # 检查归一化方法，仅在robust方法时输出robust参数
    normalization_method = delay_norm_params.get('normalization_method', 'robust')
    
    # 初始化所有可能需要的参数，确保在两种归一化方法下都有定义
    delay_scaler_center_ = None
    delay_scaler_scale_ = None
    robust_scale_min = None
    robust_scale_max = None
    quantile_params = None
    
    if normalization_method == 'robust':
        # 提取延迟归一化参数
        delay_scaler_center_ = delay_norm_params['delay_scaler_center_']
        delay_scaler_scale_ = delay_norm_params['delay_scaler_scale_']
        robust_scale_min = delay_norm_params['robust_scale_min']
        robust_scale_max = delay_norm_params['robust_scale_max']
        
        print(
            f"RobustScaler - Center: {delay_scaler_center_[0]:.4f}, Scale: {delay_scaler_scale_[0]:.4f}"
        )
        print(
            f"RobustScale - Min: {robust_scale_min:.4f}, Max: {robust_scale_max:.4f}"
        )
    else:
        # 提取分位数归一化参数
        quantile_params = delay_norm_params['quantile_params']
        print(f"使用{normalization_method}归一化方法")
        print(f"分位数参数 - 数据点数: {len(quantile_params['sorted_data'])}")
    
    # 保存完整的归一化参数，包括归一化方法
    delay_norm_params['normalization_method'] = normalization_method

    # 3.2 loss_rate归一化：使用统一的归一化函数
    print("归一化丢包率值...")
    loss_rate_values = valid_df["loss_rate"].values
    
    # 检查训练数据的丢包率分布
    unique_loss_rates = np.unique(loss_rate_values)
    print(f"训练数据丢包率唯一值: {unique_loss_rates}")
    
    loss_norm, loss_norm_params = normalize_loss_rate(loss_rate_values, valid_loss_values)
    
    print(
        f"丢包率归一化 - 最小值: {np.min(loss_norm):.4f}, 最大值: {np.max(loss_norm):.4f}, 平均值: {np.mean(loss_norm):.4f}"
    )

    # 4. 准备训练数据
    # 改进：使用全部有效样本进行训练，而不是只选择连续的6000个样本
    # 这样模型可以学习完整的数据分布，包括极端值
    sample_length = len(valid_df)  # 使用所有有效样本
    start_idx = 0

    selected_df = valid_df.copy()
    selected_delay_norm = delay_norm.copy()
    selected_loss_norm = loss_norm.copy()
    selected_behavior_labels = valid_behavior_labels.copy()

    print(f"选择的样本长度: {len(selected_df)}")

    # 重新映射行为ID为0-based索引
    # 使用所有原始行为标签创建映射，确保包含所有可能的行为标签
    # 首先使用所有原始行为标签创建完整映射
    behavior_mapping = {b: i for i, b in enumerate(all_original_behaviors)}
    
    # 然后为每个保留的行为标签分配对应的模型内部ID
    selected_behavior_ids = np.array(
        [behavior_mapping[b] for b in selected_behavior_labels]
    )
    num_behaviors = len(all_original_behaviors)

    print(f"所有原始行为: {all_original_behaviors}")
    print(f"唯一行为: {np.unique(selected_behavior_labels)}")
    print(f"行为映射: {behavior_mapping}")
    print(f"行为数量: {num_behaviors}")

    # 5. 合并特征
    features = np.column_stack([selected_delay_norm, selected_loss_norm])

    # 6. 转换为张量（适应M1芯片）
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(
        0
    )  # 添加批次维度
    behavior_ids_tensor = torch.tensor(
        selected_behavior_ids, dtype=torch.long
    ).unsqueeze(0)

    # 7. 创建输出目录
    output_preprocess_dir.mkdir(parents=True, exist_ok=True)

    # 8. 保存预处理结果
    # 根据归一化方法保存不同的参数
    preprocess_data = {
        "features": features,
        "behavior_ids": selected_behavior_ids,
        "behavior_mapping": behavior_mapping,
        "loss_rate_mapping": loss_rate_mapping,
        "valid_loss_values": valid_loss_values,
        "sample_length": len(selected_df),
        "num_behaviors": num_behaviors,
        "delay_norm": selected_delay_norm,
        "loss_norm": selected_loss_norm,
        "normalization_method": normalization_method,
        "use_log_transform": delay_norm_params.get('use_log_transform', False),
    }
    
    # 保存归一化方法特定的参数
    if normalization_method == 'robust':
        # 保存robust归一化参数
        preprocess_data.update({
            "delay_scaler_center_": delay_scaler_center_,
            "delay_scaler_scale_": delay_scaler_scale_,
            "robust_scale_min": robust_scale_min,
            "robust_scale_max": robust_scale_max,
        })
    else:
        # 保存分位数归一化参数
        preprocess_data.update({
            "quantile_sorted_data": quantile_params['sorted_data'],
            "quantile_quantiles": quantile_params['quantiles'],
        })

    # 保存为npz文件
    npz_file = output_preprocess_dir / "preprocess_data.npz"
    np.savez(
        npz_file,
        **preprocess_data
    )

    print(f"预处理完成！")
    print(f"特征形状: {features.shape}")
    print(f"行为ID形状: {selected_behavior_ids.shape}")
    print(f"行为数量: {num_behaviors}")
    print(f"预处理数据已保存到: {npz_file}")

    return npz_file


def cleanup_old_files(directory, pattern):
    """清理目录下的旧文件，只保留最新的KEEP_LATEST_FILES个文件

    Args:
        directory: 要清理的目录
        pattern: 要清理的文件模式
    """
    from config import CLEANUP_OLD_FILES, KEEP_LATEST_FILES

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
        print(f"已清理旧文件: {file}")


def main():
    """主函数入口

    解析命令行参数，调用preprocess_data函数预处理扩散模型训练数据。

    命令行参数：
        python scripts/step2_0_preprocess_data.py [input_patterns_dir] [input_processed_file_or_dir] [output_preprocess_dir]

    参数说明：
        input_patterns_dir: 行为模式目录路径，包含行为标签和合法丢包值 (默认: data/results/patterns)
        input_processed_file_or_dir: 处理后的数据文件或目录路径 (默认: data/processed)
        output_preprocess_dir: 预处理结果的输出目录路径 (默认: data/results/preprocess)
    """
    import argparse
    from config import PATTERNS_DIR, PROCESSED_DIR, PREPROCESS_DIR

    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="预处理扩散模型训练数据")
    parser.add_argument(
        "input_patterns",
        nargs="?",
        type=Path,
        default=PATTERNS_DIR,
        help="行为模式目录路径，包含行为标签和合法丢包值 (默认: data/results/patterns)",
    )
    parser.add_argument(
        "input_processed",
        nargs="?",
        type=Path,
        default=PROCESSED_DIR,
        help="处理后的数据文件或目录路径 (默认: data/processed)",
    )
    parser.add_argument(
        "output_preprocess",
        nargs="?",
        type=Path,
        default=PREPROCESS_DIR,
        help="预处理结果的输出目录路径 (默认: data/results/preprocess)",
    )

    args = parser.parse_args()

    input_patterns_dir = args.input_patterns
    input_processed_path = args.input_processed
    output_preprocess_dir = args.output_preprocess

    # 确保输出目录存在
    output_preprocess_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧文件
    cleanup_old_files(output_preprocess_dir, "preprocess_data.npz")
    cleanup_old_files(output_preprocess_dir, "merged_processed_data.csv")

    # 预处理数据
    if input_processed_path.is_file():
        # 如果输入是一个文件，处理单个文件
        preprocess_data(input_patterns_dir, input_processed_path, output_preprocess_dir)
    elif input_processed_path.is_dir():
        # 如果输入是一个文件夹，合并所有处理后的文件
        processed_files = list(input_processed_path.glob("*.csv"))
        if not processed_files:
            print(f"警告: 在 {input_processed_path} 中未找到 .csv 文件")
            sys.exit(1)

        # 合并所有处理后的文件
        all_processed_df = []
        for processed_file in processed_files:
            df = pd.read_csv(processed_file, parse_dates=["timestamp"])
            all_processed_df.append(df)

        # 合并为一个DataFrame
        merged_processed_df = pd.concat(all_processed_df, ignore_index=True)
        print(
            f"合并了 {len(processed_files)} 个处理后的文件，总样本数: {len(merged_processed_df)}"
        )

        # 保存合并后的文件
        merged_file_path = output_preprocess_dir / "merged_processed_data.csv"
        merged_processed_df.to_csv(merged_file_path, index=False)

        print(f"\n正在处理行为模式和合并后的数据...")
        # 预处理合并后的数据
        preprocess_data(input_patterns_dir, merged_file_path, output_preprocess_dir)
    else:
        print(f"错误: 输入 {input_processed_path} 不是文件或目录")
        sys.exit(1)

    print("步骤2.0：预处理训练数据完成！")


if __name__ == "__main__":
    main()
