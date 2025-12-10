#!/usr/bin/env python3
"""
样本生成模块
负责使用训练好的条件扩散模型生成网络状态样本
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from network_simulation.condition_generation.diffusion_model import (
    ConditionDiffusionModel,
)
from network_simulation.condition_generation.normalization import Normalizer
from network_simulation.condition_generation.constraint_injector import (
    ConstraintInjector,
)
from network_simulation.utils.logger import get_logger
from config import (
    DEFAULT_WINDOW_SIZE,
    DEFAULT_STRIDE,
    DEFAULT_SAMPLE_LENGTH,
    DEFAULT_NUM_GROUPS,
)

# 获取日志记录器
logger = get_logger(__name__)


class SampleGenerator:
    """样本生成器类

    该类使用训练好的条件扩散模型生成网络状态样本数据。
    """

    def __init__(self):
        """初始化样本生成器类"""
        logger.info("初始化样本生成器类")

        # 从配置文件加载参数
        self.window_size = DEFAULT_WINDOW_SIZE
        self.stride = DEFAULT_STRIDE
        self.sample_length = DEFAULT_SAMPLE_LENGTH
        self.num_groups = DEFAULT_NUM_GROUPS

        # 设备设置
        self.device = self._get_device()
        logger.info(f"使用设备: {self.device}")

        # 初始化约束注入器
        self.constraint_injector = None

    def _get_device(self) -> torch.device:
        """获取可用的计算设备

        Returns:
            可用的计算设备（MPS/CUDA/CPU）
        """
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def load_valid_loss_values(self, input_patterns_dir: Path) -> tuple:
        """加载上下行独立的合法丢包值

        尝试从多个位置加载valid_loss_values.json文件，包括：
        1. input_patterns_dir/metadata/valid_loss_values_up.json 和 valid_loss_values_down.json（首选）
        2. input_patterns_dir/metadata/valid_loss_values.json（兼容旧版本）
        3. input_patterns_dir.parent/features/valid_loss_values.json
        4. input_patterns_dir.parent/features/merged_valid_loss_values.json

        Args:
            input_patterns_dir: 行为模式目录路径

        Returns:
            tuple: 包含两个列表的元组，分别为上下行合法丢包值

        Raises:
            FileNotFoundError: 如果在所有位置都找不到valid_loss_values.json文件
        """
        # 尝试加载上下行独立的合法丢包值
        metadata_dir = input_patterns_dir / "metadata"
        up_file = metadata_dir / "valid_loss_values_up.json"
        down_file = metadata_dir / "valid_loss_values_down.json"

        if up_file.exists() and down_file.exists():
            with open(up_file) as f:
                valid_loss_up = json.load(f)
            with open(down_file) as f:
                valid_loss_down = json.load(f)
            logger.info(
                f"加载上下行独立合法丢包值: 上行 {valid_loss_up}, 下行 {valid_loss_down}"
            )
            return valid_loss_up, valid_loss_down
        else:
            # 兼容旧版本，加载合并的合法丢包值
            valid_loss_values_file = metadata_dir / "valid_loss_values.json"
            if not valid_loss_values_file.exists():
                # 尝试从features目录加载（优化流水线的情况）
                features_dir = input_patterns_dir.parent / "features"
                valid_loss_values_file = features_dir / "valid_loss_values.json"
                if not valid_loss_values_file.exists():
                    # 尝试从features目录加载合并后的合法丢包值（原始流水线的情况）
                    valid_loss_values_file = (
                        features_dir / "merged_valid_loss_values.json"
                    )
                    if not valid_loss_values_file.exists():
                        raise FileNotFoundError(
                            f"在 {input_patterns_dir} 及其父目录中未找到 valid_loss_values.json 文件"
                        )

            with open(valid_loss_values_file, "r") as f:
                valid_loss_values = json.load(f)

            logger.info(f"加载合并合法丢包值: {valid_loss_values}")
            return valid_loss_values, valid_loss_values

    def load_behavior_labels(self, input_patterns_dir: Path) -> np.ndarray:
        """加载行为标签

        从input_patterns_dir目录中查找并加载行为标签文件，支持JSON和NPY格式。
        优先寻找JSON文件，兼容旧版本；如果找不到JSON文件，寻找NPY文件。

        Args:
            input_patterns_dir: 行为模式目录路径

        Returns:
            np.ndarray: 行为标签数组
        """
        # 优先寻找JSON文件
        json_files = list(input_patterns_dir.glob("behavior_labels_*.json"))
        if json_files:
            with open(json_files[0], "r") as f:
                behavior_labels_data = json.load(f)
            behavior_ids = np.array(behavior_labels_data["labels"])
            return behavior_ids
        else:
            # 寻找NPY文件，优先使用上行标签
            npy_files = list(input_patterns_dir.glob("labels_*.npy"))
            if not npy_files:
                raise FileNotFoundError(f"在 {input_patterns_dir} 中未找到行为标签文件")

            # 优先使用上行标签
            up_files = [f for f in npy_files if "up" in f.name]
            if up_files:
                return np.load(up_files[0])
            else:
                # 使用第一个找到的标签文件
                return np.load(npy_files[0])

    def expand_behavior_labels(
        self, behavior_ids: np.ndarray, processed_df: pd.DataFrame
    ) -> np.ndarray:
        """扩展行为标签到样本级别

        将窗口级别的行为标签扩展到每个样本点，确保每个时间点都有对应的行为标签。

        Args:
            behavior_ids: 窗口级别的行为标签数组
            processed_df: 处理后的数据DataFrame

        Returns:
            np.ndarray: 样本级别的行为标签数组
        """
        num_windows = len(behavior_ids)

        # 初始化行为标签数组，-2表示未分配
        expanded_behavior_ids = np.full(len(processed_df), -2, dtype=int)

        # 直接为每个时间窗口分配对应的行为标签
        for i in range(num_windows):
            window_start = i * self.stride
            window_end = window_start + self.window_size

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
        last_window_start = (num_windows - 1) * self.stride
        if -2 in expanded_behavior_ids[last_window_start:]:
            expanded_behavior_ids[last_window_start:] = behavior_ids[-1]

        return expanded_behavior_ids

    def _calculate_dtw_distance(self, a, b) -> float:
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
        dtw = np.zeros((n + 1, m + 1))
        dtw[:, 0] = np.inf
        dtw[0, :] = np.inf
        dtw[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = euclidean([a[i - 1]], [b[j - 1]])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

        return dtw[n, m]

    def _collect_label_sequences(
        self,
        expanded_behavior_ids: np.ndarray,
        processed_df: pd.DataFrame,
        unique_non_noise_labels: list,
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
                # 提取这些样本的delay1和loss_rate1序列（上行数据）
                label_sequence = processed_df.iloc[sample_indices][
                    ["delay1", "loss_rate1"]
                ].values.flatten()
                label_sequences[label] = label_sequence
        return label_sequences

    def _find_closest_label(
        self,
        noise_sequence: np.ndarray,
        label_sequences: dict,
        unique_non_noise_labels: list,
    ) -> int:
        """为噪声序列找到最接近的非噪声标签

        Args:
            noise_sequence: 噪声序列
            label_sequences: 包含每个标签及其代表序列的字典
            unique_non_noise_labels: 非噪声标签列表

        Returns:
            int: 最接近的非噪声标签
        """
        min_distance = float("inf")
        closest_label = unique_non_noise_labels[0]

        for label, sequence in label_sequences.items():
            # 确保序列长度相同
            if len(sequence) > len(noise_sequence):
                sequence = sequence[: len(noise_sequence)]
            elif len(sequence) < len(noise_sequence):
                # 填充较短的序列
                sequence = np.pad(
                    sequence, (0, len(noise_sequence) - len(sequence)), "constant"
                )

            distance = self._calculate_dtw_distance(noise_sequence, sequence)
            if distance < min_distance:
                min_distance = distance
                closest_label = label

        return closest_label

    def fix_noise_labels(
        self, expanded_behavior_ids: np.ndarray, processed_df: pd.DataFrame
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
        label_sequences = self._collect_label_sequences(
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
            # 提取这些样本的delay1和loss_rate1序列（上行数据）
            noise_sequence = processed_df.iloc[start:end][
                ["delay1", "loss_rate1"]
            ].values.flatten()

            # 计算与每个非噪声标签的DTW距离
            closest_label = self._find_closest_label(
                noise_sequence, label_sequences, unique_non_noise_labels
            )

            # 替换噪声标签为最近的非噪声标签
            processed_behavior_ids[i] = closest_label

        logger.info("噪声标签替换完成")
        return processed_behavior_ids

    def _find_mixed_sample_groups(
        self,
        valid_processed_df: pd.DataFrame,
        valid_expanded_behavior_ids: np.ndarray,
        check_length: int = 100,
    ) -> list:
        """寻找包含多个类别的样本组

        Args:
            valid_processed_df: 处理后的数据DataFrame
            valid_expanded_behavior_ids: 扩展后的行为标签数组
            check_length: 检查长度

        Returns:
            list: 包含混合类别样本组的列表
        """
        mixed_samples_found = []

        # 遍历所有可能的起始索引，寻找混合类别样本组
        for idx in range(
            0, len(valid_processed_df) - self.sample_length + 1, self.sample_length // 2
        ):  # 步长减半，增加找到的概率
            # 检查更长的序列，提高多类别检测的准确性
            check_longer_length = 200
            if idx + check_longer_length <= len(valid_expanded_behavior_ids):
                current_sequence = valid_expanded_behavior_ids[
                    idx : idx + check_longer_length
                ]
            else:
                current_sequence = valid_expanded_behavior_ids[idx : idx + check_length]

            # 统计当前序列的主要行为类别（出现次数最多的行为类别）
            unique_current, counts_current = np.unique(
                current_sequence, return_counts=True
            )
            main_behavior = unique_current[np.argmax(counts_current)]
            main_behavior_ratio = np.max(counts_current) / len(current_sequence)

            # 检查是否包含多个类别
            if len(unique_current) >= 2:  # 至少包含2个不同类别
                mixed_samples_found.append(
                    {
                        "idx": idx,
                        "unique_count": len(unique_current),
                        "main_behavior": main_behavior,
                        "main_behavior_ratio": main_behavior_ratio,
                        "unique_current": unique_current,
                    }
                )

        # 按照包含的类别数量排序，优先选择包含更多类别的样本组
        mixed_samples_found.sort(key=lambda x: x["unique_count"], reverse=True)

        return mixed_samples_found

    def _find_remaining_sample_groups(
        self,
        valid_processed_df: pd.DataFrame,
        valid_expanded_behavior_ids: np.ndarray,
        check_length: int,
        start_indices: list,
        group_behavior_types: list,
    ) -> tuple:
        """寻找剩余样本组

        Args:
            valid_processed_df: 处理后的数据DataFrame
            valid_expanded_behavior_ids: 扩展后的行为标签数组
            check_length: 检查长度
            start_indices: 已找到的起始索引列表
            group_behavior_types: 已找到的组行为类型列表

        Returns:
            tuple: 包含起始索引列表和组行为类型列表的元组
        """
        # 遍历所有可能的起始索引，步长为sample_length，避免重叠
        for idx in range(
            0, len(valid_processed_df) - self.sample_length + 1, self.sample_length
        ):
            if len(start_indices) >= self.num_groups:
                break

            # 只检查前100个行为ID，加速寻找过程
            current_sequence = valid_expanded_behavior_ids[idx : idx + check_length]

            # 统计当前序列的主要行为类别（出现次数最多的行为类别）
            unique_current, counts_current = np.unique(
                current_sequence, return_counts=True
            )
            main_behavior = unique_current[np.argmax(counts_current)]
            main_behavior_ratio = np.max(counts_current) / len(current_sequence)

            # 检查是否包含多个类别
            if len(unique_current) >= 2:  # 至少包含2个不同类别
                # 避免重复的主要行为类别
                if main_behavior not in [
                    t for t in group_behavior_types if not t.startswith("mixed_")
                ]:
                    start_indices.append(idx)
                    group_behavior_types.append(f"mixed_{main_behavior}")
                    logger.info(
                        f"找到混合类别样本组 - 起始索引: {idx}, 包含 {len(unique_current)} 个类别, 主要行为: {main_behavior}, 比例: {main_behavior_ratio:.2f}"
                    )
            else:
                # 避免重复的主要行为类别
                if main_behavior not in [
                    t for t in group_behavior_types if not t.startswith("mixed_")
                ]:
                    start_indices.append(idx)
                    group_behavior_types.append(main_behavior)
                    logger.info(
                        f"找到样本组 - 起始索引: {idx}, 主要行为类别: {main_behavior}, 比例: {main_behavior_ratio:.2f}"
                    )

        return start_indices, group_behavior_types

    def _find_additional_sample_groups(
        self,
        valid_processed_df: pd.DataFrame,
        valid_expanded_behavior_ids: np.ndarray,
        check_length: int,
        start_indices: list,
        group_behavior_types: list,
    ) -> tuple:
        """使用额外策略寻找样本组

        Args:
            valid_processed_df: 处理后的数据DataFrame
            valid_expanded_behavior_ids: 扩展后的行为标签数组
            check_length: 检查长度
            start_indices: 已找到的起始索引列表
            group_behavior_types: 已找到的组行为类型列表

        Returns:
            tuple: 包含起始索引列表和组行为类型列表的元组
        """
        # 遍历所有可能的起始索引，找到不同的序列
        for idx in range(
            0, len(valid_processed_df) - self.sample_length + 1, self.sample_length // 2
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
            logger.info(
                f"找到额外样本组 - 起始索引: {idx}, 主要行为类别: {main_behavior}"
            )

            # 如果已经找到了足够的样本组，停止寻找
            if len(start_indices) >= self.num_groups:
                break

        return start_indices, group_behavior_types

    def _find_random_sample_groups(
        self,
        valid_processed_df: pd.DataFrame,
        start_indices: list,
        group_behavior_types: list,
    ) -> tuple:
        """使用随机选择补充样本组

        Args:
            valid_processed_df: 处理后的数据DataFrame
            start_indices: 已找到的起始索引列表
            group_behavior_types: 已找到的组行为类型列表

        Returns:
            tuple: 包含起始索引列表和组行为类型列表的元组
        """
        import random

        # 只尝试10次随机选择，避免无限循环
        max_attempts = 10
        attempts = 0

        while len(start_indices) < self.num_groups and attempts < max_attempts:
            # 随机选择一个起始索引
            random_idx = random.randint(0, len(valid_processed_df) - self.sample_length)

            # 添加到列表中
            start_indices.append(random_idx)
            group_behavior_types.append(1)  # 默认行为类别
            logger.info(f"随机选择样本组 - 起始索引: {random_idx}, 主要行为类别: 1")

            attempts += 1

        return start_indices, group_behavior_types

    def select_sample_groups(
        self, valid_processed_df: pd.DataFrame, valid_expanded_behavior_ids: np.ndarray
    ) -> tuple:
        """选择样本组

        找到多组真实连续序列，每组对应不同的主要行为类别。

        Args:
            valid_processed_df: 处理后的数据DataFrame
            valid_expanded_behavior_ids: 扩展后的行为标签数组

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
        mixed_samples_found = self._find_mixed_sample_groups(
            valid_processed_df, valid_expanded_behavior_ids, check_length
        )

        # 选择前num_groups个样本组
        for sample in mixed_samples_found:
            if len(start_indices) >= self.num_groups:
                break

            # 避免重复的主要行为类别
            if sample["main_behavior"] not in [
                t for t in group_behavior_types if not t.startswith("mixed_")
            ]:
                start_indices.append(sample["idx"])
                group_behavior_types.append(f"mixed_{sample['main_behavior']}")
                logger.info(
                    f"找到多样化样本组 - 起始索引: {sample['idx']}, 包含 {sample['unique_count']} 个类别, 主要行为: {sample['main_behavior']}, 比例: {sample['main_behavior_ratio']:.2f}"
                )

        # 如果找不到足够的混合类别样本组，使用原来的逻辑寻找样本组
        if len(start_indices) < self.num_groups:
            logger.info(
                f"只找到 {len(start_indices)} 个混合类别样本组，使用原始逻辑寻找剩余样本组..."
            )
            start_indices, group_behavior_types = self._find_remaining_sample_groups(
                valid_processed_df,
                valid_expanded_behavior_ids,
                check_length,
                start_indices,
                group_behavior_types,
            )

        # 如果找到的样本组不足，使用额外的策略
        if len(start_indices) < self.num_groups:
            logger.info(
                f"只找到 {len(start_indices)} 组样本，使用额外策略寻找剩余样本组..."
            )
            start_indices, group_behavior_types = self._find_additional_sample_groups(
                valid_processed_df,
                valid_expanded_behavior_ids,
                check_length,
                start_indices,
                group_behavior_types,
            )

        # 如果仍然找不到足够的样本组，使用随机选择
        if len(start_indices) < self.num_groups:
            logger.info(f"仍然只找到 {len(start_indices)} 组样本，使用随机选择补充...")
            start_indices, group_behavior_types = self._find_random_sample_groups(
                valid_processed_df, start_indices, group_behavior_types
            )

        return start_indices, group_behavior_types

    def load_model(self, input_model_path: Path) -> tuple:
        """加载模型

        加载训练好的条件扩散模型，并返回模型实例、行为映射和归一化参数。

        Args:
            input_model_path: 训练好的扩散模型文件路径

        Returns:
            tuple: 包含模型实例、行为映射和归一化参数的元组
        """
        logger.info(f"正在加载模型: {input_model_path}")
        checkpoint = torch.load(
            input_model_path, map_location=self.device, weights_only=False
        )
        behavior_mapping = checkpoint["behavior_mapping"]
        num_behaviors = len(behavior_mapping)

        # 初始化条件扩散模型，使用4D输入（上下行延迟和丢包率）
        model = ConditionDiffusionModel(
            input_dim=4,  # 输入维度：上行延迟、上行丢包率、下行延迟、下行丢包率
            num_behaviors=num_behaviors,
            behavior_embed_dim=32,  # 行为嵌入维度
            T=1000,  # 扩散步数
        )

        # 加载模型状态并设置为评估模式（添加strict=False处理架构变化）
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.to(self.device)
        model.eval()

        # 从checkpoint中加载反归一化参数
        # 检查是否使用对数变换
        use_log_transform = checkpoint.get("use_log_transform", False)
        # 将numpy布尔值转换为Python布尔值
        if isinstance(use_log_transform, np.ndarray):
            use_log_transform = use_log_transform.item()

        # 打印反归一化参数，用于调试
        logger.info("\n反归一化参数：")
        normalization_method = "robust"  # 固定使用robust归一化方法
        logger.info(f"归一化方法: {normalization_method}")
        logger.info(f"是否使用对数变换: {use_log_transform}")

        # 加载robust归一化参数
        delay_scaler_center_ = checkpoint[
            "delay_scaler_center_"
        ]  # 注意：这里存储的是center_而不是mean_
        delay_scaler_scale_ = checkpoint["delay_scaler_scale_"]
        robust_scale_min = checkpoint.get("robust_scale_min", -1.0)
        robust_scale_max = checkpoint.get("robust_scale_max", 1.0)

        # 加载clipped_min和clipped_max参数
        clipped_min = checkpoint.get("clipped_min", -5.0)
        clipped_max = checkpoint.get("clipped_max", 5.0)

        logger.info(f"delay_scaler_center_: {delay_scaler_center_}")
        logger.info(f"delay_scaler_scale_: {delay_scaler_scale_}")
        logger.info(f"robust_scale_min: {robust_scale_min}")
        logger.info(f"robust_scale_max: {robust_scale_max}")
        logger.info(f"clipped_min: {clipped_min}")
        logger.info(f"clipped_max: {clipped_max}")

        # 构建robust归一化参数
        delay_norm_params = {
            "normalization_method": normalization_method,
            "use_log_transform": use_log_transform,
            "delay_scaler_center_": delay_scaler_center_,
            "delay_scaler_scale_": delay_scaler_scale_,
            "robust_scale_min": robust_scale_min,
            "robust_scale_max": robust_scale_max,
            "clipped_min": clipped_min,
            "clipped_max": clipped_max,
        }

        return model, behavior_mapping, delay_norm_params

    def generate_single_sample_group(
        self,
        i: int,
        start_idx: int,
        group_behavior_type: any,
        valid_processed_df: pd.DataFrame,
        valid_expanded_behavior_ids: np.ndarray,
        behavior_mapping: dict,
        model: ConditionDiffusionModel,
        delay_norm_params: dict,
        loss_norm_params: dict,
        output_generation_dir: Path,
    ) -> tuple:
        """生成单个样本组

        为指定的起始索引生成对应的网络状态数据。

        Args:
            i: 样本组索引
            start_idx: 起始索引
            group_behavior_type: 组行为类型
            valid_processed_df: 处理后的数据DataFrame
            valid_expanded_behavior_ids: 扩展后的行为标签数组
            behavior_mapping: 行为映射字典
            model: 条件扩散模型实例
            delay_norm_params: 延迟归一化参数
            loss_norm_params: 丢包率归一化参数
            output_generation_dir: 生成样本的输出目录路径

        Returns:
            tuple: 包含原始样本文件路径和生成样本文件路径的元组，
                  如果生成失败则返回(None, None)
        """
        logger.info(f"\n处理样本组 {i + 1}/{self.num_groups}...")
        logger.info(f"起始索引: {start_idx}, 主要行为类别: {group_behavior_type}")

        # 选择当前组的样本
        selected_df = valid_processed_df.iloc[
            start_idx : start_idx + self.sample_length
        ].copy()
        selected_behavior_ids = valid_expanded_behavior_ids[
            start_idx : start_idx + self.sample_length
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
                closest_behavior = min(
                    model_behavior_keys, key=lambda x: abs(x - behavior_id)
                )
                model_behavior_ids.append(behavior_mapping[closest_behavior])
                logger.warning(
                    f"行为ID {behavior_id} 不在模型映射中，已转换为最接近的行为ID {closest_behavior}"
                )

        # 确保样本长度符合要求
        if len(selected_behavior_ids) < self.sample_length:
            logger.warning(f"样本组 {i + 1} 样本不足 {self.sample_length}，跳过")
            return None, None

        # 转换行为ID为张量，使用转换后的model_behavior_ids
        behavior_ids_tensor = (
            torch.tensor(model_behavior_ids, dtype=torch.long)
            .unsqueeze(0)
            .to(self.device)
        )

        # 生成样本
        logger.info("正在生成网络状态数据...")
        with torch.no_grad():
            generated = model.sample(behavior_ids_tensor)

        # 转换回numpy数组
        generated = generated.cpu().numpy()[0]

        # 获取上下行合法丢包值
        valid_loss_up = loss_norm_params.get(
            "valid_loss_values_up", loss_norm_params["valid_loss_values"]
        )
        valid_loss_down = loss_norm_params.get(
            "valid_loss_values_down", loss_norm_params["valid_loss_values"]
        )

        # 初始化归一化器并加载归一化参数，使用上下行独立的合法丢包值
        normalizer = Normalizer(valid_loss_up, valid_loss_down)

        # 构建归一化参数字典，支持上下行独立参数
        normalization_params = {
            "delay_up": delay_norm_params,
            "delay_down": delay_norm_params,  # 暂时使用相同的延迟参数，后续可扩展为上下行独立
            "loss_rate_up": {"valid_loss_values": valid_loss_up},
            "loss_rate_down": {"valid_loss_values": valid_loss_down},
        }
        normalizer.normalization_params = normalization_params

        # 检查生成数据的维度，支持4D或2D输入
        is_4d = generated.shape[1] == 4
        logger.info(
            f"检测到 {'4D' if is_4d else '2D'} 输入，按{'双流' if is_4d else '单流复制'}处理"
        )

        if is_4d:
            # 4D 双流处理
            delay1, loss1, delay2, loss2 = normalizer.denormalize4d(
                generated[:, 0], generated[:, 1], generated[:, 2], generated[:, 3]
            )

            # 确保延迟非负
            delay1 = np.clip(delay1, a_min=0, a_max=None)
            delay2 = np.clip(delay2, a_min=0, a_max=None)

            # 验证生成的数据是否符合网络约束
            validation_up = self.constraint_injector.validate_sequence(
                delay1, loss1, direction="up"
            )
            validation_down = self.constraint_injector.validate_sequence(
                delay2, loss2, direction="down"
            )

            # 打印反归一化后的统计，用于调试
            logger.info(
                f"反归一化后统计 - 延迟1: 最小值: {delay1.min():.4f}, 最大值: {delay1.max():.4f}, 平均值: {delay1.mean():.4f}, 标准差: {delay1.std():.4f}"
            )
            logger.info(
                f"反归一化后统计 - 丢包率1: 最小值: {loss1.min():.4f}, 最大值: {loss1.max():.4f}, 平均值: {loss1.mean():.4f}, 标准差: {loss1.std():.4f}"
            )
            logger.info(
                f"反归一化后统计 - 延迟2: 最小值: {delay2.min():.4f}, 最大值: {delay2.max():.4f}, 平均值: {delay2.mean():.4f}, 标准差: {delay2.std():.4f}"
            )
            logger.info(
                f"反归一化后统计 - 丢包率2: 最小值: {loss2.min():.4f}, 最大值: {loss2.max():.4f}, 平均值: {loss2.mean():.4f}, 标准差: {loss2.std():.4f}"
            )
        else:
            # 2D 单流处理，复制到双流
            delay_norm = generated[:, 0]
            loss_norm = generated[:, 1]
            delay, loss_rate = normalizer.denormalize(delay_norm, loss_norm)

            # 确保延迟非负
            delay = np.clip(delay, a_min=0, a_max=None)

            # 复制到双流
            delay1 = delay
            loss1 = loss_rate
            delay2 = delay.copy()
            loss2 = loss_rate.copy()

            # 验证生成的数据是否符合网络约束
            validation_up = self.constraint_injector.validate_sequence(
                delay1, loss1, direction="up"
            )
            validation_down = self.constraint_injector.validate_sequence(
                delay2, loss2, direction="down"
            )

            # 打印反归一化后的统计，用于调试
            logger.info(
                f"反归一化后统计 - 延迟1: 最小值: {delay1.min():.4f}, 最大值: {delay1.max():.4f}, 平均值: {delay1.mean():.4f}, 标准差: {delay1.std():.4f}"
            )
            logger.info(
                f"反归一化后统计 - 丢包率1: 最小值: {loss1.min():.4f}, 最大值: {loss1.max():.4f}, 平均值: {loss1.mean():.4f}, 标准差: {loss1.std():.4f}"
            )
            logger.info(
                f"反归一化后统计 - 延迟2: 最小值: {delay2.min():.4f}, 最大值: {delay2.max():.4f}, 平均值: {delay2.mean():.4f}, 标准差: {delay2.std():.4f}"
            )
            logger.info(
                f"反归一化后统计 - 丢包率2: 最小值: {loss2.min():.4f}, 最大值: {loss2.max():.4f}, 平均值: {loss2.mean():.4f}, 标准差: {loss2.std():.4f}"
            )

        logger.info(
            f"生成数据验证结果 - 上行: {validation_up}, 下行: {validation_down}"
        )

        # 创建生成的数据Frame，包含上下行数据
        generated_df = selected_df.copy()
        generated_df["delay1"] = delay1
        generated_df["loss_rate1"] = loss1
        generated_df["delay2"] = delay2
        generated_df["loss_rate2"] = loss2

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
        self,
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
            tuple: 包含原始样本文件路径列表和生成样本文件路径列表的元组

        Raises:
            FileNotFoundError: 如果输入文件或目录不存在
            ValueError: 如果数据格式不符合要求
        """
        logger.info(f"正在使用模型生成样本: {input_model_path}")

        # 加载上下行独立的合法丢包值
        valid_loss_up, valid_loss_down = self.load_valid_loss_values(input_patterns_dir)

        # 初始化约束注入器：确保生成的丢包率符合合法范围
        self.constraint_injector = ConstraintInjector(valid_loss_up, valid_loss_down)

        # 加载处理后的数据：包含原始网络状态数据
        processed_df = pd.read_csv(input_processed_file, parse_dates=["timestamp"])

        # 加载行为标签
        behavior_ids = self.load_behavior_labels(input_patterns_dir)

        # 扩展行为标签到样本级别（使用10秒窗口，50%重叠，与预处理一致）
        expanded_behavior_ids = self.expand_behavior_labels(behavior_ids, processed_df)

        # 优化噪声标签处理，使用DTW距离最近的标签替换噪声标签
        processed_behavior_ids = self.fix_noise_labels(
            expanded_behavior_ids, processed_df
        )

        # 保存处理后的行为标签和对应的处理后数据
        valid_expanded_behavior_ids = processed_behavior_ids
        valid_processed_df = processed_df.copy()

        # 确保valid_processed_df有足够的样本
        if len(valid_processed_df) < self.sample_length:
            logger.warning(f"有效样本数不足 {self.sample_length}，无法生成样本")
            return None, None

        # 选择样本组
        start_indices, group_behavior_types = self.select_sample_groups(
            valid_processed_df, valid_expanded_behavior_ids
        )

        # 为每组样本生成对应的网络状态数据
        original_files = []
        generated_files = []

        # 加载模型
        model, behavior_mapping, delay_norm_params = self.load_model(input_model_path)

        # 构建丢包率归一化参数字典，包含上下行独立的合法丢包值
        loss_norm_params = {
            "valid_loss_values": valid_loss_up,  # 兼容旧版本
            "valid_loss_values_up": valid_loss_up,
            "valid_loss_values_down": valid_loss_down,
        }

        # 为每组样本生成数据
        for i, (start_idx, group_behavior_type) in enumerate(
            zip(start_indices, group_behavior_types)
        ):
            original_file, generated_file = self.generate_single_sample_group(
                i,
                start_idx,
                group_behavior_type,
                valid_processed_df,
                valid_expanded_behavior_ids,
                behavior_mapping,
                model,
                delay_norm_params,
                loss_norm_params,
                output_generation_dir,
            )

            if original_file and generated_file:
                original_files.append(original_file)
                generated_files.append(generated_file)

        return original_files, generated_files
