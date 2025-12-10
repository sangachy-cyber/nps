#!/usr/bin/env python3
"""
训练数据预处理模块
负责将处理后的数据和行为标签转换为适合扩散模型训练的格式
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from network_simulation.utils.logger import get_logger
from config import DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE

# 获取日志记录器
logger = get_logger(__name__)


class TrainingDataPreprocessor:
    """训练数据预处理类

    该类对网络数据进行预处理，包括行为标签对齐、噪声剔除、特征归一化等，
    为扩散模型训练准备格式化的输入数据。
    """

    def __init__(self):
        """初始化训练数据预处理类"""
        logger.info("初始化训练数据预处理类")

        # 从配置文件加载窗口参数
        self.window_size = DEFAULT_WINDOW_SIZE  # 从配置文件加载
        self.stride = DEFAULT_STRIDE  # 从配置文件加载

    def preprocess_data(
        self, input_patterns_dir: Path, input_processed_file: Path, output_preprocess_dir: Path
    ) -> Path:
        """预处理扩散模型训练数据

        该函数对上下行网络数据进行预处理，为扩散模型训练准备格式化的输入数据。
        主要处理步骤包括：
        1. 加载并验证输入文件
        2. 加载窗口特征矩阵
        3. 从原始数据文件提取窗口数据
        4. 归一化上下行延迟和丢包率数据
        5. 格式化4D特征数据
        6. 保存预处理结果

        Args:
            input_patterns_dir: 行为模式目录路径，包含window_features_rule.npy
            input_processed_file: 处理后的数据文件路径，包含timestamp、delay1、loss_rate1、delay2、loss_rate2等字段
            output_preprocess_dir: 预处理结果的输出目录路径

        Returns:
            Path: 预处理数据文件路径

        Raises:
            FileNotFoundError: 如果输入文件或目录不存在
            ValueError: 如果数据格式不符合要求
        """
        logger.info(f"正在为扩散模型预处理数据: {input_patterns_dir}")

        # 1. 加载窗口特征矩阵
        window_features_file = input_patterns_dir / "window_features_rule.npy"
        if not window_features_file.exists():
            raise FileNotFoundError(f"在 {input_patterns_dir} 中未找到 window_features_rule.npy 文件")
        window_features = np.load(window_features_file)
        num_windows = window_features.shape[0]
        logger.info(f"加载窗口特征矩阵，形状: {window_features.shape}, 窗口数量: {num_windows}")

        # 2. 加载上下行独立的合法丢包值
        valid_loss_values_up = None
        valid_loss_values_down = None

        # 优先加载上下行独立的合法丢包值
        valid_loss_up_file = input_patterns_dir / "metadata" / "valid_loss_values_up.json"
        valid_loss_down_file = input_patterns_dir / "metadata" / "valid_loss_values_down.json"

        if valid_loss_up_file.exists() and valid_loss_down_file.exists():
            with open(valid_loss_up_file, "r") as f:
                valid_loss_values_up = json.load(f)
            with open(valid_loss_down_file, "r") as f:
                valid_loss_values_down = json.load(f)
            logger.info(f"加载上下行独立合法丢包值: 上行 {valid_loss_values_up}, 下行 {valid_loss_values_down}")
        else:
            # 兼容旧版本，加载合并的合法丢包值
            valid_loss_values_file = input_patterns_dir / "metadata" / "valid_loss_values.json"
            if valid_loss_values_file.exists():
                with open(valid_loss_values_file, "r") as f:
                    valid_loss_values = json.load(f)
                # 上下行使用相同的合法丢包值
                valid_loss_values_up = valid_loss_values
                valid_loss_values_down = valid_loss_values
                logger.info(f"加载合并合法丢包值: {valid_loss_values}")
            else:
                # 使用默认合法丢包值
                valid_loss_values_up = [0.0, 1/3, 0.5, 2/3, 1.0]
                valid_loss_values_down = [0.0, 1/3, 0.5, 2/3, 1.0]
                logger.info("使用默认合法丢包值")

        # 3. 加载处理后的数据文件
        if not input_processed_file.exists():
            raise FileNotFoundError(f"处理后的数据文件不存在: {input_processed_file}")
        logger.info(f"加载处理后的数据文件: {input_processed_file}")
        processed_df = pd.read_csv(input_processed_file, parse_dates=["timestamp"])
        logger.info(f"处理后的数据文件大小: {len(processed_df)} 行")

        # 4. 从处理后的数据中提取窗口数据
        # 初始化归一化器
        from .normalization import Normalizer
        normalizer = Normalizer(valid_loss_values_up, valid_loss_values_down)

        # 统计归一化前的数据范围
        all_delay1 = []
        all_loss1 = []
        all_delay2 = []
        all_loss2 = []

        # 计算滑动窗口数量
        total_samples = len(processed_df)
        window_samples = self.window_size
        slide_samples = self.stride
        calculated_num_windows = (total_samples - window_samples) // slide_samples + 1
        logger.info(f"根据滑动窗口计算的窗口数量: {calculated_num_windows}")

        # 使用较小的窗口数量
        actual_num_windows = min(num_windows, calculated_num_windows)
        if actual_num_windows != num_windows:
            logger.warning(f"特征矩阵窗口数量 {num_windows} 大于计算的窗口数量 {calculated_num_windows}，使用 {actual_num_windows} 个窗口")
            window_features = window_features[:actual_num_windows]

        # 生成所有窗口的数据
        valid_windows = []
        for i in range(actual_num_windows):
            # 计算当前窗口的起始和结束索引
            start_idx = i * slide_samples
            end_idx = start_idx + window_samples

            # 检查窗口有效性
            if end_idx > total_samples:
                logger.warning(f"窗口 {i} 超出数据范围，跳过")
                continue

            # 提取窗口数据
            window_df = processed_df.iloc[start_idx:end_idx]

            # 检查是否包含必要的列
            required_cols = {"delay1", "loss_rate1", "delay2", "loss_rate2"}
            if not required_cols.issubset(window_df.columns):
                logger.warning(f"窗口 {i} 缺少必要列，当前列: {list(window_df.columns)}, 必要列: {list(required_cols)}，跳过")
                continue

            valid_windows.append(i)

            # 提取上下行数据
            delay1 = window_df["delay1"].values
            loss1 = window_df["loss_rate1"].values
            delay2 = window_df["delay2"].values
            loss2 = window_df["loss_rate2"].values

            all_delay1.extend(delay1)
            all_loss1.extend(loss1)
            all_delay2.extend(delay2)
            all_loss2.extend(loss2)

        # 检查是否有有效窗口
        if not valid_windows:
            raise ValueError("没有找到符合要求的有效窗口，无法进行预处理")

        # 将列表转换为numpy数组
        all_delay1 = np.array(all_delay1)
        all_loss1 = np.array(all_loss1)
        all_delay2 = np.array(all_delay2)
        all_loss2 = np.array(all_loss2)

        logger.info("所有窗口数据统计:")
        logger.info(f"  delay1 - 最小值: {np.min(all_delay1):.4f}, 最大值: {np.max(all_delay1):.4f}, 平均值: {np.mean(all_delay1):.4f}")
        logger.info(f"  loss1 - 最小值: {np.min(all_loss1):.4f}, 最大值: {np.max(all_loss1):.4f}, 平均值: {np.mean(all_loss1):.4f}")
        logger.info(f"  delay2 - 最小值: {np.min(all_delay2):.4f}, 最大值: {np.max(all_delay2):.4f}, 平均值: {np.mean(all_delay2):.4f}")
        logger.info(f"  loss2 - 最小值: {np.min(all_loss2):.4f}, 最大值: {np.max(all_loss2):.4f}, 平均值: {np.mean(all_loss2):.4f}")

        # 归一化所有数据
        delay1_norm, loss1_norm, delay2_norm, loss2_norm = normalizer.normalize4d(all_delay1, all_loss1, all_delay2, all_loss2)

        logger.info("归一化后数据统计:")
        logger.info(f"  delay1_norm - 最小值: {np.min(delay1_norm):.4f}, 最大值: {np.max(delay1_norm):.4f}, 平均值: {np.mean(delay1_norm):.4f}")
        logger.info(f"  loss1_norm - 最小值: {np.min(loss1_norm):.4f}, 最大值: {np.max(loss1_norm):.4f}, 平均值: {np.mean(loss1_norm):.4f}")
        logger.info(f"  delay2_norm - 最小值: {np.min(delay2_norm):.4f}, 最大值: {np.max(delay2_norm):.4f}, 平均值: {np.mean(delay2_norm):.4f}")
        logger.info(f"  loss2_norm - 最小值: {np.min(loss2_norm):.4f}, 最大值: {np.max(loss2_norm):.4f}, 平均值: {np.mean(loss2_norm):.4f}")

        # 然后将归一化后的数据拆分为窗口序列
        sequences = []
        conditions = []
        behavior_ids = []

        idx = 0
        for i in valid_windows:
            # 计算当前窗口的起始和结束索引
            start_idx = i * slide_samples
            end_idx = start_idx + window_samples
            window_size = window_samples

            # 提取当前窗口的归一化数据
            window_delay1_norm = delay1_norm[idx:idx+window_size]
            window_loss1_norm = loss1_norm[idx:idx+window_size]
            window_delay2_norm = delay2_norm[idx:idx+window_size]
            window_loss2_norm = loss2_norm[idx:idx+window_size]
            idx += window_size

            # 构造4D序列 [d1, l1, d2, l2]
            sequence = np.column_stack([window_delay1_norm, window_loss1_norm, window_delay2_norm, window_loss2_norm])
            sequences.append(sequence)

            # 获取当前窗口对应的特征向量
            condition = window_features[i]
            conditions.append(condition)

            # 使用窗口索引作为行为ID
            behavior_ids.append(i)

        # 转换为numpy数组
        sequences = np.array(sequences)
        conditions = np.array(conditions)
        behavior_ids = np.array(behavior_ids)

        logger.info("处理后数据形状:")
        logger.info(f"  序列形状: {sequences.shape}")
        logger.info(f"  条件形状: {conditions.shape}")
        logger.info(f"  行为ID形状: {behavior_ids.shape}")

        # 5. 创建输出目录
        output_preprocess_dir.mkdir(parents=True, exist_ok=True)

        # 6. 保存预处理结果
        # 获取归一化参数
        normalization_method = normalizer.normalization_params['delay_up'].get('normalization_method', 'robust')
        delay1_params = normalizer.normalization_params['delay_up']
        delay2_params = normalizer.normalization_params['delay_down']
        loss1_params = normalizer.normalization_params['loss_rate_up']
        loss2_params = normalizer.normalization_params['loss_rate_down']

        preprocess_data = {
            "sequences": sequences,  # 4D序列 [batch, seq_len, 4]
            "conditions": conditions,  # 条件向量 [batch, cond_dim]
            "behavior_ids": behavior_ids,
            "loss_rate_mapping_up": loss1_params['loss_rate_mapping'],
            "loss_rate_mapping_down": loss2_params['loss_rate_mapping'],
            "valid_loss_values_up": valid_loss_values_up,
            "valid_loss_values_down": valid_loss_values_down,
            "sample_length": len(sequences),
            "num_behaviors": len(np.unique(behavior_ids)),
            "normalization_method": normalization_method,
            # 上行归一化参数
            "delay1_scaler_center_": delay1_params['delay_scaler_center_'],
            "delay1_scaler_scale_": delay1_params['delay_scaler_scale_'],
            "delay1_robust_scale_min": delay1_params['robust_scale_min'],
            "delay1_robust_scale_max": delay1_params['robust_scale_max'],
            "delay1_clipped_min": delay1_params['clipped_min'],
            "delay1_clipped_max": delay1_params['clipped_max'],
            "delay1_use_log_transform": delay1_params.get('use_log_transform', True),
            # 下行归一化参数
            "delay2_scaler_center_": delay2_params['delay_scaler_center_'],
            "delay2_scaler_scale_": delay2_params['delay_scaler_scale_'],
            "delay2_robust_scale_min": delay2_params['robust_scale_min'],
            "delay2_robust_scale_max": delay2_params['robust_scale_max'],
            "delay2_clipped_min": delay2_params['clipped_min'],
            "delay2_clipped_max": delay2_params['clipped_max'],
            "delay2_use_log_transform": delay2_params.get('use_log_transform', True),
        }

        # 保存为npz文件
        npz_file = output_preprocess_dir / "preprocess_data.npz"
        np.savez(
            npz_file,
            **preprocess_data
        )

        logger.info("预处理完成！")
        logger.info(f"序列形状: {sequences.shape}")
        logger.info(f"条件形状: {conditions.shape}")
        logger.info(f"行为ID形状: {behavior_ids.shape}")
        logger.info(f"预处理数据已保存到: {npz_file}")

        return npz_file
