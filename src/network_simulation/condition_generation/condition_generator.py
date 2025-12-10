#!/usr/bin/env python3
"""
Condition Generation Module
Responsible for generating network simulation data based on conditions
"""

import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from ..utils.logger import get_logger
from .diffusion_model import ConditionDiffusionModel
from .constraint_injector import ConstraintInjector
from .normalization import Normalizer


logger = get_logger(__name__)

class ConditionGenerator:
    """Generates network simulation data based on behavior conditions"""

    def __init__(self, valid_loss_values_up: List[float] = None, valid_loss_values_down: List[float] = None, config: dict = None):
        self.time_granularity = 0.1  # 100ms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.valid_loss_values_up = valid_loss_values_up or [0.0, 1 / 3, 0.5, 2 / 3, 1.0]
        self.valid_loss_values_down = valid_loss_values_down or [0.0, 1 / 3, 0.5, 2 / 3, 1.0]
        self.config = config or {}

        # Initialize constraint injector，支持上下行独立参数
        self.constraint_injector = ConstraintInjector(self.valid_loss_values_up, self.valid_loss_values_down)

        # Initialize normalizer，支持上下行独立参数
        self.normalizer = Normalizer(self.valid_loss_values_up, self.valid_loss_values_down)

        # Initialize diffusion model
        self.diffusion_model = None
        self._init_diffusion_model()

    def _init_diffusion_model(self):
        """初始化扩散模型"""
        # 导入默认配置
        try:
            from ...config import (
                DEFAULT_INPUT_DIM,
                DEFAULT_BEHAVIOR_EMBED_DIM,
                DEFAULT_T
            )
            self.diffusion_model = ConditionDiffusionModel(
                input_dim=DEFAULT_INPUT_DIM,
                behavior_embed_dim=DEFAULT_BEHAVIOR_EMBED_DIM,
                T=DEFAULT_T,
            )
        except ImportError:
            # 导入失败时使用默认值
            self.diffusion_model = ConditionDiffusionModel(
                input_dim=4,
                behavior_embed_dim=32,
                T=1000,
            )
        self.diffusion_model.to(self.device)

    def load_model(self, model_path: Path):
        """
        加载模型检查点并初始化归一化参数

        Args:
            model_path: 模型检查点路径
        """
        logger.info(f"从 {model_path} 加载模型检查点")

        # 加载模型检查点
        checkpoint = torch.load(model_path, map_location=self.device)

        # 加载模型状态
        self.diffusion_model.load_state_dict(checkpoint["model_state_dict"])

        # 从检查点加载归一化参数
        self.normalizer.from_checkpoint(checkpoint)

        logger.info(f"模型已加载: {model_path}")
        logger.info("归一化参数已初始化")

    def generate(
        self, schedule: Dict, patterns: Dict, duration: int = 600
    ) -> pd.DataFrame:
        """Generate network simulation data based on schedule and patterns

        Args:
            schedule: 行为计划，包含行为类型序列或连续条件向量序列
            patterns: 行为模式定义
            duration: 生成数据的持续时间（秒）

        Returns:
            pd.DataFrame: 包含timestamp、delay1、loss_rate1、delay2、loss_rate2列的数据帧
        """
        # Calculate total samples needed
        total_samples = int(duration / self.time_granularity)

        # Initialize simulation data with dual stream columns
        simulation_data = {
            "timestamp": [],
            "delay1": [], "loss_rate1": [],  # 上行数据
            "delay2": [], "loss_rate2": []   # 下行数据
        }

        # Generate timestamp sequence
        start_time = pd.Timestamp.now()
        timestamps = [
            start_time + pd.Timedelta(seconds=i * self.time_granularity)
            for i in range(total_samples)
        ]
        simulation_data["timestamp"] = timestamps

        # Process schedule and generate data - 支持4D双流数据
        delay1_sequence, loss1_sequence, delay2_sequence, loss2_sequence = self._generate_sequences(
            schedule, patterns, total_samples
        )

        # Apply physical constraints
        delay1_sequence = np.array(delay1_sequence)
        loss1_sequence = np.array(loss1_sequence)
        delay2_sequence = np.array(delay2_sequence)
        loss2_sequence = np.array(loss2_sequence)

        # Validate and process sequences - 验证上下行数据
        validation_up = self.constraint_injector.validate_sequence(
            delay1_sequence, loss1_sequence
        )
        validation_down = self.constraint_injector.validate_sequence(
            delay2_sequence, loss2_sequence
        )

        if not validation_up["all_valid"] or not validation_down["all_valid"]:
            logger.warning(
                "Generated sequence has physical constraint violations. Applying post-processing."
            )
            # 这里可以添加后处理逻辑，但当前实现中_generate_behavior_segment已经确保了基本合理性

        # 保存生成的数据
        simulation_data["delay1"] = delay1_sequence.tolist()
        simulation_data["loss_rate1"] = loss1_sequence.tolist()
        simulation_data["delay2"] = delay2_sequence.tolist()
        simulation_data["loss_rate2"] = loss2_sequence.tolist()

        # Create dataframe
        df = pd.DataFrame(simulation_data)
        return df

    def generate_sequences(
        self, schedule: Dict, patterns: Dict, duration: int = 600
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Generate dual stream delay and loss rate sequences based on schedule

        Args:
            schedule: 行为计划，包含行为类型序列或连续条件向量序列
            patterns: 行为模式定义
            duration: 生成数据的持续时间（秒）

        Returns:
            Tuple of 4 lists: (delay1, loss1, delay2, loss2)
        """
        # Calculate total samples needed
        total_samples = int(duration / self.time_granularity)

        # Create behavior ID sequence or condition vector sequence
        if 'conditions' in schedule:
            # 使用连续条件向量序列
            condition_vectors = schedule['conditions']
            # 确保条件向量长度匹配总样本数
            if len(condition_vectors) != total_samples:
                logger.warning(f"条件向量长度 {len(condition_vectors)} 与总样本数 {total_samples} 不匹配，将进行插值")
                # 简单线性插值，可根据需要实现更复杂的插值方法
                from scipy.interpolate import interp1d

                # 假设条件向量是2D数组 (n_original, cond_dim)
                n_original = len(condition_vectors)
                x_original = np.linspace(0, total_samples-1, n_original)
                x_new = np.arange(total_samples)

                # 对每个条件维度进行插值
                condition_vectors = np.array(condition_vectors)
                cond_dim = condition_vectors.shape[1]
                condition_vectors_interp = np.zeros((total_samples, cond_dim))

                for dim in range(cond_dim):
                    f = interp1d(x_original, condition_vectors[:, dim], kind='linear')
                    condition_vectors_interp[:, dim] = f(x_new)

                condition_vectors = condition_vectors_interp

            # Use diffusion model with condition vectors to generate sequence
            delay1_sequence, loss1_sequence, delay2_sequence, loss2_sequence = self._generate_with_diffusion(condition_vector=condition_vectors)
        else:
            # 使用行为ID序列
            behavior_ids = self._generate_behavior_id_sequence(
                schedule, patterns, total_samples
            )

            # Use diffusion model with behavior IDs to generate sequence
            delay1_sequence, loss1_sequence, delay2_sequence, loss2_sequence = self._generate_with_diffusion(behavior_ids=behavior_ids)

        return delay1_sequence, loss1_sequence, delay2_sequence, loss2_sequence

    def _generate_behavior_id_sequence(
        self, schedule: Dict, patterns: Dict, total_samples: int
    ) -> List[int]:
        """Generate behavior ID sequence based on schedule"""
        # Map behavior type to cluster label
        behavior_cluster_map = patterns.get(
            "behavior_cluster_map",
            {"stable": 0, "burst_loss": 1, "high_jitter": 2, "recovery": 3},
        )

        # Initialize behavior ID sequence
        behavior_ids = [0] * total_samples  # 默认使用稳定行为

        # Process each schedule segment
        for segment in schedule["segments"]:
            start_time = segment["start_time"]
            end_time = segment["end_time"]
            behavior_type = segment["behavior_type"]

            # Calculate start and end indices
            start_idx = int(start_time / self.time_granularity)
            end_idx = int(end_time / self.time_granularity)

            # Map behavior type to cluster label
            cluster_label = behavior_cluster_map.get(behavior_type, 0)

            # Fill behavior ID sequence
            for i in range(start_idx, min(end_idx, total_samples)):
                behavior_ids[i] = cluster_label

        return behavior_ids

    def generate_dual_stream_sequences(
        self, behavior_ids: Optional[List[int]] = None, condition_vector: Optional[torch.Tensor] = None
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """使用扩散模型生成4D双流序列

        Args:
            behavior_ids: 行为ID序列，与condition_vector二选一
            condition_vector: 连续条件向量序列，与behavior_ids二选一

        Returns:
            Tuple of 4 lists: (delay1, loss1, delay2, loss2)
        """
        # 确保提供了行为ID或条件向量
        if behavior_ids is None and condition_vector is None:
            raise ValueError("必须提供behavior_ids或condition_vector")

        with torch.no_grad():
            if condition_vector is not None:
                # 使用连续条件向量生成序列
                # 转换为张量，形状: (batch_size, seq_len, cond_dim)
                condition_tensor = torch.tensor(condition_vector, dtype=torch.float32).unsqueeze(0)
                condition_tensor = condition_tensor.to(self.device)

                # Generate sequence using diffusion model with condition vector
                generated = self.diffusion_model.sample(condition_vector=condition_tensor)
            else:
                # 使用行为ID生成序列
                # Convert behavior_ids to tensor, shape: (batch_size, seq_len)
                behavior_ids_tensor = torch.tensor(behavior_ids, dtype=torch.long).unsqueeze(0)
                behavior_ids_tensor = behavior_ids_tensor.to(self.device)

                # Generate sequence using diffusion model with behavior_ids
                generated = self.diffusion_model.sample(behavior_ids_tensor)

        # 优先使用模型的denormalize方法进行反归一化
        try:
            # 使用模型的denormalize方法，支持4D输出
            delay1, loss1, delay2, loss2 = self.diffusion_model.denormalize(generated)
            # 从(batch_size, seq_len)中获取第一个样本，并确保延迟非负
            delay1 = np.clip(delay1[0], 0, None)
            delay2 = np.clip(delay2[0], 0, None)
            loss1 = loss1[0]
            loss2 = loss2[0]
        except ValueError as e:
            logger.warning(f"模型反归一化失败: {e}, 将使用normalizer进行反归一化")

            # 模型没有归一化参数，使用normalizer进行反归一化
            # Move to CPU and convert to numpy array
            generated_sequence = generated.cpu().numpy()[0]  # (seq_len, input_dim)

            if generated_sequence.shape[1] == 4:
                # 4D输入，双流数据
                delay1_norm = generated_sequence[:, 0]
                loss1_norm = generated_sequence[:, 1]
                delay2_norm = generated_sequence[:, 2]
                loss2_norm = generated_sequence[:, 3]

                # 使用4D反归一化方法
                delay1, loss1, delay2, loss2 = self.normalizer.denormalize4d(
                    delay1_norm, loss1_norm, delay2_norm, loss2_norm
                )

                # 确保延迟非负
                delay1 = np.clip(delay1, 0, None)
                delay2 = np.clip(delay2, 0, None)
            else:
                # 2D输入，单流数据，复制到双流
                delay_norm = generated_sequence[:, 0]
                loss_norm = generated_sequence[:, 1]

                # 使用2D反归一化方法
                delay, loss_rate = self.normalizer.denormalize(delay_norm, loss_norm)

                # 确保延迟非负
                delay = np.clip(delay, 0, None)

                # 复制到双流
                delay1 = delay
                loss1 = loss_rate
                delay2 = delay.copy()
                loss2 = loss_rate.copy()

        return delay1.tolist(), loss1.tolist(), delay2.tolist(), loss2.tolist()

    def generate_behavior_segment(
        self, behavior_type: str, patterns: Dict, duration: int = 60, _start_time: float = 0.0
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Generate a segment of dual stream network data for a specific behavior type

        Args:
            behavior_type: 行为类型字符串
            patterns: 行为模式定义
            duration: 生成数据的持续时间（秒）
            _start_time: 起始时间（秒），未使用

        Returns:
            Tuple of 4 lists: (delay1, loss1, delay2, loss2)
        """
        # Calculate number of samples needed
        num_samples = int(duration / self.time_granularity)

        # Map behavior type to cluster label
        behavior_cluster_map = patterns.get(
            "behavior_cluster_map",
            {"stable": 0, "burst_loss": 1, "high_jitter": 2, "recovery": 3},
        )

        cluster_label = behavior_cluster_map.get(behavior_type, 0)

        # Generate behavior ID sequence
        behavior_ids = [cluster_label] * num_samples

        # Use diffusion model to generate segment
        return self._generate_with_diffusion(behavior_ids)

    def save(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save generated simulation data"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    def load_schedule(self, schedule_path: Path) -> Dict:
        """Load behavior schedule from file"""
        with open(schedule_path, "r") as f:
            return json.load(f)
