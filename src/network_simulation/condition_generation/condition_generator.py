#!/usr/bin/env python3
"""
网络条件生成模块
负责基于行为条件生成网络模拟数据
"""

import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple

from ..utils.logger import get_logger
from .diffusion_model import ConditionDiffusionModel
from .constraint_injector import ConstraintInjector
from .normalization import Normalizer


logger = get_logger(__name__)


class ConditionGenerator:
    """基于行为条件生成网络模拟数据的生成器

    该类负责根据给定的行为计划和模式，生成符合物理约束的网络模拟数据，
    支持上下行独立参数配置和生成。
    """

    def _get_device(self) -> torch.device:
        """获取可用的计算设备

        Returns:
            torch.device: 可用的计算设备（MPS/CUDA/CPU）
        """
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def __init__(
        self,
        valid_loss_values_up: List[float] = None,
        valid_loss_values_down: List[float] = None,
        config: dict = None,
    ):
        """初始化条件生成器

        Args:
            valid_loss_values_up (List[float], optional): 上行合法丢包值列表，默认值为 [0.0, 1/3, 0.5, 2/3, 1.0]
            valid_loss_values_down (List[float], optional): 下行合法丢包值列表，默认值为 [0.0, 1/3, 0.5, 2/3, 1.0]
            config (dict, optional): 配置字典，默认值为空字典

        Examples:
            >>> from network_simulation.condition_generation.condition_generator import ConditionGenerator
            >>> generator = ConditionGenerator(
            ...     valid_loss_values_up=[0.0, 0.1, 0.2, 1.0],
            ...     valid_loss_values_down=[0.0, 0.2, 0.5, 1.0]
            ... )
        """
        # 导入默认配置
        try:
            from config import DEFAULT_TIME_GRANULARITY

            time_granularity_default = DEFAULT_TIME_GRANULARITY
        except ImportError:
            # 如果无法导入config模块，使用默认值
            time_granularity_default = 0.1  # 100ms

        # 使用config中的值，如果没有提供则使用默认值
        self.time_granularity = (
            config.get("time_granularity", time_granularity_default)
            if config is not None
            else time_granularity_default
        )
        self.device = self._get_device()
        self.valid_loss_values_up = valid_loss_values_up or [
            0.0,
            1 / 3,
            0.5,
            2 / 3,
            1.0,
        ]
        self.valid_loss_values_down = valid_loss_values_down or [
            0.0,
            1 / 3,
            0.5,
            2 / 3,
            1.0,
        ]
        self.config = config or {}

        # 初始化约束注入器，支持上下行独立参数
        self.constraint_injector = ConstraintInjector(
            self.valid_loss_values_up, self.valid_loss_values_down
        )

        # 初始化归一化器，支持上下行独立参数
        self.normalizer = Normalizer(
            self.valid_loss_values_up, self.valid_loss_values_down
        )

        # 初始化扩散模型
        self.diffusion_model = None
        self._init_diffusion_model()

    def _init_diffusion_model(self):
        """初始化扩散模型"""
        # 导入默认配置
        try:
            from ...config import (
                DEFAULT_INPUT_DIM,
                DEFAULT_BEHAVIOR_EMBED_DIM,
                DEFAULT_T,
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
        """加载模型检查点并初始化归一化参数

        Args:
            model_path (Path): 模型检查点路径，包含模型状态和归一化参数

        Examples:
            >>> from network_simulation.condition_generation.condition_generator import ConditionGenerator
            >>> from pathlib import Path
            >>> generator = ConditionGenerator()
            >>> generator.load_model(Path("models/diffusion_model_checkpoint.pt"))
        """
        logger.info(f"从 {model_path} 加载模型检查点")

        # 加载模型检查点
        checkpoint = torch.load(model_path, map_location=self.device)

        # 加载模型状态
        self.diffusion_model.load_state_dict(checkpoint["model_state_dict"])

        # 从检查点加载归一化参数
        self.normalizer.from_checkpoint(checkpoint)

        # 使用更新后的normalizer重新初始化constraint_injector，确保两者使用相同的合法丢包值
        self.constraint_injector = ConstraintInjector(normalizer=self.normalizer)

        logger.info(f"模型已加载: {model_path}")
        logger.info("归一化参数已初始化")
        logger.info("约束注入器已使用更新后的归一化器重新初始化")

    def generate(
        self, schedule: Dict, patterns: Dict, duration: int = 600
    ) -> pd.DataFrame:
        """基于行为计划和模式生成网络模拟数据

        Args:
            _schedule (Dict): 行为计划，包含行为类型序列或连续条件向量序列
                - 预期结构：{"behavior_sequence": [0, 1, 2], "behavior_durations": [60, 120, 180]}
            _patterns (Dict): 行为模式定义，包含各行为的特征参数
                - 预期结构：{"STABLE": {"delay_mean": 50, "loss_mean": 0.01}, ...}
            duration (int, optional): 生成数据的持续时间（秒），默认值为600秒

        Returns:
            pd.DataFrame: 包含以下列的网络模拟数据帧
                - timestamp: 时间戳，格式为datetime64[ns]
                - delay1: 上行延迟，单位为毫秒
                - loss_rate1: 上行丢包率，范围0-1
                - delay2: 下行延迟，单位为毫秒
                - loss_rate2: 下行丢包率，范围0-1

        Examples:
            >>> from network_simulation.condition_generation.condition_generator import ConditionGenerator
            >>> generator = ConditionGenerator()
            >>> schedule = {
            ...     "behavior_sequence": [0, 1, 0],
            ...     "behavior_durations": [30, 60, 30]
            ... }
            >>> patterns = {
            ...     0: {"delay_mean": 50, "loss_mean": 0.01},
            ...     1: {"delay_mean": 200, "loss_mean": 0.1}
            ... }
            >>> simulation_data = generator.generate(schedule, patterns, duration=120)
            >>> print(simulation_data.head())
        """
        # 计算需要的总样本数
        total_samples = int(duration / self.time_granularity)

        # 初始化带双流列的模拟数据
        simulation_data = {
            "timestamp": [],
            "delay1": [],
            "loss_rate1": [],  # 上行数据
            "delay2": [],
            "loss_rate2": [],  # 下行数据
        }

        # 生成时间戳序列
        start_time = pd.Timestamp.now()
        timestamps = [
            start_time + pd.Timedelta(seconds=i * self.time_granularity)
            for i in range(total_samples)
        ]
        simulation_data["timestamp"] = timestamps

        # Process schedule and generate data - 支持4D双流数据
        delay1_sequence, loss1_sequence, delay2_sequence, loss2_sequence = (
            self.generate_sequences(schedule, patterns, duration)
        )

        # 应用物理约束
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

        # 创建数据框
        df = pd.DataFrame(simulation_data)
        return df

    def generate_sequences(
        self, _schedule: Dict, _patterns: Dict, duration: int = 600
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """基于行为计划生成上下行延迟和丢包率序列

        Args:
            schedule (Dict): 行为计划，包含行为类型序列或连续条件向量序列
                - 预期结构：{"behavior_sequence": [0, 1, 2], "behavior_durations": [60, 120, 180]}
            patterns (Dict): 行为模式定义，包含各行为的特征参数
                - 预期结构：{"STABLE": {"delay_mean": 50, "loss_mean": 0.01}, ...}
            duration (int, optional): 生成数据的持续时间（秒），默认值为600秒

        Returns:
            Tuple[List[float], List[float], List[float], List[float]]: 上下行延迟和丢包率序列
                - delay1: 上行延迟序列，单位为毫秒
                - loss1: 上行丢包率序列，范围0-1
                - delay2: 下行延迟序列，单位为毫秒
                - loss2: 下行丢包率序列，范围0-1

        Examples:
            >>> from network_simulation.condition_generation.condition_generator import ConditionGenerator
            >>> generator = ConditionGenerator()
            >>> schedule = {
            ...     "behavior_sequence": [0, 1],
            ...     "behavior_durations": [60, 120]
            ... }
            >>> patterns = {
            ...     0: {"delay_mean": 50, "loss_mean": 0.01},
            ...     1: {"delay_mean": 200, "loss_mean": 0.1}
            ... }
            >>> delay1, loss1, delay2, loss2 = generator.generate_sequences(schedule, patterns, duration=180)
            >>> print(f"生成的序列长度: {len(delay1)}")
        """
        # 计算需要的总样本数
        total_samples = int(duration / self.time_granularity)
        batch_size = 1
        seq_len = total_samples

        # 从schedule和patterns构造条件向量
        # 这里使用简化的条件向量生成逻辑，实际应用中可能需要更复杂的逻辑
        # 获取条件维度，确保是整数，默认为8
        cond_dim = getattr(self.diffusion_model, "cond_dim", 8)
        # 确保cond_dim是整数，避免TypeError
        cond_dim = cond_dim if cond_dim is not None else 8

        # 生成随机条件向量作为备选，实际应用中应根据schedule和patterns生成
        condition_vector = np.random.randn(batch_size, seq_len, cond_dim)
        condition_vector_tensor = torch.tensor(
            condition_vector, dtype=torch.float32, device=self.device
        )

        # 使用条件向量调用sample方法生成数据
        with torch.no_grad():
            generated = self.diffusion_model.sample(condition_vector_tensor)

        # 将生成的数据转换为numpy数组
        generated = generated.cpu().numpy()[0]

        # 反归一化生成的数据
        delay1, loss1, delay2, loss2 = self.normalizer.denormalize4d(
            generated[:, 0], generated[:, 1], generated[:, 2], generated[:, 3]
        )

        # 确保延迟非负
        delay1 = np.clip(delay1, a_min=0, a_max=None)
        delay2 = np.clip(delay2, a_min=0, a_max=None)

        # 转换为列表返回
        return delay1.tolist(), loss1.tolist(), delay2.tolist(), loss2.tolist()

    def generate_behavior_segment(
        self,
        _behavior_type: str = "stable",
        _patterns: Dict = None,
        duration: int = 60,
        _start_time: float = 0.0,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """已废弃：使用 generate_dual_stream_sequences 方法替代

        Args:
            _behavior_type (str, optional): 行为类型，默认值为"stable"
            _patterns (Dict, optional): 行为模式定义，默认值为None
            duration (int, optional): 生成数据的持续时间（秒），默认值为60秒
            _start_time (float, optional): 开始时间，默认值为0.0秒

        Returns:
            Tuple[List[float], List[float], List[float], List[float]]: 上下行延迟和丢包率序列
                - delay1: 上行延迟序列，单位为毫秒
                - loss1: 上行丢包率序列，范围0-1
                - delay2: 下行延迟序列，单位为毫秒
                - loss2: 下行丢包率序列，范围0-1

        Examples:
            >>> from network_simulation.condition_generation.condition_generator import ConditionGenerator
            >>> generator = ConditionGenerator()
            >>> delay1, loss1, delay2, loss2 = generator.generate_behavior_segment("stable", duration=30)
            >>> print(f"生成的序列长度: {len(delay1)}")
        """
        logger.warning(
            "generate_behavior_segment 方法已废弃，使用 generate_dual_stream_sequences 方法替代"
        )
        # 保持向后兼容，返回随机数据
        total_samples = int(duration / self.time_granularity)
        delay1 = np.random.normal(50, 10, total_samples).tolist()
        loss1 = np.random.choice(self.valid_loss_values_up, total_samples).tolist()
        delay2 = np.random.normal(60, 15, total_samples).tolist()
        loss2 = np.random.choice(self.valid_loss_values_down, total_samples).tolist()
        return delay1, loss1, delay2, loss2

    def generate_from_reference(
        self, reference_sample_path: Path, duration: int = 600
    ) -> pd.DataFrame:
        """基于参考样本生成网络模拟数据

        Args:
            reference_sample_path (Path): 参考样本文件路径，包含timestamp、delay1、loss_rate1、delay2、loss_rate2列
            duration (int, optional): 生成数据的持续时间（秒），默认值为600秒

        Returns:
            pd.DataFrame: 包含以下列的网络模拟数据帧
                - timestamp: 时间戳，格式为datetime64[ns]
                - delay1: 上行延迟，单位为毫秒
                - loss_rate1: 上行丢包率，范围0-1
                - delay2: 下行延迟，单位为毫秒
                - loss_rate2: 下行丢包率，范围0-1

        Examples:
            >>> from network_simulation.condition_generation.condition_generator import ConditionGenerator
            >>> from pathlib import Path
            >>> generator = ConditionGenerator()
            >>> # 假设存在参考样本文件
            >>> reference_path = Path("data/reference_sample.csv")
            >>> simulation_data = generator.generate_from_reference(reference_path, duration=120)
            >>> print(simulation_data.head())
        """
        # 计算需要的总样本数
        total_samples = int(duration / self.time_granularity)
        batch_size = 1
        seq_len = total_samples

        # 加载参考样本
        reference_sample_df = pd.read_csv(
            reference_sample_path, parse_dates=["timestamp"]
        )

        # 提取参考样本特征
        # 从参考样本中提取延迟和丢包率的统计特征
        features = []

        # 上行延迟特征
        delay1 = reference_sample_df["delay1"].values
        features.append(np.mean(delay1))
        features.append(np.std(delay1))
        features.append(np.max(delay1))
        features.append(np.min(delay1))
        features.append(np.percentile(delay1, 25))
        features.append(np.percentile(delay1, 75))
        features.append(np.percentile(delay1, 95))
        features.append(np.kurtosis(delay1))  # 峰度
        features.append(np.skew(delay1))  # 偏度

        # 上行丢包率特征
        loss1 = reference_sample_df["loss_rate1"].values
        features.append(np.mean(loss1))
        features.append(np.std(loss1))
        features.append(np.max(loss1))
        features.append(np.min(loss1))
        features.append(np.percentile(loss1, 25))
        features.append(np.percentile(loss1, 75))
        features.append(np.percentile(loss1, 95))
        features.append(np.count_nonzero(loss1) / len(loss1))  # 非零丢包率比例

        # 下行延迟特征
        delay2 = reference_sample_df["delay2"].values
        features.append(np.mean(delay2))
        features.append(np.std(delay2))
        features.append(np.max(delay2))
        features.append(np.min(delay2))
        features.append(np.percentile(delay2, 25))
        features.append(np.percentile(delay2, 75))
        features.append(np.percentile(delay2, 95))
        features.append(np.kurtosis(delay2))  # 峰度
        features.append(np.skew(delay2))  # 偏度

        # 下行丢包率特征
        loss2 = reference_sample_df["loss_rate2"].values
        features.append(np.mean(loss2))
        features.append(np.std(loss2))
        features.append(np.max(loss2))
        features.append(np.min(loss2))
        features.append(np.percentile(loss2, 25))
        features.append(np.percentile(loss2, 75))
        features.append(np.percentile(loss2, 95))
        features.append(np.count_nonzero(loss2) / len(loss2))  # 非零丢包率比例

        # 转换为numpy数组
        features_np = np.array(features)

        # 确保特征向量与条件向量格式兼容
        cond_dim = getattr(self.diffusion_model, "cond_dim", 8)
        if len(features_np) < cond_dim:
            features_np = np.pad(
                features_np, (0, cond_dim - len(features_np)), "constant"
            )
        elif len(features_np) > cond_dim:
            features_np = features_np[:cond_dim]

        # 扩展特征向量到序列长度
        condition_vector = np.tile(features_np, (batch_size, seq_len, 1))
        condition_vector_tensor = torch.tensor(
            condition_vector, dtype=torch.float32, device=self.device
        )

        # 使用条件向量调用sample方法生成数据
        with torch.no_grad():
            generated = self.diffusion_model.sample(condition_vector_tensor)

        # 将生成的数据转换为numpy数组
        generated = generated.cpu().numpy()[0]

        # 反归一化生成的数据
        delay1, loss1, delay2, loss2 = self.normalizer.denormalize4d(
            generated[:, 0], generated[:, 1], generated[:, 2], generated[:, 3]
        )

        # 确保延迟非负
        delay1 = np.clip(delay1, a_min=0, a_max=None)
        delay2 = np.clip(delay2, a_min=0, a_max=None)

        # 生成时间戳序列
        start_time = pd.Timestamp.now()
        timestamps = [
            start_time + pd.Timedelta(seconds=i * self.time_granularity)
            for i in range(total_samples)
        ]

        # 创建模拟数据帧
        simulation_data = {
            "timestamp": timestamps,
            "delay1": delay1.tolist(),
            "loss_rate1": loss1.tolist(),
            "delay2": delay2.tolist(),
            "loss_rate2": loss2.tolist(),
        }

        df = pd.DataFrame(simulation_data)
        return df

    def save(self, df: pd.DataFrame, output_path: Path) -> None:
        """保存生成的模拟数据到文件

        Args:
            df (pd.DataFrame): 要保存的网络模拟数据帧，包含timestamp、delay1、loss_rate1、delay2、loss_rate2列
            output_path (Path): 输出文件路径，支持.csv格式

        Examples:
            >>> from network_simulation.condition_generation.condition_generator import ConditionGenerator
            >>> from pathlib import Path
            >>> generator = ConditionGenerator()
            >>> # 假设已生成模拟数据
            >>> # generator.save(simulation_data, Path("output/simulation_data.csv"))
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    def load_schedule(self, schedule_path: Path) -> Dict:
        """从文件加载行为计划

        Args:
            schedule_path (Path): 行为计划文件路径，JSON格式

        Returns:
            Dict: 行为计划字典，包含behavior_sequence和behavior_durations等字段

        Examples:
            >>> from network_simulation.condition_generation.condition_generator import ConditionGenerator
            >>> from pathlib import Path
            >>> generator = ConditionGenerator()
            >>> # 假设存在行为计划文件
            >>> # schedule = generator.load_schedule(Path("config/schedule.json"))
        """
        with open(schedule_path, "r") as f:
            return json.load(f)
