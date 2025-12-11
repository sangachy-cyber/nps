#!/usr/bin/env python3
"""
数据处理模块
负责处理原始数据和预处理训练数据
"""

import pandas as pd
import numpy as np
from pathlib import Path

from network_simulation.utils.logger import get_logger
from network_simulation.data_processing.data_loader import DataLoader

# 初始化日志记录器
logger = get_logger(__name__)


class DataProcessor:
    """数据处理器，负责处理原始数据和预处理训练数据

    该类提供了数据处理的高级接口，包括原始数据处理和训练数据预处理功能。
    """

    def __init__(self):
        """初始化数据处理器

        创建 DataLoader 实例用于数据加载和预处理。
        """
        self.data_loader = DataLoader()

    def process_raw_data(self, input_file: Path, output_dir: Path) -> Path:
        """处理单个原始网络数据文件

        该函数读取原始网络数据文件，解析延迟和丢包率数据，并将其转换为结构化的CSV格式。

        Args:
            input_file (Path): 原始数据文件路径，包含网络测量数据
            output_dir (Path): 处理后数据的输出目录路径

        Returns:
            Path: 处理后的数据文件路径

        Raises:
            FileNotFoundError: 如果输入文件不存在
            ValueError: 如果数据格式不符合要求

        Examples:
            >>> from pathlib import Path
            >>> from network_simulation.data_processing.data_processor import DataProcessor
            >>> data_processor = DataProcessor()
            >>> input_file = Path("data/raw/network_data.csv")
            >>> output_dir = Path("data/processed")
            >>> output_file = data_processor.process_raw_data(input_file, output_dir)
            >>> print(f"处理完成，输出文件: {output_file}")
        """
        logger.info(f"正在处理原始数据文件: {input_file}")

        # 加载原始数据
        df = self.data_loader.load(input_file)

        # 预处理数据
        processed_df = self.data_loader.preprocess(df)

        # 保存处理后的数据
        output_file = output_dir / f"{input_file.stem}_processed.csv"
        self.data_loader.save(processed_df, output_file)

        logger.info(f"原始数据处理完成，保存到: {output_file}")
        return output_file

    def preprocess_data(
        self,
        patterns_dir: Path,
        processed_file: Path,
        output_dir: Path,
        direction: str = "up",
    ) -> Path:
        """预处理训练数据

        该函数从处理后的数据文件和模式文件中加载数据，进行预处理，准备用于模型训练。

        Args:
            patterns_dir (Path): 模式文件目录路径
            processed_file (Path): 处理后的数据文件路径
            output_dir (Path): 预处理数据的输出目录路径
            direction (str, optional): 数据方向，可选值："up"（上行）或 "down"（下行），默认为 "up"

        Returns:
            Path: 预处理数据的输出文件路径

        Raises:
            FileNotFoundError: 如果输入文件不存在
            ValueError: 如果数据格式不符合要求

        Examples:
            >>> from pathlib import Path
            >>> data_processor = DataProcessor()
            >>> patterns_dir = Path("patterns")
            >>> processed_file = Path("data/processed/network_data_processed.csv")
            >>> output_dir = Path("data/preprocessed")
            >>> output_file = data_processor.preprocess_data(patterns_dir, processed_file, output_dir, direction="up")
            >>> print(f"预处理完成，输出文件: {output_file}")
        """
        logger.info(f"正在预处理训练数据: {processed_file}，方向: {direction}")

        # 验证方向参数
        if direction not in ["up", "down"]:
            raise ValueError(f"无效的方向参数: {direction}，必须是 'up' 或 'down'")

        # 加载处理后的数据
        processed_df = pd.read_csv(processed_file, parse_dates=["timestamp"])

        # 加载行为标签
        # 根据方向构建标签文件名
        labels_file = patterns_dir / f"labels_rule_{direction}.npy"
        if not labels_file.exists():
            raise FileNotFoundError(f"行为标签文件不存在: {labels_file}")

        import numpy as np

        labels = np.load(labels_file)
        logger.info(f"加载{direction}行行为标签，共 {len(labels)} 个标签")

        # 加载行为统计信息
        behavior_stats_file = patterns_dir / f"behavior_statistics_{direction}.json"
        if not behavior_stats_file.exists():
            raise FileNotFoundError(f"行为统计信息文件不存在: {behavior_stats_file}")

        import json

        with open(behavior_stats_file, "r") as f:
            behavior_stats = json.load(f)
        logger.info(f"加载{direction}行行为统计信息，共 {len(behavior_stats)} 种行为")

        # 加载合法丢包值
        valid_loss_file = patterns_dir / "valid_loss_values.json"
        if not valid_loss_file.exists():
            raise FileNotFoundError(f"合法丢包值文件不存在: {valid_loss_file}")

        with open(valid_loss_file, "r") as f:
            valid_loss_values = json.load(f)
        logger.info(f"加载合法丢包值: {valid_loss_values}")

        # 预处理上下行数据
        preprocessed_df = self._preprocess_data(processed_df, labels)

        # 保存预处理后的数据
        output_file = output_dir / f"{processed_file.stem}_preprocessed.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        preprocessed_df.to_csv(output_file, index=False)

        logger.info(f"训练数据预处理完成，保存到: {output_file}")
        return output_file

    def _preprocess_data(
        self, processed_df: pd.DataFrame, labels: np.ndarray
    ) -> pd.DataFrame:
        """预处理上下行数据

        Args:
            processed_df (pd.DataFrame): 处理后的数据
            labels (np.ndarray): 行为标签数组

        Returns:
            pd.DataFrame: 预处理后的数据，包含行为标签

        Examples:
            >>> from network_simulation.data_processing.data_processor import DataProcessor
            >>> import pandas as pd
            >>> import numpy as np
            >>> data_processor = DataProcessor()
            >>> processed_df = pd.DataFrame({
            ...     'timestamp': pd.date_range('2025-01-01', periods=100, freq='100ms'),
            ...     'delay1': np.random.normal(50, 10, 100),
            ...     'loss_rate1': np.random.choice([0, 0.01, 0.05], 100),
            ...     'delay2': np.random.normal(60, 15, 100),
            ...     'loss_rate2': np.random.choice([0, 0.01, 0.05], 100)
            ... })
            >>> labels = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], 100)
            >>> preprocessed_df = data_processor._preprocess_data(processed_df, labels)
            >>> print(preprocessed_df.columns)
            Index(['timestamp', 'delay1', 'loss_rate1', 'delay2', 'loss_rate2', 'behavior_label'], dtype='object')
        """
        # 确保数据长度与标签长度匹配
        if len(processed_df) != len(labels):
            logger.warning(
                f"数据长度 {len(processed_df)} 与标签长度 {len(labels)} 不匹配，将截断数据"
            )
            min_length = min(len(processed_df), len(labels))
            processed_df = processed_df.iloc[:min_length]
            labels = labels[:min_length]

        # 添加行为标签
        processed_df["behavior_label"] = labels

        # 过滤掉无效标签
        valid_df = processed_df[processed_df["behavior_label"] != -1]
        logger.info(f"过滤掉 {len(processed_df) - len(valid_df)} 个无效标签")

        return valid_df
