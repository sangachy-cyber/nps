#!/usr/bin/env python3
"""
Data Processing Module
Responsible for loading, preprocessing, and saving network data files
"""

import pandas as pd
from pathlib import Path
from network_simulation.utils.logger import get_logger

# 初始化日志记录器
logger = get_logger(__name__)


class DataLoader:
    """网络模拟数据加载器和预处理器

    该类负责从文件中加载网络数据，支持CSV和HoloWAN Recorder File格式，
    并进行预处理，包括数据格式转换、归一化和重采样等操作。
    """

    def __init__(self):
        """初始化数据加载器

        初始化时间粒度为100ms，符合项目要求。
        """
        self.time_granularity = 0.1  # 100ms as per requirements

    def load(self, file_path: Path) -> pd.DataFrame:
        """从文件加载原始网络数据（支持CSV和HoloWAN Recorder File格式）

        Args:
            file_path (Path): 数据文件路径

        Returns:
            pd.DataFrame: 加载并初步处理后的网络数据

        Raises:
            ValueError: 当数据格式不支持或加载失败时抛出

        Examples:
            >>> from pathlib import Path
            >>> from network_simulation.data_processing.data_loader import DataLoader
            >>> data_loader = DataLoader()
            >>> # 加载CSV格式数据
            >>> csv_file = Path("data/raw/network_data.csv")
            >>> csv_df = data_loader.load(csv_file)
            >>> # 加载HoloWAN格式数据
            >>> holowan_file = Path("data/raw/holowan_recorder.txt")
            >>> holowan_df = data_loader.load(holowan_file)
        """
        logger.info(f"开始加载数据文件: {file_path}")
        try:
            with open(file_path, "r") as f:
                first_line = f.readline().strip()

            # Check if it's a HoloWAN Recorder File
            if first_line == "HoloWAN Recorder File (www.msytest.com)":
                logger.info("识别到HoloWAN Recorder File格式")
                df = self._load_holowan_file(file_path)
            else:
                # 作为标准CSV加载
                logger.info("识别到标准CSV格式")
                df = pd.read_csv(
                    file_path,
                    parse_dates=["timestamp"],
                )

            # 只处理双通道数据
            expected_cols = ["delay1", "loss_rate1", "delay2", "loss_rate2"]
            if not all(col in df.columns for col in expected_cols):
                raise ValueError(
                    f"只支持双通道数据，期望列名: {expected_cols}，实际列名: {df.columns.tolist()}"
                )

            # 确保loss_rate1和loss_rate2在0-1范围内
            for col in ["loss_rate1", "loss_rate2"]:
                # 如果值是百分比形式（大于1），转换为小数
                if df[col].max() > 1:
                    logger.debug(f"{col}值大于1，转换为小数形式")
                    df[col] = df[col] / 100.0
                # Ensure loss_rate is between 0 and 1
                df[col] = df[col].clip(0, 1)
            logger.debug("确保上下行丢包率在0-1范围内")

            # 添加原始文件路径列
            df["file_path"] = str(file_path)
            logger.info(f"成功加载数据，共 {len(df)} 行")
            return df
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise ValueError(f"Failed to load data from {file_path}: {e}")

    def _load_holowan_file(self, file_path: Path) -> pd.DataFrame:
        """加载HoloWAN Recorder File格式的数据

        Args:
            file_path (Path): HoloWAN Recorder File文件路径

        Returns:
            pd.DataFrame: 转换为标准格式的网络数据

        Examples:
            >>> from pathlib import Path
            >>> data_loader = DataLoader()
            >>> holowan_file = Path("data/raw/holowan_recorder.txt")
            >>> # 内部方法，不建议直接调用
            >>> # df = data_loader._load_holowan_file(holowan_file)
        """
        logger.debug("开始解析HoloWAN Recorder File格式")
        # Read metadata
        metadata = {}
        with open(file_path, "r") as f:
            for i in range(12):
                line = f.readline().strip()
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip().strip('"')

        # Extract start time and interval
        start_time = pd.to_datetime(metadata["Start Time"])
        interval = float(metadata["Interval(sec)"])
        logger.debug(f"HoloWAN文件元数据: 开始时间={start_time}, 间隔={interval}秒")

        # Read data rows, skipping header and separator
        df = pd.read_csv(
            file_path,
            skiprows=15,  # Skip first 15 lines (metadata + empty line + column names + separator)
            header=None,
            names=[
                "Delay1(ms)",
                "Loss1(%)",
                "Bandwidth1(Mbps)",
                "Delay2(ms)",
                "Loss2(%)",
                "Bandwidth2(Mbps)",
            ],
        )
        logger.debug(f"读取了 {len(df)} 行HoloWAN原始数据")

        # Generate timestamp sequence
        timestamps = [
            start_time + pd.Timedelta(seconds=i * interval) for i in range(len(df))
        ]

        # 处理上行数据
        # When Bandwidth1 is 0, it indicates 100% packet loss
        loss_rate1 = (
            df["Loss1(%)"].astype(float).values / 100.0
        )  # Convert percentage to decimal
        bandwidth1 = df["Bandwidth1(Mbps)"].astype(float).values
        loss_rate1[bandwidth1 == 0] = 1.0  # Set 100% loss when bandwidth is 0
        logger.debug(f"处理了 {sum(bandwidth1 == 0)} 个上行带宽为0的100%丢包情况")

        # 处理下行数据
        # When Bandwidth2 is 0, it indicates 100% packet loss
        loss_rate2 = (
            df["Loss2(%)"].astype(float).values / 100.0
        )  # Convert percentage to decimal
        bandwidth2 = df["Bandwidth2(Mbps)"].astype(float).values
        loss_rate2[bandwidth2 == 0] = 1.0  # Set 100% loss when bandwidth is 0
        logger.debug(f"处理了 {sum(bandwidth2 == 0)} 个下行带宽为0的100%丢包情况")

        result_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "delay1": df["Delay1(ms)"].values,
                "loss_rate1": loss_rate1,
                "bandwidth1": bandwidth1,
                "delay2": df["Delay2(ms)"].values,
                "loss_rate2": loss_rate2,
                "bandwidth2": bandwidth2,
                "file_path": str(file_path),
            }
        )

        logger.debug(f"成功转换为标准格式，共 {len(result_df)} 行")
        return result_df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理原始网络数据

        Args:
            df (pd.DataFrame): 原始网络数据

        Returns:
            pd.DataFrame: 预处理后的网络数据

        Raises:
            ValueError: 当数据缺少必要列时抛出

        Examples:
            >>> data_loader = DataLoader()
            >>> raw_df = pd.DataFrame({
            ...     'timestamp': pd.date_range('2025-01-01', periods=100, freq='100ms'),
            ...     'delay1': np.random.normal(50, 10, 100),
            ...     'loss_rate1': np.random.choice([0, 0.01, 0.05], 100),
            ...     'delay2': np.random.normal(60, 15, 100),
            ...     'loss_rate2': np.random.choice([0, 0.01, 0.05], 100),
            ...     'file_path': 'test_data.csv'
            ... })
            >>> processed_df = data_loader.preprocess(raw_df)
        """
        logger.info(f"开始预处理数据，原始数据共 {len(df)} 行")

        # 如果数据框为空，直接返回
        if len(df) == 0:
            logger.warning("输入数据为空，直接返回")
            return df

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.debug("按时间戳排序数据")

        # 只支持双通道数据，确保必要列存在
        expected_cols = ["delay1", "loss_rate1", "delay2", "loss_rate2"]
        for col in expected_cols:
            if col not in df.columns:
                raise ValueError(f"预处理需要双通道数据，缺少列: {col}")

        # 确保带宽列存在
        for col in ["bandwidth1", "bandwidth2"]:
            if col not in df.columns:
                df[col] = 1000.0  # 默认带宽1000Mbps

        # 检查上下行延迟是否超过2000ms
        delay1_exceed_idx = df[df["delay1"] > 2000].index
        delay2_exceed_idx = df[df["delay2"] > 2000].index

        # 合并两个延迟超过阈值的索引
        delay_exceed_idx = delay1_exceed_idx.union(delay2_exceed_idx)

        if not delay_exceed_idx.empty:
            # Get the first occurrence index
            cutoff_idx = delay_exceed_idx[0]
            # Keep only data before this index
            df = df.iloc[:cutoff_idx]
            logger.warning(f"检测到延迟超过2000ms，截断数据至 {cutoff_idx} 行")
            if len(df) == 0:
                logger.warning("截断后数据为空")
                return df

        # 保存原始文件路径
        file_path = df["file_path"].iloc[0] if "file_path" in df.columns else "unknown"

        # 上下行数据的聚合函数
        agg_func = {
            "delay1": "mean",
            "loss_rate1": "mean",
            "bandwidth1": "mean",
            "delay2": "mean",
            "loss_rate2": "mean",
            "bandwidth2": "mean",
        }

        # Resample to 100ms granularity
        df_resampled = (
            df.set_index("timestamp")
            .resample(f"{int(self.time_granularity * 1000)}ms")
            .agg(agg_func)
            .reset_index()
        )
        logger.debug(
            f"重采样到{self.time_granularity}秒粒度，得到 {len(df_resampled)} 行数据"
        )

        # Fill missing values using time-appropriate interpolation
        df_resampled = df_resampled.interpolate(method="pad")
        logger.debug("使用前向填充插值填充缺失值")

        # Ensure loss_rate is between 0 and 1
        df_resampled["loss_rate1"] = df_resampled["loss_rate1"].clip(0, 1)
        df_resampled["loss_rate2"] = df_resampled["loss_rate2"].clip(0, 1)
        logger.debug("确保上下行丢包率在0-1范围内")

        # 添加回文件路径列
        df_resampled["file_path"] = file_path

        logger.info(f"预处理完成，共 {len(df_resampled)} 行数据")
        return df_resampled

    def save(self, df: pd.DataFrame, output_path: Path) -> None:
        """保存处理后的数据到文件

        Args:
            df (pd.DataFrame): 处理后的网络数据
            output_path (Path): 输出文件路径

        Examples:
            >>> data_loader = DataLoader()
            >>> processed_df = pd.DataFrame({
            ...     'timestamp': pd.date_range('2025-01-01', periods=100, freq='100ms'),
            ...     'delay1': np.random.normal(50, 10, 100),
            ...     'loss_rate1': np.random.choice([0, 0.01, 0.05], 100),
            ...     'delay2': np.random.normal(60, 15, 100),
            ...     'loss_rate2': np.random.choice([0, 0.01, 0.05], 100),
            ...     'file_path': 'test_data.csv'
            ... })
            >>> output_path = Path("data/processed/test_output.csv")
            >>> data_loader.save(processed_df, output_path)
        """
        logger.info(f"开始保存数据到文件: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"成功保存 {len(df)} 行数据到 {output_path}")
