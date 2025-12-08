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
    """Data loader and preprocessor for network simulation data"""

    def __init__(self):
        self.time_granularity = 0.1  # 100ms as per requirements

    def load(self, file_path: Path) -> pd.DataFrame:
        """Load raw network data from file (supports CSV and HoloWAN Recorder File formats)"""
        logger.info(f"开始加载数据文件: {file_path}")
        try:
            with open(file_path, "r") as f:
                first_line = f.readline().strip()

            # Check if it's a HoloWAN Recorder File
            if first_line == "HoloWAN Recorder File (www.msytest.com)":
                logger.info("识别到HoloWAN Recorder File格式")
                df = self._load_holowan_file(file_path)
            else:
                # Load as standard CSV
                logger.info("识别到标准CSV格式")
                df = pd.read_csv(
                    file_path,
                    parse_dates=["timestamp"],
                    dtype={"delay": float, "loss_rate": float},
                )

                # Ensure loss_rate is between 0 and 1
                # If values are in percentage (greater than 1), convert to decimal
                if df["loss_rate"].max() > 1:
                    logger.debug("丢包率值大于1，转换为小数形式")
                    df["loss_rate"] = df["loss_rate"] / 100.0

                # Ensure loss_rate is between 0 and 1
                df["loss_rate"] = df["loss_rate"].clip(0, 1)
                logger.debug("确保丢包率在0-1范围内")

            # 添加原始文件路径列
            df['file_path'] = str(file_path)
            logger.info(f"成功加载数据，共 {len(df)} 行")
            return df
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise ValueError(f"Failed to load data from {file_path}: {e}")

    def _load_holowan_file(self, file_path: Path) -> pd.DataFrame:
        """Load HoloWAN Recorder File format"""
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

        # Create standard format dataframe with proper loss rate handling
        # When Bandwidth1 is 0, it indicates 100% packet loss
        # Ensure Loss1(%) is numeric
        loss_rate = df["Loss1(%)"].astype(float).values / 100.0  # Convert percentage to decimal initially
        bandwidth1 = df["Bandwidth1(Mbps)"].astype(float).values

        # Set loss_rate to 1.0 (100%) when Bandwidth1 is 0
        loss_rate[bandwidth1 == 0] = 1.0
        logger.debug(f"处理了 {sum(bandwidth1 == 0)} 个带宽为0的100%丢包情况")

        result_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "delay": df["Delay1(ms)"].values,
                "loss_rate": loss_rate,
                "file_path": str(file_path),
            }
        )

        logger.debug(f"成功转换为标准格式，共 {len(result_df)} 行")
        return result_df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw network data"""
        logger.info(f"开始预处理数据，原始数据共 {len(df)} 行")

        # 如果数据框为空，直接返回
        if len(df) == 0:
            logger.warning("输入数据为空，直接返回")
            return df

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.debug("按时间戳排序数据")

        # Check for delay > 2000ms and truncate data if found
        # Find the index where delay first exceeds 2000ms
        delay_exceed_idx = df[df["delay"] > 2000].index
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
        file_path = df['file_path'].iloc[0] if 'file_path' in df.columns else 'unknown'
        
        # Resample to 100ms granularity
        df_resampled = (
            df.set_index("timestamp")
            .resample(f"{int(self.time_granularity * 1000)}ms")
            .agg({"delay": "mean", "loss_rate": "mean"})
            .reset_index()
        )
        logger.debug(f"重采样到{self.time_granularity}秒粒度，得到 {len(df_resampled)} 行数据")

        # Fill missing values using linear interpolation
        df_resampled = df_resampled.interpolate(method="linear")
        logger.debug("使用线性插值填充缺失值")

        # Ensure loss_rate is between 0 and 1
        df_resampled["loss_rate"] = df_resampled["loss_rate"].clip(0, 1)
        logger.debug("确保丢包率在0-1范围内")
        
        # 添加回文件路径列
        df_resampled['file_path'] = file_path

        logger.info(f"预处理完成，共 {len(df_resampled)} 行数据")
        return df_resampled

    def save(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save processed data to file"""
        logger.info(f"开始保存数据到文件: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"成功保存 {len(df)} 行数据到 {output_path}")
