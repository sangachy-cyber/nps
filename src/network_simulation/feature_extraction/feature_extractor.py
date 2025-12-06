#!/usr/bin/env python3
"""
特征提取模块
负责从处理后的网络数据中提取网络行为特征
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Dict, List
from network_simulation.utils.logger import get_logger

# 初始化日志记录器
logger = get_logger(__name__)


class FeatureExtractor:
    """从处理后的网络数据中提取网络行为特征"""

    def __init__(self, config=None):
        # 导入配置
        if config is None:
            try:
                # 尝试相对导入
                from ...config import DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE
                self.window_samples = DEFAULT_WINDOW_SIZE  # 从配置文件加载窗口大小
                self.slide_samples = DEFAULT_STRIDE  # 从配置文件加载滑动步长
                logger.info(f"从配置文件加载参数: window_size={self.window_samples}, stride={self.slide_samples}")
            except ImportError:
                # 导入失败时使用默认值
                self.window_samples = 100  # 默认窗口大小
                self.slide_samples = 50  # 默认滑动步长
                logger.warning("配置文件导入失败，使用默认参数")
        else:
            self.window_samples = config.get('window_size', 100)  # 从配置字典加载
            self.slide_samples = config.get('stride', 50)
            logger.info(f"从配置字典加载参数: window_size={self.window_samples}, stride={self.slide_samples}")

        self.time_granularity = 0.1  # 100ms
        self.window_size = self.window_samples * self.time_granularity  # 计算窗口大小（秒）
        self.slide_step = self.slide_samples * self.time_granularity  # 计算滑动步长（秒）
        logger.info(f"特征提取器初始化完成，窗口大小: {self.window_size}秒, 滑动步长: {self.slide_step}秒")

        # 合法丢包值集合和映射表，将在运行时自动提取
        self.valid_loss_values = []
        self.loss_mode_mapping = {}

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load processed network data"""
        return pd.read_csv(file_path, parse_dates=["timestamp"])

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from processed network data"""
        logger.info(f"开始提取特征，数据行数: {len(df)}")

        # Extract valid loss values first
        self.valid_loss_values = self._extract_valid_loss_values(df)
        logger.info(f"提取到合法丢包值: {self.valid_loss_values}")

        # Build loss mode mapping from valid loss values
        self._build_loss_mode_mapping()
        logger.debug(f"构建丢包模式映射: {self.loss_mode_mapping}")

        features_list = []

        # Calculate number of windows
        total_samples = len(df)
        num_windows = (total_samples - self.window_samples) // self.slide_samples + 1
        logger.info(f"将数据划分为 {num_windows} 个窗口，窗口大小: {self.window_samples} 采样点, 滑动步长: {self.slide_samples} 采样点")

        for i in range(num_windows):
            start_idx = i * self.slide_samples
            end_idx = start_idx + self.window_samples

            # Get window data
            window = df.iloc[start_idx:end_idx].copy()

            # Extract features
            features = self._extract_window_features(df, window, start_idx, end_idx)
            features_list.append(features)

            if (i + 1) % 100 == 0 or i + 1 == num_windows:
                logger.info(f"已处理 {i + 1}/{num_windows} 个窗口")

        # Create features dataframe
        features_df = pd.DataFrame(features_list)
        logger.info(f"特征提取完成，共提取 {len(features_df)} 条特征记录")

        return features_df

    def _extract_valid_loss_values(self, df: pd.DataFrame) -> List[float]:
        """Extract valid loss values from the entire dataset with tolerance matching

        Args:
            df: Processed network data dataframe

        Returns:
            List of unique valid loss values sorted in ascending order
        """
        loss_rates = df["loss_rate"].values
        unique_values = np.unique(loss_rates)

        # Remove NaN values if any
        unique_values = unique_values[~np.isnan(unique_values)]

        # Apply tolerance matching to merge similar values
        tolerance = 1e-3
        valid_values = []

        for val in sorted(unique_values):
            # Check if this value is close to any already in valid_values
            if not valid_values or all(
                np.abs(val - existing) > tolerance for existing in valid_values
            ):
                valid_values.append(val)

        return valid_values

    def _build_loss_mode_mapping(self) -> None:
        """Build loss mode mapping from valid loss values"""
        self.loss_mode_mapping = {
            val: idx for idx, val in enumerate(self.valid_loss_values)
        }

    def normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using appropriate scalers:
        - delay-related features: StandardScaler
        - loss-related features: RobustScaler
        """
        logger.info(f"开始归一化特征，特征行数: {len(features_df)}")

        # Create a copy to avoid modifying the original
        normalized_features = features_df.copy()

        # Separate feature columns into delay-related and loss-related
        delay_features = [
            "feat_delay_std",
            "feat_delay_trend",
            "feat_delay_acf_5",
        ]
        logger.debug(f"延迟相关特征: {delay_features}")

        loss_features = [
            "feat_loss_nonzero_ratio",
            "feat_loss_high_ratio",
            "feat_max_consec_loss",
            "feat_loss_unique_values",
            "feat_loss_mode_encoded",
            "feat_loss_pattern_std",
        ]
        logger.debug(f"丢包相关特征: {loss_features}")

        # Check if all feature columns exist
        all_features = delay_features + loss_features
        missing_columns = [
            col for col in all_features if col not in normalized_features.columns
        ]
        if missing_columns:
            logger.error(f"缺少特征列: {missing_columns}")
            raise ValueError(f"Missing feature columns: {missing_columns}")

        # Initialize scalers
        from sklearn.preprocessing import RobustScaler

        delay_scaler = StandardScaler()
        loss_scaler = RobustScaler()

        # Normalize the features with appropriate scalers
        normalized_features[delay_features] = delay_scaler.fit_transform(
            normalized_features[delay_features]
        )
        logger.debug("延迟相关特征已使用StandardScaler归一化")

        normalized_features[loss_features] = loss_scaler.fit_transform(
            normalized_features[loss_features]
        )
        logger.debug("丢包相关特征已使用RobustScaler归一化")

        logger.info("特征归一化完成")
        return normalized_features

    def _extract_window_features(
        self, df: pd.DataFrame, window: pd.DataFrame, start_idx: int, end_idx: int
    ) -> Dict:
        """Extract features for a single window"""
        features = {
            "window_start": start_idx,
            "window_end": end_idx,
            "window_start_time": window["timestamp"].iloc[0],
            "window_end_time": window["timestamp"].iloc[-1],
        }

        # Extract delay and loss data
        delays = window["delay"].values
        loss_rates = window["loss_rate"].values

        # 1. 时序动态特征
        # feat_delay_std: 10秒窗口内时延的标准差
        features["feat_delay_std"] = np.std(delays)

        # feat_loss_nonzero_ratio: 10秒窗口内 loss_rate > 0 的采样点比例
        non_zero_loss = np.sum(loss_rates > 0)
        features["feat_loss_nonzero_ratio"] = non_zero_loss / len(loss_rates)

        # feat_loss_high_ratio: 10秒窗口内 loss_rate ≥ 0.5 的采样点比例
        high_loss = np.sum(loss_rates >= 0.5)
        features["feat_loss_high_ratio"] = high_loss / len(loss_rates)

        # 2. 突发性特征
        # feat_max_consec_loss: 窗口内最长连续 loss_rate > 0 的采样点数量
        features["feat_max_consec_loss"] = self._calculate_max_consecutive_loss(
            loss_rates
        )

        # feat_loss_unique_values: 窗口内非零 loss_rate 的唯一取值个数
        non_zero_loss_values = loss_rates[loss_rates > 0]
        features["feat_loss_unique_values"] = (
            len(np.unique(non_zero_loss_values)) if len(non_zero_loss_values) > 0 else 0
        )

        # 3. 长期趋势特征
        # feat_delay_trend: 基于10秒窗口内 delay 序列的 Theil-Sen 稳健斜率估计
        features["feat_delay_trend"] = self._calculate_delay_trend(delays)

        # 4. 统计分布特征
        # feat_delay_acf_5: 时延序列的5阶自相关系数
        features["feat_delay_acf_5"] = self._calculate_acf(delays, lag=5)

        # feat_loss_mode_encoded: 将窗口内 loss_rate 的众数映射为序数编码
        features["feat_loss_mode_encoded"] = self._calculate_loss_mode_encoded(
            loss_rates
        )

        # feat_loss_pattern_std: 仅在 loss_rate > 0 的点上计算标准差
        features["feat_loss_pattern_std"] = (
            np.std(non_zero_loss_values) if len(non_zero_loss_values) > 0 else 0.0
        )

        return features

    def _calculate_max_consecutive_loss(self, loss_rates: np.ndarray) -> int:
        """Calculate the maximum number of consecutive samples with loss_rate > 0"""
        max_consec = 0
        current_consec = 0

        for loss in loss_rates:
            if loss > 0:
                current_consec += 1
                if current_consec > max_consec:
                    max_consec = current_consec
            else:
                current_consec = 0

        return max_consec

    def _calculate_delay_trend(self, delays: np.ndarray) -> float:
        """Calculate delay trend using Theil-Sen robust slope estimation or linear regression
        If valid points > 80% of total, use linear regression, otherwise use Theil-Sen estimator
        """
        if len(delays) < 2:
            return 0.0

        x = np.arange(len(delays))

        # Check if we have enough valid points (non-nan values)
        valid_mask = ~np.isnan(delays)
        valid_count = np.sum(valid_mask)
        valid_ratio = valid_count / len(delays)

        if valid_ratio < 0.2:  # Not enough valid points
            return 0.0

        if valid_ratio > 0.8:  # Use linear regression for most valid points
            slope, _, _, _, _ = linregress(x[valid_mask], delays[valid_mask])
        else:  # Use Theil-Sen estimator for robust estimation
            # Implement simple Theil-Sen estimator
            slopes = []
            for i in range(len(x) - 1):
                for j in range(i + 1, len(x)):
                    if not (np.isnan(delays[i]) or np.isnan(delays[j])):
                        slope_ij = (delays[j] - delays[i]) / (x[j] - x[i])
                        slopes.append(slope_ij)

            if slopes:
                slope = np.median(slopes)
            else:
                slope = 0.0

        return slope

    def _calculate_loss_mode_encoded(self, loss_rates: np.ndarray) -> int:
        """Calculate the mode of loss_rates and map it to ordinal encoding"""
        # Calculate mode
        unique_values, counts = np.unique(loss_rates, return_counts=True)
        if len(unique_values) == 0:
            return 0

        mode = unique_values[np.argmax(counts)]

        # Find the closest valid loss value if exact match not found
        if mode not in self.loss_mode_mapping:
            # Calculate distances to all valid loss values
            distances = [abs(mode - valid_val) for valid_val in self.valid_loss_values]
            closest_idx = np.argmin(distances)
            mode = self.valid_loss_values[closest_idx]

        # Map mode to ordinal encoding
        return self.loss_mode_mapping[mode]

    def _calculate_acf(self, data: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation function at specified lag"""
        if len(data) < lag + 1:
            return 0.0

        # Normalize data
        data_normalized = data - np.mean(data)

        # Calculate autocovariance
        numerator = np.sum(data_normalized[:-lag] * data_normalized[lag:])
        denominator = np.sum(data_normalized**2)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def save(self, features: pd.DataFrame, output_path: Path) -> None:
        """Save extracted features to file"""
        logger.info(f"开始保存特征到文件: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output_path, index=False)
        logger.info(f"特征保存完成，保存行数: {len(features)}")
