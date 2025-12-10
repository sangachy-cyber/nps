#!/usr/bin/env python3
"""
特征提取模块
负责从处理后的网络数据中提取网络行为特征
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
from pathlib import Path
from typing import Dict, List
from config import CONGESTION_DELAY_THRESHOLD, CONGESTION_LOSS_THRESHOLD
from network_simulation.utils.logger import get_logger

# 初始化日志记录器
logger = get_logger(__name__)


class FeatureExtractor:
    """从处理后的网络数据中提取网络行为特征"""

    def __init__(self, config=None):
        # 导入配置
        # 先定义默认值
        default_window_size = 100
        default_stride = 50

        if config is None:
            try:
                # 使用绝对导入
                from config import DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE
                self.window_samples = DEFAULT_WINDOW_SIZE  # 从配置文件加载窗口大小
                self.slide_samples = DEFAULT_STRIDE  # 从配置文件加载滑动步长
                default_window_size = DEFAULT_WINDOW_SIZE
                default_stride = DEFAULT_STRIDE
                logger.info(f"从配置文件加载参数: window_size={self.window_samples}, stride={self.slide_samples}")
            except ImportError as e:
                # 导入失败时使用默认值
                self.window_samples = default_window_size  # 默认窗口大小
                self.slide_samples = default_stride  # 默认滑动步长
                logger.warning(f"配置文件导入失败: {e}, 使用默认参数")
        else:
            self.window_samples = config.get('window_size', default_window_size)  # 从配置字典加载，使用默认值作为备选
            self.slide_samples = config.get('stride', default_stride)
            logger.info(f"从配置字典加载参数: window_size={self.window_samples}, stride={self.slide_samples}")

        self.time_granularity = 0.1  # 100ms
        self.window_size = self.window_samples * self.time_granularity  # 计算窗口大小（秒）
        self.slide_step = self.slide_samples * self.time_granularity  # 计算滑动步长（秒）
        logger.info(f"特征提取器初始化完成，窗口大小: {self.window_size}秒, 滑动步长: {self.slide_step}秒")

        # 记录拥塞判定阈值
        from config import CONGESTION_DELAY_THRESHOLD, CONGESTION_LOSS_THRESHOLD
        logger.info(f"拥塞判定阈值: delay ≥ {CONGESTION_DELAY_THRESHOLD}ms, loss ≥ {CONGESTION_LOSS_THRESHOLD}")

        # 合法丢包值集合和映射表，将在运行时自动提取
        self.valid_loss_values = []
        self.loss_mode_mapping = {}

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """加载处理后的网络数据

        从CSV文件中加载处理后的网络数据，并进行基本的数据验证。

        Args:
            file_path: 处理后网络数据的CSV文件路径

        Returns:
            包含处理后网络数据的DataFrame
        """
        return pd.read_csv(file_path, parse_dates=["timestamp"])

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """从处理后的网络数据中提取特征

        从处理后的网络数据中提取网络行为特征，包括统计特征、时序特征等。

        Args:
            df: 包含处理后网络数据的DataFrame

        Returns:
            包含提取特征的DataFrame
        """
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

        # Remove highly correlated features
        features_df = self.remove_highly_correlated_features(features_df)
        logger.info(f"去除高度相关特征后，剩余特征数: {len([col for col in features_df.columns if col.startswith('feat_')])}")

        return features_df

    def _extract_valid_loss_values(self, df: pd.DataFrame) -> List[float]:
        """Extract valid loss values from the entire dataset with tolerance matching

        Args:
            df: Processed network data dataframe

        Returns:
            List of unique valid loss values sorted in ascending order
        """
        # 合并上下行丢包率数据
        loss_rates = np.concatenate([df["loss_rate1"].values, df["loss_rate2"].values])

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
        # 显式排序，确保映射关系的稳定性和一致性
        self.valid_loss_values = sorted(self.valid_loss_values)
        self.loss_mode_mapping = {
            val: idx for idx, val in enumerate(self.valid_loss_values)
        }

    def remove_highly_correlated_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features using correlation matrix threshold"""
        logger.info(f"开始去除高度相关特征，特征行数: {len(features_df)}")

        # Get feature columns (excluding non-feature columns like window_start, window_end, etc.)
        feature_columns = [col for col in features_df.columns if col.startswith('feat_')]

        if len(feature_columns) <= 1:
            logger.info("只有1个或更少的特征，不需要去除相关性")
            return features_df

        # Calculate correlation matrix
        corr_matrix = features_df[feature_columns].corr().abs()

        # Create a mask to identify highly correlated features
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than 0.8
        to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

        if to_drop:
            logger.info(f"去除高度相关特征: {to_drop}")
            # Drop the highly correlated features
            return features_df.drop(to_drop, axis=1)
        else:
            logger.info("没有高度相关的特征需要去除")
            return features_df

    def normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using appropriate scalers:
        - delay-related features: RobustScaler
        - loss-related features: RobustScaler
        """
        logger.info(f"开始归一化特征，特征行数: {len(features_df)}")

        # Create a copy to avoid modifying the original
        normalized_features = features_df.copy()

        # Get feature columns (excluding non-feature columns)
        feature_columns = [col for col in normalized_features.columns if col.startswith('feat_')]

        # Separate feature columns into delay-related and loss-related
        delay_features = [
            col for col in feature_columns
            if 'delay' in col
        ]
        logger.debug(f"延迟相关特征: {delay_features}")

        loss_features = [
            col for col in feature_columns
            if 'loss' in col or 'congestion' in col or 'burst' in col
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

        # Normalize only if there are features to normalize
        if delay_features:
            delay_scaler = RobustScaler()
            normalized_features[delay_features] = delay_scaler.fit_transform(
                normalized_features[delay_features]
            )
            logger.debug("延迟相关特征已使用RobustScaler归一化")

        if loss_features:
            loss_scaler = RobustScaler()
            normalized_features[loss_features] = loss_scaler.fit_transform(
                normalized_features[loss_features]
            )
            logger.debug("丢包相关特征已使用RobustScaler归一化")

        logger.info("特征归一化完成")
        return normalized_features

    def _extract_window_features(
        self, _df: pd.DataFrame, window: pd.DataFrame, start_idx: int, end_idx: int
    ) -> Dict:
        """Extract features for a single window"""
        features = {
            "window_start": start_idx,
            "window_end": end_idx,
            "window_start_time": window["timestamp"].iloc[0],
            "window_end_time": window["timestamp"].iloc[-1],
        }

        # 上行数据
        delay1 = window["delay1"].values
        loss_rate1 = window["loss_rate1"].values
        delay1_log1 = np.log(1 + delay1)

        # 下行数据
        delay2 = window["delay2"].values
        loss_rate2 = window["loss_rate2"].values
        delay2_log1 = np.log(1 + delay2)

        # 1. 时序动态特征
        # 上行特征
        features["feat_delay1_std"] = np.std(delay1_log1)
        features["feat_loss1_nonzero_ratio"] = np.sum(loss_rate1 > 0) / len(loss_rate1)
        features["feat_loss1_high_ratio"] = np.sum(loss_rate1 >= 0.5) / len(loss_rate1)

        # 下行特征
        features["feat_delay2_std"] = np.std(delay2_log1)
        features["feat_loss2_nonzero_ratio"] = np.sum(loss_rate2 > 0) / len(loss_rate2)
        features["feat_loss2_high_ratio"] = np.sum(loss_rate2 >= 0.5) / len(loss_rate2)

        # 2. 突发性特征
        # 上行特征
        features["feat_max_consec_loss1"] = self._calculate_max_consecutive_loss(loss_rate1)
        features["feat_max_congestion_run1"] = self._calculate_max_congestion_run(delay1, loss_rate1)

        # 下行特征
        features["feat_max_consec_loss2"] = self._calculate_max_consecutive_loss(loss_rate2)
        features["feat_max_congestion_run2"] = self._calculate_max_congestion_run(delay2, loss_rate2)

        # 3. 长期趋势特征
        # 上行特征
        features["feat_delay1_trend"] = self._calculate_delay_trend(delay1_log1)

        # 下行特征
        features["feat_delay2_trend"] = self._calculate_delay_trend(delay2_log1)

        # 4. 统计分布特征
        # 上行特征
        features["feat_delay1_acf_5"] = self._calculate_acf(delay1_log1, lag=5)
        features["feat_loss1_mode_encoded"] = self._calculate_loss_mode_encoded(loss_rate1)

        # 下行特征
        features["feat_delay2_acf_5"] = self._calculate_acf(delay2_log1, lag=5)
        features["feat_loss2_mode_encoded"] = self._calculate_loss_mode_encoded(loss_rate2)

        # 5. 跨方向关联特征（新增）
        features["feat_delay_ratio"] = (np.mean(delay1) + 1) / (np.mean(delay2) + 1)
        features["feat_loss_symmetry"] = 1.0 - abs(np.mean(loss_rate1) - np.mean(loss_rate2))

        # 上行最大拥塞运行长度
        max_congestion_run1 = features.get("feat_max_congestion_run1", 0)
        # 下行最大拥塞运行长度
        max_congestion_run2 = features.get("feat_max_congestion_run2", 0)

        # 计算拥塞匹配度
        if max_congestion_run1 > 0 or max_congestion_run2 > 0:
            features["feat_congestion_match"] = min(max_congestion_run1, max_congestion_run2) / max(max_congestion_run1, max_congestion_run2, 1)
        else:
            features["feat_congestion_match"] = 1.0  # 无拥塞时匹配度为1

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

    def _calculate_max_congestion_run(self, delays: np.ndarray, loss_rates: np.ndarray) -> int:
        """Calculate the maximum number of consecutive samples where delay ≥ CONGESTION_DELAY_THRESHOLD and loss ≥ CONGESTION_LOSS_THRESHOLD"""
        max_consec = 0
        current_consec = 0

        for delay, loss in zip(delays, loss_rates):
            if delay >= CONGESTION_DELAY_THRESHOLD and loss >= CONGESTION_LOSS_THRESHOLD:
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
