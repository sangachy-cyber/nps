#!/usr/bin/env python3
"""
特征提取模块
负责从处理后的网络数据中提取网络行为特征
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress, theilslopes
from pathlib import Path
from typing import Dict, List
from network_simulation.utils.logger import get_logger

# 初始化日志记录器
logger = get_logger(__name__)


class FeatureExtractor:
    """从处理后的网络数据中提取网络行为特征

    该类负责从处理后的网络数据中提取各种网络行为特征，包括统计特征、时序特征、
    突发性特征和跨方向关联特征，用于后续的模式识别和模型训练。
    """

    def __init__(self, config=None):
        """初始化特征提取器

        Args:
            config (dict, optional): 配置字典，包含以下可选参数:
                - window_size: 窗口大小（采样点），默认值: 100
                - stride: 滑动步长（采样点），默认值: 50
                - congestion_delay_threshold: 拥塞延迟阈值（ms），默认值: 200.0
                - congestion_loss_threshold: 拥塞丢包率阈值，默认值: 0.25
                - time_granularity: 时间粒度（秒），默认值: 0.1
                - acf_lag: ACF计算的滞后阶数，默认值: 5
                - loss_rate_high_threshold: 高丢包率阈值，默认值: 0.5
                - tolerance: 丢包率值匹配的容差，默认值: 1e-3

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> # 使用默认配置初始化
            >>> extractor = FeatureExtractor()
            >>> # 使用自定义配置初始化
            >>> custom_config = {
            ...     'window_size': 200,
            ...     'stride': 100,
            ...     'congestion_delay_threshold': 150.0
            ... }
            >>> extractor = FeatureExtractor(config=custom_config)
        """
        # 统一处理配置字典，避免多次判断 config is not None
        config = config or {}

        # 定义默认值
        default_window_size = 100
        default_stride = 50
        default_congestion_delay_threshold = 200.0
        default_congestion_loss_threshold = 0.25
        default_time_granularity = 0.1
        default_acf_lag = 5
        default_loss_rate_high_threshold = 0.5
        default_tolerance = 1e-3

        # 从配置字典加载参数，使用默认值作为后备
        self.window_samples = config.get("window_size", default_window_size)
        self.slide_samples = config.get("stride", default_stride)
        self.congestion_delay_threshold = config.get(
            "congestion_delay_threshold", default_congestion_delay_threshold
        )
        self.congestion_loss_threshold = config.get(
            "congestion_loss_threshold", default_congestion_loss_threshold
        )
        self.TIME_GRANULARITY = config.get(
            "time_granularity", default_time_granularity
        )  # 从配置加载时间粒度
        self.ACF_LAG = config.get("acf_lag", default_acf_lag)
        self.LOSS_RATE_HIGH_THRESHOLD = config.get(
            "loss_rate_high_threshold", default_loss_rate_high_threshold
        )
        self.TOLERANCE = config.get("tolerance", default_tolerance)

        logger.debug(
            f"特征提取器参数: window_size={self.window_samples}, stride={self.slide_samples}"
        )
        logger.debug(
            f"拥塞判定阈值: delay ≥ {self.congestion_delay_threshold}ms, loss ≥ {self.congestion_loss_threshold}"
        )

        self.time_granularity = self.TIME_GRANULARITY  # 时间粒度
        self.window_size = (
            self.window_samples * self.time_granularity
        )  # 计算窗口大小（秒）
        self.slide_step = (
            self.slide_samples * self.time_granularity
        )  # 计算滑动步长（秒）
        logger.info(
            f"特征提取器初始化完成，窗口大小: {self.window_size}秒, 滑动步长: {self.slide_step}秒"
        )

        # 合法丢包值集合和映射表，将在运行时自动提取
        self.valid_loss_values = []
        self.loss_mode_mapping = {}

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """加载处理后的网络数据

        从CSV文件中加载处理后的网络数据，并进行基本的数据验证。

        Args:
            file_path (Path): 处理后网络数据的CSV文件路径

        Returns:
            pd.DataFrame: 包含处理后网络数据的DataFrame

        Examples:
            >>> from pathlib import Path
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> extractor = FeatureExtractor()
            >>> file_path = Path("data/processed/network_data_processed.csv")
            >>> df = extractor.load_data(file_path)
            >>> print(f"加载的数据行数: {len(df)}")
        """
        return pd.read_csv(file_path, parse_dates=["timestamp"])

    def extract(
        self,
        df: pd.DataFrame,
        normalize: bool = False,
        correlation_threshold: float = 0.8,
        remove_correlated: bool = True,
    ) -> pd.DataFrame:
        """从处理后的网络数据中提取特征

        从处理后的网络数据中提取网络行为特征，包括统计特征、时序特征、
        突发性特征和跨方向关联特征。

        Args:
            df (pd.DataFrame): 包含处理后网络数据的DataFrame，必须包含以下列：
                - timestamp: 时间戳
                - delay1: 上行延迟（ms）
                - loss_rate1: 上行丢包率
                - delay2: 下行延迟（ms）
                - loss_rate2: 下行丢包率
            normalize (bool, optional): 是否对提取的特征进行归一化
                - 默认值: False，返回原始特征
                - 设置为True时，使用RobustScaler进行归一化
            correlation_threshold (float, optional): 高度相关特征的阈值，大于该值的特征将被去除
                - 默认值: 0.8
            remove_correlated (bool, optional): 是否去除高度相关的特征
                - 默认值: True
                - 设置为False时，保留所有提取的特征

        Returns:
            pd.DataFrame: 包含提取特征的DataFrame，特征列名以"feat_"前缀开头

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> import pandas as pd
            >>> import numpy as np
            >>> extractor = FeatureExtractor()
            >>> # 创建示例数据
            >>> df = pd.DataFrame({
            ...     'timestamp': pd.date_range('2025-01-01', periods=200, freq='100ms'),
            ...     'delay1': np.random.normal(50, 10, 200),
            ...     'loss_rate1': np.random.choice([0, 0.01, 0.05], 200),
            ...     'delay2': np.random.normal(60, 15, 200),
            ...     'loss_rate2': np.random.choice([0, 0.01, 0.05], 200)
            ... })
            >>> # 提取特征
            >>> features_df = extractor.extract(df, normalize=False)
            >>> print(f"提取的特征数量: {len([col for col in features_df.columns if col.startswith('feat_')])}")
            >>> print(f"特征行数: {len(features_df)}")
        """
        logger.info(f"开始提取特征，数据行数: {len(df)}, 归一化: {normalize}")

        # 检查是否包含所有必需的双通道列
        required_cols = {"timestamp", "delay1", "loss_rate1", "delay2", "loss_rate2"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(
                f"Missing required dual-channel columns: {sorted(missing)}. "
                "This system only supports bidirectional network data (e.g., client↔server)."
            )

        # 检查输入数据中是否包含NaN值，直接抛出异常
        if df[required_cols].isnull().values.any():
            nan_count = df[required_cols].isnull().sum().sum()
            total_count = df[required_cols].size
            raise ValueError(
                f"输入数据中包含 {nan_count} 个NaN值，占总数据的 {(nan_count / total_count) * 100:.2f}%。"
                f"请检查原始数据或数据处理过程，定位NaN值产生的具体原因。"
            )

        # 首先提取合法丢包值
        self.valid_loss_values = self._extract_valid_loss_values(df)
        logger.debug(f"提取到合法丢包值: {self.valid_loss_values}")

        # 从合法丢包值构建丢包模式映射
        self._build_loss_mode_mapping()
        logger.debug(f"构建丢包模式映射: {self.loss_mode_mapping}")

        features_list = []

        # 计算窗口数量
        total_samples = len(df)

        # 检查是否有足够的数据进行特征提取
        if total_samples < self.window_samples:
            logger.error(
                f"数据样本数 {total_samples} 小于窗口大小 {self.window_samples}，无法提取特征"
            )
            raise ValueError(
                f"Not enough samples for feature extraction: {total_samples} < {self.window_samples}"
            )

        num_windows = (total_samples - self.window_samples) // self.slide_samples + 1
        logger.debug(
            f"将数据划分为 {num_windows} 个窗口，窗口大小: {self.window_samples} 采样点, 滑动步长: {self.slide_samples} 采样点"
        )

        for i in range(num_windows):
            start_idx = i * self.slide_samples
            end_idx = start_idx + self.window_samples

            # 获取窗口数据
            window = df.iloc[start_idx:end_idx].copy()

            # Extract features - this method handles both single and dual channel data
            features = self._extract_window_features(window, start_idx, end_idx)
            features_list.append(features)

            if (i + 1) % 100 == 0 or i + 1 == num_windows:
                logger.debug(f"已处理 {i + 1}/{num_windows} 个窗口")

        # 创建特征数据框
        features_df = pd.DataFrame(features_list)
        logger.info(f"特征提取完成，共提取 {len(features_df)} 条特征记录")

        # 移除高度相关的特征（如果需要）
        if remove_correlated:
            features_df = self.remove_highly_correlated_features(
                features_df, correlation_threshold
            )
            logger.info(
                f"去除高度相关特征后，剩余特征数: {len([col for col in features_df.columns if col.startswith('feat_')])}"
            )
        else:
            logger.info(
                f"跳过高度相关特征移除，保留所有 {len([col for col in features_df.columns if col.startswith('feat_')])} 个特征"
            )

        # 如果请求则应用归一化
        if normalize:
            logger.info("对特征进行归一化处理")
            features_df = self.normalize_features(features_df)

        return features_df

    def _extract_valid_loss_values(self, df: pd.DataFrame) -> List[float]:
        """从整个数据集中提取合法丢包值

        Args:
            df (pd.DataFrame): 包含网络数据的DataFrame

        Returns:
            List[float]: 排序后的合法丢包值列表

        Raises:
            ValueError: 当数据格式不符合要求或没有找到合法丢包值时抛出

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> import pandas as pd
            >>> import numpy as np
            >>> extractor = FeatureExtractor()
            >>> df = pd.DataFrame({
            ...     'delay1': np.random.normal(50, 10, 100),
            ...     'loss_rate1': np.random.choice([0, 0.01, 0.05], 100),
            ...     'delay2': np.random.normal(60, 15, 100),
            ...     'loss_rate2': np.random.choice([0, 0.01, 0.05, 0.1], 100)
            ... })
            >>> valid_loss_values = extractor._extract_valid_loss_values(df)
            >>> print(f"提取的合法丢包值: {valid_loss_values}")
        """
        # 只支持双通道数据，精确匹配列名
        if all(col in df.columns for col in ["loss_rate1", "loss_rate2"]):
            loss_cols = ["loss_rate1", "loss_rate2"]
        else:
            raise ValueError(
                "Only dual-channel data supported. Expected columns: 'loss_rate1' and 'loss_rate2'"
            )

        loss_rates = np.concatenate([df[col].values for col in loss_cols])
        valid_values = sorted(list(set(loss_rates[~np.isnan(loss_rates)])))

        if not valid_values:
            raise ValueError("No valid loss values found")

        return valid_values

    def _build_loss_mode_mapping(self) -> None:
        """从合法丢包值构建丢包模式映射

        该方法根据提取的合法丢包值构建丢包模式映射表，用于后续的丢包模式编码。

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> extractor = FeatureExtractor()
            >>> extractor.valid_loss_values = [0, 0.01, 0.05, 0.1]
            >>> extractor._build_loss_mode_mapping()
            >>> print(f"丢包模式映射: {extractor.loss_mode_mapping}")
        """
        # 显式排序，确保映射关系的稳定性和一致性
        self.valid_loss_values = sorted(self.valid_loss_values)
        self.loss_mode_mapping = {
            val: idx for idx, val in enumerate(self.valid_loss_values)
        }

    def remove_highly_correlated_features(
        self, features_df: pd.DataFrame, correlation_threshold: float = 0.8
    ) -> pd.DataFrame:
        """使用相关矩阵阈值去除高度相关特征

        Args:
            features_df (pd.DataFrame): 包含特征数据的DataFrame
            correlation_threshold (float, optional): 相关性阈值，大于该阈值的特征将被去除，默认值: 0.8

        Returns:
            pd.DataFrame: 去除高度相关特征后的DataFrame

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> import pandas as pd
            >>> import numpy as np
            >>> extractor = FeatureExtractor()
            >>> # 创建示例特征数据
            >>> features_df = pd.DataFrame({
            ...     'feat_delay1_mean': np.random.normal(50, 10, 100),
            ...     'feat_delay1_std': np.random.normal(10, 2, 100),
            ...     'feat_delay1_raw_mean': np.random.normal(50, 10, 100),  # 与 feat_delay1_mean 高度相关
            ...     'feat_loss1_mean': np.random.normal(0.05, 0.02, 100),
            ...     'feat_loss1_std': np.random.normal(0.02, 0.01, 100)
            ... })
            >>> # 去除高度相关特征
            >>> filtered_features = extractor.remove_highly_correlated_features(features_df, correlation_threshold=0.8)
            >>> print(f"原始特征数量: {len(features_df.columns)}")
            >>> print(f"过滤后特征数量: {len(filtered_features.columns)}")
        """
        logger.info(
            f"开始去除高度相关特征，特征行数: {len(features_df)}, 相关性阈值: {correlation_threshold}"
        )

        # Get feature columns (excluding non-feature columns like window_start, window_end, etc.)
        feature_columns = [
            col for col in features_df.columns if col.startswith("feat_")
        ]

        if len(feature_columns) <= 1:
            logger.info("只有1个或更少的特征，不需要去除相关性")
            return features_df

        # 计算相关系数矩阵
        corr_matrix = features_df[feature_columns].corr().abs()

        # 创建掩码以识别高度相关的特征
        # 选择相关系数矩阵的上三角部分
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than correlation_threshold
        # 定义稳定行为检测的关键特征，这些特征永远不会被移除
        key_stable_features = [
            "feat_loss1_nonzero_ratio",
            "feat_loss2_nonzero_ratio",
            "feat_max_congestion_run1",
            "feat_max_congestion_run2",
            "feat_delay1_std",
            "feat_delay2_std",
            "feat_loss1_std",
            "feat_loss2_std",
            "feat_delay1_mean",
            "feat_delay2_mean",
            "feat_loss1_mean",
            "feat_loss2_mean",
        ]

        to_drop = [
            column
            for column in upper.columns
            # 只移除非关键特征且与其他特征高度相关的特征
            if column not in key_stable_features
            and any(upper[column] > correlation_threshold)
        ]

        if to_drop:
            logger.info(f"去除高度相关特征: {to_drop}")
            # 去除高度相关的特征
            return features_df.drop(to_drop, axis=1)
        else:
            logger.info("没有高度相关的特征需要去除")
            return features_df

    def normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """使用合适的缩放器对特征进行归一化

        - 延迟相关特征: 使用 RobustScaler
        - 丢包相关特征: 使用 RobustScaler

        Args:
            features_df (pd.DataFrame): 包含特征数据的DataFrame

        Returns:
            pd.DataFrame: 归一化后的特征数据

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> import pandas as pd
            >>> import numpy as np
            >>> extractor = FeatureExtractor()
            >>> # 创建示例特征数据
            >>> features_df = pd.DataFrame({
            ...     'feat_delay1_mean': np.random.normal(50, 10, 100),
            ...     'feat_delay1_std': np.random.normal(10, 2, 100),
            ...     'feat_loss1_mean': np.random.normal(0.05, 0.02, 100),
            ...     'feat_loss1_std': np.random.normal(0.02, 0.01, 100)
            ... })
            >>> # 归一化特征
            >>> normalized_features = extractor.normalize_features(features_df)
            >>> print(f"归一化后的延迟特征均值: {normalized_features['feat_delay1_mean'].mean():.4f}")
            >>> print(f"归一化后的丢包特征均值: {normalized_features['feat_loss1_mean'].mean():.4f}")
        """
        logger.info(f"开始归一化特征，特征行数: {len(features_df)}")

        # 创建副本以避免修改原始数据
        normalized_features = features_df.copy()

        # 获取特征列（排除非特征列）
        feature_columns = [
            col for col in normalized_features.columns if col.startswith("feat_")
        ]

        # 将特征列分为延迟相关、丢包相关和行为判断相关
        # 行为判断相关特征（如loss_nonzero_ratio）不进行归一化，保持原始值便于判断
        behavior_judgment_features = [
            col for col in feature_columns if "nonzero_ratio" in col
        ]
        logger.debug(f"行为判断相关特征: {behavior_judgment_features}")

        # 延迟相关特征（排除行为判断相关特征）
        delay_features = [
            col
            for col in feature_columns
            if "delay" in col
            and col in normalized_features.columns
            and col not in behavior_judgment_features
        ]
        logger.debug(f"延迟相关特征: {delay_features}")

        # 丢包相关特征（排除行为判断相关特征）
        loss_features = [
            col
            for col in feature_columns
            if ("loss" in col or "congestion" in col or "burst" in col)
            and col in normalized_features.columns
            and col not in behavior_judgment_features
        ]
        logger.debug(f"丢包相关特征: {loss_features}")

        # 检查是否缺少特征列
        all_expected_features = [
            col
            for col in feature_columns
            if "delay" in col or "loss" in col or "congestion" in col or "burst" in col
        ]
        missing_columns = [
            col
            for col in all_expected_features
            if col not in normalized_features.columns
        ]
        if missing_columns:
            logger.warning(f"缺少特征列，将跳过这些列的归一化: {missing_columns}")

        # 初始化缩放器
        from sklearn.preprocessing import RobustScaler

        # 只有当有特征需要归一化时才进行归一化
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
        self, window: pd.DataFrame, start_idx: int, end_idx: int
    ) -> Dict:
        """为单个窗口提取特征

        Args:
            window (pd.DataFrame): 单个窗口的数据
            start_idx (int): 窗口的起始索引
            end_idx (int): 窗口的结束索引

        Returns:
            Dict: 包含窗口特征的字典

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> import pandas as pd
            >>> import numpy as np
            >>> extractor = FeatureExtractor()
            >>> window = pd.DataFrame({
            ...     'timestamp': pd.date_range('2025-01-01', periods=100, freq='100ms'),
            ...     'delay1': np.random.normal(50, 10, 100),
            ...     'loss_rate1': np.random.choice([0, 0.01, 0.05], 100),
            ...     'delay2': np.random.normal(60, 15, 100),
            ...     'loss_rate2': np.random.choice([0, 0.01, 0.05], 100)
            ... })
            >>> features = extractor._extract_window_features(window, 0, 99)
            >>> print(f"提取的窗口特征数量: {len(features)}")
        """
        features = {
            "window_start": start_idx,
            "window_end": end_idx,
            "window_start_time": window["timestamp"].iloc[0],
            "window_end_time": window["timestamp"].iloc[-1],
        }

        # 只处理双通道数据
        self._extract_dual_channel_features(window, features)

        return features

    def _detect_data_channel_type(self, window: pd.DataFrame) -> str:
        """检测数据通道类型（仅支持双通道）

        Args:
            window (pd.DataFrame): 单个窗口的数据

        Returns:
            str: 数据通道类型，目前只返回 "dual"

        Raises:
            ValueError: 当数据格式不符合要求时抛出

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> import pandas as pd
            >>> import numpy as np
            >>> extractor = FeatureExtractor()
            >>> window = pd.DataFrame({
            ...     'timestamp': pd.date_range('2025-01-01', periods=100, freq='100ms'),
            ...     'delay1': np.random.normal(50, 10, 100),
            ...     'loss_rate1': np.random.choice([0, 0.01, 0.05], 100),
            ...     'delay2': np.random.normal(60, 15, 100),
            ...     'loss_rate2': np.random.choice([0, 0.01, 0.05], 100)
            ... })
            >>> channel_type = extractor._detect_data_channel_type(window)
            >>> print(f"数据通道类型: {channel_type}")
        """
        has_dual_channel = all(
            col in window.columns
            for col in ["delay1", "loss_rate1", "delay2", "loss_rate2"]
        )

        if has_dual_channel:
            return "dual"
        else:
            raise ValueError(
                f"Invalid data format. Only dual-channel data supported. Expected columns: delay1/loss_rate1/delay2/loss_rate2, got: {window.columns.tolist()}"
            )

    def _extract_dual_channel_features(
        self, window: pd.DataFrame, features: Dict
    ) -> None:
        """为双通道数据提取特征

        Args:
            window (pd.DataFrame): 单个窗口的数据
            features (Dict): 特征字典，用于存储提取的特征

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> import pandas as pd
            >>> import numpy as np
            >>> extractor = FeatureExtractor()
            >>> window = pd.DataFrame({
            ...     'timestamp': pd.date_range('2025-01-01', periods=100, freq='100ms'),
            ...     'delay1': np.random.normal(50, 10, 100),
            ...     'loss_rate1': np.random.choice([0, 0.01, 0.05], 100),
            ...     'delay2': np.random.normal(60, 15, 100),
            ...     'loss_rate2': np.random.choice([0, 0.01, 0.05], 100)
            ... })
            >>> features = {'window_start': 0, 'window_end': 99}
            >>> extractor._extract_dual_channel_features(window, features)
            >>> print(f"提取的特征数量: {len(features) - 2}")  # 减去 window_start 和 window_end
        """
        # 上行数据，确保非负
        delay1 = window["delay1"].values
        loss_rate1 = window["loss_rate1"].values
        # 确保延迟值非负，避免log1p时出现无效值
        delay1_clipped = np.maximum(delay1, 0)
        delay1_log1 = np.log1p(delay1_clipped)

        # 下行数据，确保非负
        delay2 = window["delay2"].values
        loss_rate2 = window["loss_rate2"].values
        # 确保延迟值非负，避免log1p时出现无效值
        delay2_clipped = np.maximum(delay2, 0)
        delay2_log1 = np.log1p(delay2_clipped)

        # 提取时序动态特征
        self._extract_temporal_features(
            delay1, loss_rate1, delay1_log1, features, suffix="1"
        )
        self._extract_temporal_features(
            delay2, loss_rate2, delay2_log1, features, suffix="2"
        )

        # 提取突发性特征
        self._extract_burst_features(delay1, loss_rate1, features, suffix="1")
        self._extract_burst_features(delay2, loss_rate2, features, suffix="2")

        # 提取长期趋势特征
        self._extract_trend_features(delay1_log1, features, suffix="1")
        self._extract_trend_features(delay2_log1, features, suffix="2")

        # 提取统计分布特征
        self._extract_statistical_features(
            delay1_log1, loss_rate1, features, suffix="1"
        )
        self._extract_statistical_features(
            delay2_log1, loss_rate2, features, suffix="2"
        )

        # 提取跨方向关联特征
        self._extract_cross_channel_features(
            delay1, delay2, loss_rate1, loss_rate2, features
        )

    def _extract_temporal_features(
        self,
        delay: np.ndarray,
        loss_rate: np.ndarray,
        delay_log1: np.ndarray,
        features: Dict,
        suffix: str,
    ) -> None:
        """提取时序动态特征

        Args:
            delay (np.ndarray): 原始延迟数据
            loss_rate (np.ndarray): 丢包率数据
            delay_log1 (np.ndarray): 对数变换后的延迟数据
            features (Dict): 特征字典，用于存储提取的特征
            suffix (str): 特征名称的后缀（如 "1" 表示上行，"2" 表示下行）

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> import numpy as np
            >>> extractor = FeatureExtractor()
            >>> delay = np.random.normal(50, 10, 100)
            >>> loss_rate = np.random.choice([0, 0.01, 0.05], 100)
            >>> delay_log1 = np.log1p(delay)
            >>> features = {}
            >>> extractor._extract_temporal_features(delay, loss_rate, delay_log1, features, "1")
            >>> print(f"提取的时序特征: {[k for k in features.keys() if 'delay1' in k or 'loss1' in k]}")
        """
        # 延迟特征（对数尺度和原始尺度）
        features[f"feat_delay{suffix}_std"] = np.std(delay_log1)  # 对数尺度的延迟标准差
        features[f"feat_delay{suffix}_mean"] = np.mean(delay_log1)  # 对数尺度的延迟均值
        features[f"feat_delay{suffix}_raw_std"] = np.std(delay)  # 原始尺度的延迟标准差
        features[f"feat_delay{suffix}_raw_mean"] = np.mean(delay)  # 原始尺度的延迟均值

        # 丢包率特征
        features[f"feat_loss{suffix}_nonzero_ratio"] = np.sum(loss_rate > 0) / len(
            loss_rate
        )
        features[f"feat_loss{suffix}_high_ratio"] = np.sum(
            loss_rate >= self.LOSS_RATE_HIGH_THRESHOLD
        ) / len(loss_rate)
        features[f"feat_loss{suffix}_mean"] = np.mean(loss_rate)
        features[f"feat_loss{suffix}_std"] = np.std(loss_rate)

    def _extract_burst_features(
        self, delay: np.ndarray, loss_rate: np.ndarray, features: Dict, suffix: str
    ) -> None:
        """提取突发性特征

        Args:
            delay (np.ndarray): 原始延迟数据
            loss_rate (np.ndarray): 丢包率数据
            features (Dict): 特征字典，用于存储提取的特征
            suffix (str): 特征名称的后缀（如 "1" 表示上行，"2" 表示下行）

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> import numpy as np
            >>> extractor = FeatureExtractor()
            >>> delay = np.random.normal(50, 10, 100)
            >>> loss_rate = np.random.choice([0, 0.01, 0.05], 100)
            >>> features = {}
            >>> extractor._extract_burst_features(delay, loss_rate, features, "1")
            >>> print(f"提取的突发性特征: {[k for k in features.keys() if 'max_consec_loss1' in k or 'max_congestion_run1' in k]}")
        """
        features[f"feat_max_consec_loss{suffix}"] = (
            self._calculate_max_consecutive_loss(loss_rate)
        )
        features[f"feat_max_congestion_run{suffix}"] = (
            self._calculate_max_congestion_run(delay, loss_rate)
        )

    def _extract_trend_features(
        self, delay_log1: np.ndarray, features: Dict, suffix: str
    ) -> None:
        """提取长期趋势特征

        Args:
            delay_log1 (np.ndarray): 对数变换后的延迟数据
            features (Dict): 特征字典，用于存储提取的特征
            suffix (str): 特征名称的后缀（如 "1" 表示上行，"2" 表示下行）

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> import numpy as np
            >>> extractor = FeatureExtractor()
            >>> delay_log1 = np.log1p(np.random.normal(50, 10, 100))
            >>> features = {}
            >>> extractor._extract_trend_features(delay_log1, features, "1")
            >>> print(f"提取的趋势特征: {[k for k in features.keys() if 'delay1_trend' in k]}")
        """
        features[f"feat_delay{suffix}_trend"] = self._calculate_delay_trend(delay_log1)

    def _extract_statistical_features(
        self, delay_log1: np.ndarray, loss_rate: np.ndarray, features: Dict, suffix: str
    ) -> None:
        """提取统计分布特征

        Args:
            delay_log1 (np.ndarray): 对数变换后的延迟数据
            loss_rate (np.ndarray): 丢包率数据
            features (Dict): 特征字典，用于存储提取的特征
            suffix (str): 特征名称的后缀（如 "1" 表示上行，"2" 表示下行）

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> import numpy as np
            >>> extractor = FeatureExtractor()
            >>> delay_log1 = np.log1p(np.random.normal(50, 10, 100))
            >>> loss_rate = np.random.choice([0, 0.01, 0.05], 100)
            >>> extractor.valid_loss_values = [0, 0.01, 0.05]
            >>> extractor._build_loss_mode_mapping()
            >>> features = {}
            >>> extractor._extract_statistical_features(delay_log1, loss_rate, features, "1")
            >>> print(f"提取的统计特征: {[k for k in features.keys() if 'acf' in k or 'mode_encoded' in k]}")
        """
        features[f"feat_delay{suffix}_acf_{self.ACF_LAG}"] = self._calculate_acf(
            delay_log1, lag=self.ACF_LAG
        )
        features[f"feat_loss{suffix}_mode_encoded"] = self._calculate_loss_mode_encoded(
            loss_rate
        )

    def _extract_cross_channel_features(
        self,
        delay1: np.ndarray,
        delay2: np.ndarray,
        loss_rate1: np.ndarray,
        loss_rate2: np.ndarray,
        features: Dict,
    ) -> None:
        """提取跨方向关联特征

        Args:
            delay1 (np.ndarray): 上行延迟数据
            delay2 (np.ndarray): 下行延迟数据
            loss_rate1 (np.ndarray): 上行丢包率数据
            loss_rate2 (np.ndarray): 下行丢包率数据
            features (Dict): 特征字典，用于存储提取的特征

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> import numpy as np
            >>> extractor = FeatureExtractor()
            >>> delay1 = np.random.normal(50, 10, 100)
            >>> delay2 = np.random.normal(60, 15, 100)
            >>> loss_rate1 = np.random.choice([0, 0.01, 0.05], 100)
            >>> loss_rate2 = np.random.choice([0, 0.01, 0.05], 100)
            >>> features = {}
            >>> extractor._extract_cross_channel_features(delay1, delay2, loss_rate1, loss_rate2, features)
            >>> print(f"提取的跨方向特征: {[k for k in features.keys() if 'delay_ratio' in k or 'loss_symmetry' in k or 'congestion_match' in k]}")
        """
        # 上下行延迟比
        features["feat_delay_ratio"] = (np.mean(delay1) + 1) / (np.mean(delay2) + 1)

        # 丢包对称性
        features["feat_loss_symmetry"] = 1.0 - abs(
            np.mean(loss_rate1) - np.mean(loss_rate2)
        )

        # 拥塞匹配度
        max_congestion_run1 = features.get("feat_max_congestion_run1", 0)
        max_congestion_run2 = features.get("feat_max_congestion_run2", 0)

        if max_congestion_run1 > 0 or max_congestion_run2 > 0:
            features["feat_congestion_match"] = min(
                max_congestion_run1, max_congestion_run2
            ) / max(max_congestion_run1, max_congestion_run2, 1)
        else:
            features["feat_congestion_match"] = 1.0  # 无拥塞时匹配度为1

    def _calculate_max_consecutive_loss(self, loss_rates: np.ndarray) -> int:
        """计算连续丢包样本的最大数量

        Args:
            loss_rates (np.ndarray): 丢包率数据

        Returns:
            int: 连续丢包样本的最大数量

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> import numpy as np
            >>> extractor = FeatureExtractor()
            >>> loss_rates = np.array([0, 0.01, 0.01, 0, 0.05, 0.05, 0.05, 0])
            >>> max_consec = extractor._calculate_max_consecutive_loss(loss_rates)
            >>> print(f"连续丢包样本的最大数量: {max_consec}")
            3
        """
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

    def _calculate_max_congestion_run(
        self, delays: np.ndarray, loss_rates: np.ndarray
    ) -> int:
        """计算连续拥塞样本的最大数量

        连续拥塞样本指的是延迟 ≥ 拥塞延迟阈值且丢包率 ≥ 拥塞丢包率阈值的连续样本。

        Args:
            delays (np.ndarray): 延迟数据
            loss_rates (np.ndarray): 丢包率数据

        Returns:
            int: 连续拥塞样本的最大数量

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> import numpy as np
            >>> extractor = FeatureExtractor(config={'congestion_delay_threshold': 100.0, 'congestion_loss_threshold': 0.2})
            >>> delays = np.array([120, 130, 90, 150, 160, 140])
            >>> loss_rates = np.array([0.3, 0.25, 0.1, 0.2, 0.25, 0.3])
            >>> max_congestion = extractor._calculate_max_congestion_run(delays, loss_rates)
            >>> print(f"连续拥塞样本的最大数量: {max_congestion}")
            3
        """
        if len(delays) == 0 or len(loss_rates) == 0:
            return 0

        # 使用向量化操作标记拥塞点
        congested = (delays >= self.congestion_delay_threshold) & (
            loss_rates >= self.congestion_loss_threshold
        )

        # 使用公共函数计算连续True序列的最大长度
        from network_simulation.utils.utils import calculate_max_consecutive_true

        return calculate_max_consecutive_true(congested)

    def _calculate_delay_trend(self, delays: np.ndarray) -> float:
        """使用 Theil-Sen 鲁棒斜率估计或线性回归计算延迟趋势

        如果有效点 > 总点数的 80%，使用线性回归，否则使用 Theil-Sen 估计器。

        Args:
            delays (np.ndarray): 延迟数据

        Returns:
            float: 延迟趋势斜率

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> import numpy as np
            >>> extractor = FeatureExtractor()
            >>> # 向上趋势
            >>> delays = np.log1p(np.linspace(50, 150, 100))
            >>> trend = extractor._calculate_delay_trend(delays)
            >>> print(f"向上趋势: {trend:.4f}")
            >>> # 向下趋势
            >>> delays = np.log1p(np.linspace(150, 50, 100))
            >>> trend = extractor._calculate_delay_trend(delays)
            >>> print(f"向下趋势: {trend:.4f}")
        """
        if len(delays) < 2:
            return 0.0

        x = np.arange(len(delays))

        # Check if we have enough valid points (non-nan values)
        valid_mask = ~np.isnan(delays)
        valid_count = np.sum(valid_mask)
        valid_ratio = valid_count / len(delays)

        if valid_ratio < 0.2:  # 有效点数量不足
            return 0.0

        # 获取有效数据点
        valid_x = x[valid_mask]
        valid_delays = delays[valid_mask]

        if len(valid_x) < 2:  # Ensure we have at least 2 valid points
            return 0.0

        try:
            if valid_ratio > 0.8:  # 对于大部分有效点，使用线性回归
                slope, _, _, _, _ = linregress(valid_x, valid_delays)
            else:  # Use scipy's Theil-Sen estimator for robust estimation
                slope, _, _, _ = theilslopes(valid_delays, valid_x)
        except Exception as e:
            # Fallback to 0 if any error occurs
            logger.warning(f"计算延迟趋势时出错: {e}")
            slope = 0.0

        return slope

    def _calculate_loss_mode_encoded(self, loss_rates: np.ndarray) -> int:
        """计算丢包率的模式并将其映射到序数编码

        Args:
            loss_rates (np.ndarray): 丢包率数组

        Returns:
            int: 编码后的丢包模式

        Raises:
            RuntimeError: 如果 loss_mode_mapping 尚未初始化

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> import numpy as np
            >>> extractor = FeatureExtractor()
            >>> extractor.valid_loss_values = [0, 0.01, 0.05]
            >>> extractor._build_loss_mode_mapping()
            >>> loss_rates = np.random.choice([0, 0.01, 0.01, 0.05], 100)
            >>> mode_encoded = extractor._calculate_loss_mode_encoded(loss_rates)
            >>> print(f"编码后的丢包模式: {mode_encoded}")
            1  # 0.01 的编码
        """
        # 计算众数
        unique_values, counts = np.unique(loss_rates, return_counts=True)
        if len(unique_values) == 0:
            logger.warning("loss_rates 数组为空，无法计算模式")
            return 0

        mode = unique_values[np.argmax(counts)]

        # 如果没有精确匹配，找到最接近的合法丢包值
        # 确保 mapping 已初始化，添加明确的断言
        assert self.loss_mode_mapping, (
            "loss_mode_mapping not initialized. Call extract() first to initialize valid_loss_values."
        )
        assert self.valid_loss_values, (
            "valid_loss_values not initialized. Call extract() first to initialize valid_loss_values."
        )

        try:
            # 找到最接近的合法丢包值
            if mode not in self.loss_mode_mapping:
                # 计算到所有合法丢包值的距离
                distances = [
                    abs(mode - valid_val) for valid_val in self.valid_loss_values
                ]
                closest_idx = np.argmin(distances)
                mode = self.valid_loss_values[closest_idx]

            # 将众数映射到序数编码
            return self.loss_mode_mapping[mode]
        except Exception as e:
            # Fallback to 0 if any error occurs
            logger.warning(f"计算丢包模式编码时出错: {e}")
            return 0

    def _calculate_acf(self, data: np.ndarray, lag: int) -> float:
        """计算指定滞后阶数的自相关函数

        Args:
            data (np.ndarray): 时间序列数据
            lag (int): 滞后阶数

        Returns:
            float: 自相关系数

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> import numpy as np
            >>> extractor = FeatureExtractor()
            >>> # 生成自相关数据
            >>> data = np.cumsum(np.random.normal(0, 1, 100))
            >>> acf = extractor._calculate_acf(data, lag=5)
            >>> print(f"滞后5阶的自相关系数: {acf:.4f}")
        """
        if len(data) < lag + 1 or np.isnan(data).any() or lag >= len(data):
            return 0.0

        # Normalize data
        data_normalized = data - np.mean(data)
        denominator = np.sum(data_normalized**2)

        if denominator == 0:
            return 0.0

        # Calculate autocovariance
        numerator = np.sum(data_normalized[:-lag] * data_normalized[lag:])

        return numerator / denominator

    def save(self, features: pd.DataFrame, output_path: Path) -> None:
        """保存提取的特征到文件

        Args:
            features (pd.DataFrame): 提取的特征数据
            output_path (Path): 输出文件路径

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> import pandas as pd
            >>> import numpy as np
            >>> from pathlib import Path
            >>> extractor = FeatureExtractor()
            >>> features = pd.DataFrame({
            ...     'feat_delay1_mean': np.random.normal(50, 10, 10),
            ...     'feat_loss1_mean': np.random.normal(0.05, 0.02, 10)
            ... })
            >>> output_path = Path("data/features/test_features.csv")
            >>> extractor.save(features, output_path)
            >>> print(f"特征已保存到: {output_path}")
        """
        logger.info(f"开始保存特征到文件: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output_path, index=False)
        logger.info(f"特征保存完成，保存行数: {len(features)}")

    def extract_from_directory(self, input_dir: Path, output_dir: Path) -> Path:
        """从目录中的多个处理后的数据文件中提取特征

        Args:
            input_dir (Path): 包含处理后数据文件的目录路径
            output_dir (Path): 特征数据的输出目录路径

        Returns:
            Path: 合并后的特征文件路径

        Examples:
            >>> from network_simulation.pattern_discovery.feature_extractor import FeatureExtractor
            >>> from pathlib import Path
            >>> extractor = FeatureExtractor()
            >>> input_dir = Path("data/processed")
            >>> output_dir = Path("data/features")
            >>> output_file = extractor.extract_from_directory(input_dir, output_dir)
            >>> print(f"合并后的特征文件: {output_file}")
        """
        logger.info(
            f"开始从目录提取特征，输入目录: {input_dir}, 输出目录: {output_dir}"
        )

        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有处理后的文件并排序
        processed_files = sorted(list(input_dir.glob("*.csv")), key=lambda x: x.name)
        if not processed_files:
            logger.warning(f"在 {input_dir} 中未找到 .csv 文件")
            raise ValueError(f"No CSV files found in {input_dir}")

        # 为每个文件单独提取特征
        all_features = []
        for processed_file in processed_files:
            # 跳过合并后的文件，避免重复处理
            if processed_file.name == "merged_processed_data.csv":
                continue

            # 加载单个文件数据
            df = self.load_data(processed_file)
            logger.info(f"处理文件: {processed_file.name}, 样本数: {len(df)}")

            # 提取特征
            features = self.extract(df)

            # 添加文件标识（使用文件名，不含扩展名）
            # 收集特征
            all_features.append(features)
            logger.info(
                f"文件 {processed_file.stem} 特征提取完成，特征数: {len(features)}"
            )

        # 合并所有特征数据
        merged_features_df = pd.concat(all_features, ignore_index=True)
        logger.info(
            f"合并了 {len(all_features)} 个文件的特征，总特征数: {len(merged_features_df)}"
        )

        # 保存合并后的特征数据
        features_output_file = output_dir / "merged_features.csv"
        self.save(merged_features_df, features_output_file)

        logger.info(f"特征提取完成，合并特征已保存到: {features_output_file}")
        return features_output_file
