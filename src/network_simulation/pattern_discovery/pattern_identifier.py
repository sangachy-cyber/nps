#!/usr/bin/env python3
"""
网络行为模式发现模块
负责使用规则事件检测引擎发现网络行为模式
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PatternIdentifier:
    """使用规则事件检测引擎识别网络行为模式

    该类使用基于规则的事件检测算法识别网络行为模式，支持上下行独立检测，
    能够识别多种网络行为模式，包括稳定、弱突发、强突发、瞬时峰值等。
    """

    # 行为标签常量定义
    BEHAVIOR_LABELS: Dict[str, int] = {
        "STABLE": 0,
        "WEAK_BURST": 1,
        "STRONG_BURST": 2,
        "INSTANT_SPIKE": 3,
        "HIGH_DELAY_NO_LOSS": 4,
        "HIGH_LOSS_STEADY": 5,
        "FREQUENT_FLUCTUATION": 6,
        "LOW_DELAY_HIGH_LOSS": 7,
        "INVALID": -1,  # 无效窗口标签
    }

    # 行为标签反向映射，便于外部解读数字标签
    LABEL_TO_NAME = {v: k for k, v in BEHAVIOR_LABELS.items()}

    def __init__(self, method: str = "rule", config: Dict = None):
        """初始化模式识别器

        Args:
            method (str, optional): 检测方法，目前只支持 "rule"，默认为 "rule"
            config (Dict): 配置字典，必须包含以下参数：
                - STRONG_BURST_DELAY_THRESHOLD: Strong Burst的延迟阈值
                - STRONG_BURST_LOSS_THRESHOLD: Strong Burst的丢包率阈值
                - STRONG_BURST_MIN_RUN: Strong Burst的最小连续点数量
                - INSTANT_SPIKE_DELAY_THRESHOLD: 瞬时峰值的延迟阈值
                - INSTANT_SPIKE_LOSS_THRESHOLD: 瞬时峰值的丢包率阈值
                - INSTANT_SPIKE_MAX_COUNT: 瞬时峰值的最大峰值点数量
                - INSTANT_SPIKE_MAX_RATIO: 瞬时峰值的最大峰值占比
                - WEAK_BURST_LOSS_NONZERO_RATIO: Weak Burst的丢包率非零比例阈值
                - WEAK_BURST_MIN_CONDITIONS: Weak Burst的最小满足条件数量
                - DYNAMIC_THRESHOLD_WINDOW_SIZE: 动态阈值计算的窗口大小
                - save_allowed_prefixes: 保存时允许的列名前缀
                - DEFAULT_DELAY_MEAN_LOW: 低延迟阈值
                - DEFAULT_DELAY_MEAN_HIGH: 高延迟阈值
                - DEFAULT_DELAY_STD_HIGH: 高延迟标准差阈值
                - DEFAULT_LOSS_MEAN_LOW: 低丢包率阈值
                - DEFAULT_LOSS_MEAN_HIGH: 高丢包率阈值
                - DEFAULT_LOSS_STD_HIGH: 高丢包率标准差阈值

        Raises:
            ValueError: 当config为None时抛出

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> config = {
            ...     'STRONG_BURST_DELAY_THRESHOLD': 400,
            ...     'STRONG_BURST_LOSS_THRESHOLD': 0.25,
            ...     'STRONG_BURST_MIN_RUN': 15,
            ...     'INSTANT_SPIKE_DELAY_THRESHOLD': 800,
            ...     'INSTANT_SPIKE_LOSS_THRESHOLD': 0.8,
            ...     'INSTANT_SPIKE_MAX_COUNT': 5,
            ...     'INSTANT_SPIKE_MAX_RATIO': 0.1,
            ...     'WEAK_BURST_LOSS_NONZERO_RATIO': 0.3,
            ...     'WEAK_BURST_MIN_CONDITIONS': 2,
            ...     'DYNAMIC_THRESHOLD_WINDOW_SIZE': 100,
            ...     'save_allowed_prefixes': ['feat_', 'window_'],
            ...     'DEFAULT_DELAY_MEAN_LOW': 50,
            ...     'DEFAULT_DELAY_MEAN_HIGH': 200,
            ...     'DEFAULT_DELAY_STD_HIGH': 100,
            ...     'DEFAULT_LOSS_MEAN_LOW': 0.01,
            ...     'DEFAULT_LOSS_MEAN_HIGH': 0.1,
            ...     'DEFAULT_LOSS_STD_HIGH': 0.05
            ... }
            >>> pattern_identifier = PatternIdentifier(method="rule", config=config)
        """
        self.method = method
        # 初始化为空列表，将在identify方法中动态获取
        self.feature_columns: List[str] = []

        # 强制要求配置
        if config is None:
            raise ValueError("Config must be provided")

        self.config = config

        # 从config中强制获取参数，不使用默认值
        self.STRONG_BURST_DELAY_THRESHOLD = self.config[
            "STRONG_BURST_DELAY_THRESHOLD"
        ]  # Strong Burst的延迟阈值
        self.STRONG_BURST_LOSS_THRESHOLD = self.config[
            "STRONG_BURST_LOSS_THRESHOLD"
        ]  # Strong Burst的丢包率阈值
        self.STRONG_BURST_MIN_RUN = self.config[
            "STRONG_BURST_MIN_RUN"
        ]  # Strong Burst的最小连续点数量

        self.INSTANT_SPIKE_DELAY_THRESHOLD = self.config[
            "INSTANT_SPIKE_DELAY_THRESHOLD"
        ]  # 瞬时峰值的延迟阈值
        self.INSTANT_SPIKE_LOSS_THRESHOLD = self.config[
            "INSTANT_SPIKE_LOSS_THRESHOLD"
        ]  # 瞬时峰值的丢包率阈值
        self.INSTANT_SPIKE_MAX_COUNT = self.config[
            "INSTANT_SPIKE_MAX_COUNT"
        ]  # 瞬时峰值的最大峰值点数量
        self.INSTANT_SPIKE_MAX_RATIO = self.config[
            "INSTANT_SPIKE_MAX_RATIO"
        ]  # 瞬时峰值的最大峰值占比

        self.WEAK_BURST_LOSS_NONZERO_RATIO = self.config[
            "WEAK_BURST_LOSS_NONZERO_RATIO"
        ]  # Weak Burst的丢包率非零比例阈值
        self.WEAK_BURST_MIN_CONDITIONS = self.config[
            "WEAK_BURST_MIN_CONDITIONS"
        ]  # Weak Burst的最小满足条件数量

        self.DYNAMIC_THRESHOLD_WINDOW_SIZE = self.config[
            "DYNAMIC_THRESHOLD_WINDOW_SIZE"
        ]  # 动态阈值计算的窗口大小
        self.SAVE_ALLOWED_PREFIXES = self.config[
            "save_allowed_prefixes"
        ]  # 保存时允许的列名前缀

        # 默认阈值常量定义，必须从配置获取
        self.DEFAULT_DELAY_MEAN_LOW = self.config[
            "DEFAULT_DELAY_MEAN_LOW"
        ]  # 低延迟阈值
        self.DEFAULT_DELAY_MEAN_HIGH = self.config[
            "DEFAULT_DELAY_MEAN_HIGH"
        ]  # 高延迟阈值
        self.DEFAULT_DELAY_STD_HIGH = self.config[
            "DEFAULT_DELAY_STD_HIGH"
        ]  # 高延迟标准差阈值
        self.DEFAULT_LOSS_MEAN_LOW = self.config[
            "DEFAULT_LOSS_MEAN_LOW"
        ]  # 低丢包率阈值
        self.DEFAULT_LOSS_MEAN_HIGH = self.config[
            "DEFAULT_LOSS_MEAN_HIGH"
        ]  # 高丢包率阈值
        self.DEFAULT_LOSS_STD_HIGH = self.config[
            "DEFAULT_LOSS_STD_HIGH"
        ]  # 高丢包率标准差阈值

        logger.debug(f"PatternIdentifier 初始化完成，使用配置: {self.config}")

    def identify(
        self, features_df: pd.DataFrame, raw_data_df: pd.DataFrame = None
    ) -> Dict:
        """使用规则事件检测引擎识别网络行为模式

        注意：
        1. 当提供 raw_data_df 时，可以检测所有行为模式，包括 INSTANT_SPIKE
        2. 当未提供 raw_data_df 时，以下行为模式无法检测：
           - INSTANT_SPIKE (瞬时峰值)
           - HIGH_DELAY_NO_LOSS (高延迟无丢包)
           - 某些依赖绝对尺度的行为模式

        Args:
            features_df (pd.DataFrame): 包含特征数据的DataFrame
            raw_data_df (pd.DataFrame, optional): 原始数据DataFrame，可选

        Returns:
            Dict: 包含识别结果的字典，包括标签、转移矩阵和各种指标
                - method: 检测方法
                - labels_up: 上行行为标签列表
                - labels_down: 下行行为标签列表
                - metrics_up: 上行行为转移质量指标
                - metrics_down: 下行行为转移质量指标
                - transition_matrix_up: 上行行为转移矩阵
                - transition_matrix_down: 下行行为转移矩阵
                - behavior_stats_up: 上行行为统计信息
                - behavior_stats_down: 下行行为统计信息
                - separation_metrics_up: 上行行为分离度指标
                - separation_metrics_down: 下行行为分离度指标
                - features: 特征矩阵
                - feature_columns: 特征列名列表

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import pandas as pd
            >>> import numpy as np
            >>> config = {
            ...     # 配置参数...
            ... }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> features_df = pd.DataFrame({
            ...     'feat_delay1_mean': np.random.normal(50, 10, 100),
            ...     'feat_loss1_mean': np.random.choice([0, 0.01, 0.05], 100),
            ...     'feat_delay2_mean': np.random.normal(60, 15, 100),
            ...     'feat_loss2_mean': np.random.choice([0, 0.01, 0.05], 100)
            ... })
            >>> results = pattern_identifier.identify(features_df)
            >>> print(f"上行标签数量: {len(results['labels_up'])}")
            >>> print(f"下行标签数量: {len(results['labels_down'])}")
        """
        # 检查输入DataFrame是否为空
        if features_df.empty:
            raise ValueError("输入特征DataFrame为空")

        # 检查必要列
        if raw_data_df is not None:
            required_cols = ["window_start", "window_end"]
            missing_cols = [
                col for col in required_cols if col not in features_df.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"raw_data_df提供时，features_df必须包含以下列: {required_cols}。缺少: {missing_cols}"
                )

            # 检查raw_data_df的索引是否为连续整数
            if not all(raw_data_df.index == range(len(raw_data_df))):
                logger.warning("raw_data_df的索引不是连续整数，将重置索引")
                raw_data_df = raw_data_df.reset_index(drop=True)

        # 动态获取特征列
        self.feature_columns = [
            col for col in features_df.columns if col.startswith("feat_")
        ]
        logger.info(f"动态获取特征列: {self.feature_columns}")

        # 检查是否有特征列
        if not self.feature_columns:
            raise ValueError("未找到特征列。期望列名以'feat_'开头")

        # 提取特征数据
        X = features_df[self.feature_columns].values

        # 计算动态阈值
        thresholds = self._calculate_dynamic_thresholds(features_df, raw_data_df)

        # 执行规则事件检测
        if self.method == "rule":
            labels_up, labels_down = self._perform_rule_based_detection(
                features_df, raw_data_df, thresholds
            )
        else:
            raise ValueError(f"未知的检测方法: {self.method}")

        # 计算上行转移矩阵和指标
        transition_matrix_up = self._calculate_transition_matrix(labels_up)
        transition_metrics_up = self._calculate_transition_metrics(transition_matrix_up)
        separation_metrics_up = self._calculate_separation_metrics(X, labels_up)
        behavior_stats_up = self._calculate_behavior_statistics(X, labels_up)

        # 计算下行转移矩阵和指标
        transition_matrix_down = self._calculate_transition_matrix(labels_down)
        transition_metrics_down = self._calculate_transition_metrics(
            transition_matrix_down
        )
        separation_metrics_down = self._calculate_separation_metrics(X, labels_down)
        behavior_stats_down = self._calculate_behavior_statistics(X, labels_down)

        # 准备结果
        results = {
            "method": self.method,
            "labels_up": labels_up.tolist(),
            "labels_down": labels_down.tolist(),
            "metrics_up": transition_metrics_up,
            "metrics_down": transition_metrics_down,
            "transition_matrix_up": transition_matrix_up.tolist(),
            "transition_matrix_down": transition_matrix_down.tolist(),
            "behavior_stats_up": behavior_stats_up,
            "behavior_stats_down": behavior_stats_down,
            "separation_metrics_up": separation_metrics_up,
            "separation_metrics_down": separation_metrics_down,
            "features": X.tolist(),
            "feature_columns": self.feature_columns,
        }

        # 如果提供了原始数据，则存储用于后续保存
        if raw_data_df is not None:
            results["raw_data"] = raw_data_df
            # 存储必要的features_df列，用于save()方法保存labeled_windows
            results["window_info"] = features_df[["window_start", "window_end"]].copy()
        else:
            results["raw_data"] = None
            results["window_info"] = None

        return results

    def _calculate_window_stats(
        self, delays: np.ndarray, loss_rates: np.ndarray, window_idx: int = None
    ) -> tuple:
        """计算窗口统计信息

        Args:
            delays (np.ndarray): 延迟数据数组
            loss_rates (np.ndarray): 丢包率数据数组
            window_idx (int, optional): 窗口索引，可选，用于日志输出

        Returns:
            tuple: 包含延迟均值、延迟标准差、丢包率均值、丢包率标准差的元组；若窗口为空返回None

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import numpy as np
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> delays = np.random.normal(50, 10, 100)
            >>> loss_rates = np.random.choice([0, 0.01, 0.05], 100)
            >>> stats = pattern_identifier._calculate_window_stats(delays, loss_rates)
            >>> delay_mean, delay_std, loss_mean, loss_std = stats
            >>> print(f"延迟均值: {delay_mean:.2f}ms, 延迟标准差: {delay_std:.2f}ms")
            >>> print(f"丢包率均值: {loss_mean:.4f}, 丢包率标准差: {loss_std:.4f}")
        """
        if len(delays) == 0 or len(loss_rates) == 0:
            # 如果窗口内没有数据，返回None表示无效窗口
            if window_idx is not None:
                logger.warning(f"窗口 {window_idx} 无数据，标记为无效窗口")
            return None
        else:
            delay_mean = np.mean(delays)
            delay_std = np.std(delays)
            loss_mean = np.mean(loss_rates)
            loss_std = np.std(loss_rates)
            return delay_mean, delay_std, loss_mean, loss_std

    def _detect_strong_burst(self, delays: np.ndarray, loss_rates: np.ndarray) -> bool:
        """检测Strong Burst行为（强突发）

        Args:
            delays (np.ndarray): 延迟数据数组
            loss_rates (np.ndarray): 丢包率数据数组

        Returns:
            bool: 是否为Strong Burst行为

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import numpy as np
            >>> config = {
            ...     'STRONG_BURST_DELAY_THRESHOLD': 400,
            ...     'STRONG_BURST_LOSS_THRESHOLD': 0.25,
            ...     'STRONG_BURST_MIN_RUN': 15,
            ...     /* 其他配置参数 */
            ... }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> # 强突发数据
            >>> delays = np.array([500, 550, 600, 580, 520] * 10)  # 连续50个点超过阈值
            >>> loss_rates = np.array([0.3, 0.35, 0.28, 0.32, 0.26] * 10)  # 连续50个点超过阈值
            >>> is_strong_burst = pattern_identifier._detect_strong_burst(delays, loss_rates)
            >>> print(f"是否为强突发: {is_strong_burst}")
            True
        """
        # 检测条件：delay ≥ 400ms且loss ≥ 0.25，连续15个点以上
        congested = (delays >= self.STRONG_BURST_DELAY_THRESHOLD) & (
            loss_rates >= self.STRONG_BURST_LOSS_THRESHOLD
        )

        # 使用公共函数计算连续True序列的最大长度
        from network_simulation.utils.loss_utils import calculate_max_consecutive_true

        max_run = calculate_max_consecutive_true(congested)

        return max_run >= self.STRONG_BURST_MIN_RUN

    def _detect_weak_burst(
        self,
        loss_mean: float,
        loss_std: float,
        delay_std: float,
        delays: np.ndarray,
        row: pd.Series,
        available_features: list,
        thresholds: Dict[str, float],
    ) -> bool:
        """检测Weak Burst行为（弱突发）

        Args:
            loss_mean (float): 原始数据计算的丢包率均值
            loss_std (float): 原始数据计算的丢包率标准差
            delay_std (float): 原始数据计算的延迟标准差
            delays (np.ndarray): 原始延迟数据数组
            row (pd.Series): 特征数据行
            available_features (list): 可用特征列表
            thresholds (Dict[str, float]): 基于原始数据计算的阈值字典

        Returns:
            bool: 是否为Weak Burst行为

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import numpy as np
            >>> import pandas as pd
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> loss_mean = 0.08
            >>> loss_std = 0.05
            >>> delay_std = 15.0
            >>> delays = np.random.normal(50, 10, 100)
            >>> row = pd.Series({})
            >>> available_features = []
            >>> thresholds = {
            ...     'loss_mean_low': 0.01,
            ...     'loss_mean_high': 0.1,
            ...     'delay_std_high': 20.0,
            ...     'loss_std_high': 0.1
            ... }
            >>> is_weak_burst = pattern_identifier._detect_weak_burst(
            ...     loss_mean, loss_std, delay_std, delays, row, available_features, thresholds
            ... )
            >>> print(f"是否为弱突发: {is_weak_burst}")
        """
        # Stable行为条件：低延迟，低丢包，低波动
        stable = (
            loss_mean <= thresholds["loss_mean_low"]
            and delay_std < thresholds["delay_std_high"] / 2
            and loss_std < thresholds["loss_std_high"] / 2
        )

        if stable:
            return False

        # 收集所有满足的条件，采用多数条件满足原则（至少2个条件）
        conditions = []

        # 条件1: 中等丢包率
        conditions.append(
            loss_mean >= thresholds["loss_mean_low"]
            and loss_mean < thresholds["loss_mean_high"]
        )

        # 条件2: 延迟上升趋势 - 使用与无原始数据版本一致的逻辑
        # 计算延迟趋势斜率，使用与FeatureExtractor相同的Theil-Sen估计，对log(1+delay)计算
        delay_trend = 0.0
        if len(delays) > 1:
            from scipy.stats import theilslopes

            x = np.arange(len(delays))
            try:
                # 确保延迟值非负，避免log1p计算错误
                delays_non_negative = np.maximum(delays, 0)
                # 使用log(1+delay)计算趋势，与FeatureExtractor保持一致
                delays_log1 = np.log1p(delays_non_negative)
                delay_trend, _, _, _ = theilslopes(delays_log1, x)
            except Exception as e:
                logger.warning(f"计算延迟趋势时出错: {e}")
        # 检测延迟上升趋势，斜率为正
        conditions.append(delay_trend > 0.0)

        # 条件3: 延迟波动大
        conditions.append(delay_std > thresholds["delay_std_high"] / 2)

        # 条件4: 丢包率波动大
        conditions.append(loss_std > thresholds["loss_std_high"] / 2)

        # 条件5: 丢包率非零比例高
        if "feat_loss_nonzero_ratio" in available_features:
            conditions.append(
                row["feat_loss_nonzero_ratio"] > self.WEAK_BURST_LOSS_NONZERO_RATIO
            )

        # 多数条件满足原则：至少满足2个条件才判定为Weak Burst
        return sum(conditions) >= self.WEAK_BURST_MIN_CONDITIONS

    def _detect_instant_spike(self, delays: np.ndarray, loss_rates: np.ndarray) -> bool:
        """检测瞬时峰值行为

        Args:
            delays (np.ndarray): 延迟数据数组
            loss_rates (np.ndarray): 丢包率数据数组

        Returns:
            bool: 是否为瞬时峰值行为

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import numpy as np
            >>> config = {
            ...     'INSTANT_SPIKE_DELAY_THRESHOLD': 800,
            ...     'INSTANT_SPIKE_LOSS_THRESHOLD': 0.8,
            ...     'INSTANT_SPIKE_MAX_COUNT': 5,
            ...     'INSTANT_SPIKE_MAX_RATIO': 0.1,
            ...     /* 其他配置参数 */
            ... }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> # 瞬时峰值数据
            >>> delays = np.array([900, 100, 120, 110, 130])  # 只有1个点超过阈值
            >>> loss_rates = np.array([0.9, 0.01, 0.02, 0.01, 0.03])  # 只有1个点超过阈值
            >>> is_instant_spike = pattern_identifier._detect_instant_spike(delays, loss_rates)
            >>> print(f"是否为瞬时峰值: {is_instant_spike}")
            True
        """
        # ≤5个点满足：loss ≥ 0.8或delay ≥ 800ms，且占比小于10%
        instant_spike_count = np.sum(
            (delays >= self.INSTANT_SPIKE_DELAY_THRESHOLD)
            | (loss_rates >= self.INSTANT_SPIKE_LOSS_THRESHOLD)
        )
        total_points = len(delays)
        spike_ratio = instant_spike_count / total_points if total_points > 0 else 0
        return (
            0 < instant_spike_count <= self.INSTANT_SPIKE_MAX_COUNT
            and spike_ratio < self.INSTANT_SPIKE_MAX_RATIO
        )

    def _detect_behavior_with_raw_data(
        self,
        i: int,
        row: pd.Series,
        delays: np.ndarray,
        loss_rates: np.ndarray,
        available_features: list,
        thresholds: Dict[str, float],
    ) -> int:
        """使用原始数据检测行为

        Args:
            i (int): 窗口索引
            row (pd.Series): 特征数据行
            delays (np.ndarray): 延迟数据数组
            loss_rates (np.ndarray): 丢包率数据数组
            available_features (list): 可用特征列表
            thresholds (Dict[str, float]): 动态计算的阈值字典

        Returns:
            int: 行为标签

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import numpy as np
            >>> import pandas as pd
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> delays = np.random.normal(50, 10, 100)
            >>> loss_rates = np.random.choice([0, 0.01, 0.05], 100)
            >>> row = pd.Series({})
            >>> available_features = []
            >>> thresholds = {
            ...     'delay_mean_low': 50,
            ...     'delay_mean_high': 200,
            ...     'loss_mean_low': 0.01,
            ...     'loss_mean_high': 0.1,
            ...     'delay_std_high': 100,
            ...     'loss_std_high': 0.05
            ... }
            >>> behavior_label = pattern_identifier._detect_behavior_with_raw_data(
            ...     0, row, delays, loss_rates, available_features, thresholds
            ... )
            >>> print(f"检测到的行为标签: {behavior_label}")
        """
        # 计算窗口统计信息
        stats = self._calculate_window_stats(delays, loss_rates, window_idx=i)

        # 如果窗口无效，标记为INVALID
        if stats is None:
            return self.BEHAVIOR_LABELS["INVALID"]

        delay_mean, delay_std, loss_mean, loss_std = stats

        # 按严重性优先级检测行为（优先级从高到低）：
        # 1. 瞬时峰值 (INSTANT_SPIKE) - 最严重的突发，延迟或丢包达到极值且持续时间极短
        if self._detect_instant_spike(delays, loss_rates):
            return self.BEHAVIOR_LABELS["INSTANT_SPIKE"]

        # 2. Strong Burst (强突发) - 高延迟高丢包且持续一段时间
        if self._detect_strong_burst(delays, loss_rates):
            return self.BEHAVIOR_LABELS["STRONG_BURST"]

        # 3. 持续高丢包 (HIGH_LOSS_STEADY) - 持续严重丢包，波动较小
        if (
            loss_mean > thresholds["loss_mean_high"]
            and loss_std < thresholds["loss_std_high"]
        ):
            return self.BEHAVIOR_LABELS["HIGH_LOSS_STEADY"]

        # 4. 高延迟无丢包 (HIGH_DELAY_NO_LOSS) - 高延迟但丢包率低，网络拥堵但无数据丢失
        if (
            delay_mean > thresholds["delay_mean_high"]
            and loss_mean <= thresholds["loss_mean_low"]
        ):
            return self.BEHAVIOR_LABELS["HIGH_DELAY_NO_LOSS"]

        # 5. 低延迟高丢包 (LOW_DELAY_HIGH_LOSS) - 低延迟但丢包率高，网络不稳定
        if (
            delay_mean < thresholds["delay_mean_low"]
            and loss_mean > thresholds["loss_mean_high"]
        ):
            return self.BEHAVIOR_LABELS["LOW_DELAY_HIGH_LOSS"]

        # 6. Weak Burst (弱突发) - 中等丢包率，延迟上升趋势，波动较大
        if self._detect_weak_burst(
            loss_mean, loss_std, delay_std, delays, row, available_features, thresholds
        ):
            return self.BEHAVIOR_LABELS["WEAK_BURST"]

        # 7. 频繁波动 (FREQUENT_FLUCTUATION) - 延迟和丢包率都有较大波动
        # 放在Weak Burst之后，避免误判，因为频繁波动的影响相对较弱
        if (
            delay_std > thresholds["delay_std_high"]
            and loss_std > thresholds["loss_std_high"]
        ):
            return self.BEHAVIOR_LABELS["FREQUENT_FLUCTUATION"]

        # 8. 默认行为：稳定 (STABLE) - 所有条件都不满足，网络状态稳定
        return self.BEHAVIOR_LABELS["STABLE"]

    def _calculate_dynamic_thresholds(
        self, features_df: pd.DataFrame, raw_data_df: pd.DataFrame = None
    ) -> Dict[str, float]:
        """计算动态阈值

        Args:
            features_df (pd.DataFrame): 包含特征数据的DataFrame
            raw_data_df (pd.DataFrame, optional): 原始数据DataFrame，可选
                - 推荐：提供原始数据，可获得更准确的阈值计算
                - 降级模式：无原始数据时，从特征列推断阈值，但精度可能下降

        Returns:
            Dict[str, float]: 计算得到的阈值字典

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import pandas as pd
            >>> import numpy as np
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> features_df = pd.DataFrame({
            ...     'feat_delay1_mean': np.random.normal(50, 10, 100),
            ...     'feat_loss1_mean': np.random.choice([0, 0.01, 0.05], 100)
            ... })
            >>> thresholds = pattern_identifier._calculate_dynamic_thresholds(features_df)
            >>> print(f"计算的动态阈值: {thresholds}")
        """
        thresholds = {}

        if raw_data_df is not None:
            # 使用原始数据计算阈值 - 推荐模式，精度更高
            logger.info("使用原始数据计算动态阈值")
            self._calculate_thresholds_from_raw_data(raw_data_df, thresholds)
        else:
            # 使用特征列计算阈值 - 降级模式，精度可能下降
            logger.warning("无原始数据，从特征列推断阈值，结果可能不够准确")
            self._calculate_thresholds_from_features(features_df, thresholds)

        # 设置默认阈值，确保所有必要的阈值都存在
        self._set_default_thresholds(thresholds)

        logger.debug(f"计算动态阈值: {thresholds}")
        return thresholds

    def _calculate_thresholds_from_raw_data(
        self, raw_data_df: pd.DataFrame, thresholds: Dict
    ) -> None:
        """从原始数据计算阈值

        Args:
            raw_data_df (pd.DataFrame): 原始数据DataFrame
            thresholds (Dict): 阈值字典，用于存储计算结果

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import pandas as pd
            >>> import numpy as np
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> raw_data_df = pd.DataFrame({
            ...     'delay1': np.random.normal(50, 10, 1000),
            ...     'loss_rate1': np.random.choice([0, 0.01, 0.05, 0.1], 1000),
            ...     'delay2': np.random.normal(60, 15, 1000),
            ...     'loss_rate2': np.random.choice([0, 0.01, 0.05, 0.1], 1000)
            ... })
            >>> thresholds = {}
            >>> pattern_identifier._calculate_thresholds_from_raw_data(raw_data_df, thresholds)
            >>> print(f"从原始数据计算的阈值: {thresholds}")
        """
        # 合并上下行数据用于计算阈值
        delays = np.concatenate(
            [raw_data_df["delay1"].values, raw_data_df["delay2"].values]
        )
        loss_rates = np.concatenate(
            [raw_data_df["loss_rate1"].values, raw_data_df["loss_rate2"].values]
        )

        # 计算分位数作为阈值
        thresholds["delay_mean_low"] = np.percentile(delays, 25)  # 低延迟阈值
        thresholds["delay_mean_high"] = np.percentile(delays, 75)  # 高延迟阈值
        thresholds["loss_mean_low"] = np.percentile(loss_rates, 25)  # 低丢包率阈值
        thresholds["loss_mean_high"] = np.percentile(loss_rates, 75)  # 高丢包率阈值

        # 安全计算延迟标准差阈值
        window_size = self.DYNAMIC_THRESHOLD_WINDOW_SIZE
        self._calculate_windowed_std_threshold(
            delays, window_size, "delay_std_high", thresholds
        )

        # 安全计算丢包率标准差阈值
        self._calculate_windowed_std_threshold(
            loss_rates, window_size, "loss_std_high", thresholds
        )

    def _calculate_windowed_std_threshold(
        self, data: np.ndarray, window_size: int, threshold_key: str, thresholds: Dict
    ) -> None:
        """计算窗口标准差阈值

        Args:
            data (np.ndarray): 输入数据数组
            window_size (int): 窗口大小
            threshold_key (str): 阈值字典中的键名
            thresholds (Dict): 阈值字典，用于存储计算结果

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import numpy as np
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> data = np.random.normal(50, 10, 1000)  # 生成1000个数据点
            >>> thresholds = {}
            >>> pattern_identifier._calculate_windowed_std_threshold(data, 100, "delay_std_high", thresholds)
            >>> print(f"计算的窗口标准差阈值: {thresholds['delay_std_high']}")
        """
        if len(data) >= window_size:
            # 使用滑动窗口计算标准差，设置 min_periods 为窗口大小的一半，避免早期噪声
            rolling_std = (
                pd.Series(data)
                .rolling(window=window_size, min_periods=window_size // 2)
                .std()
                .dropna()
                .values
            )
            thresholds[threshold_key] = np.percentile(rolling_std, 75)
        elif len(data) > 1:
            # 数据点不足一个窗口但大于1个，计算整体标准差
            thresholds[threshold_key] = np.std(data)
        else:
            # 数据点过少，使用全局默认阈值
            if threshold_key == "delay_std_high":
                thresholds[threshold_key] = self.DEFAULT_DELAY_STD_HIGH
            elif threshold_key == "loss_std_high":
                thresholds[threshold_key] = self.DEFAULT_LOSS_STD_HIGH
            else:
                thresholds[threshold_key] = 0.0

    def _calculate_thresholds_from_features(
        self, features_df: pd.DataFrame, thresholds: Dict
    ) -> None:
        """从特征列计算阈值

        注意：延迟特征（如 feat_delay1_mean）基于对数尺度 (log(1+delay)) 计算，因此延迟阈值也基于对数尺度。

        Args:
            features_df (pd.DataFrame): 包含特征数据的DataFrame
            thresholds (Dict): 阈值字典，用于存储计算结果

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import pandas as pd
            >>> import numpy as np
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> features_df = pd.DataFrame({
            ...     'feat_delay1_mean': np.random.normal(3.9, 0.2, 100),  # 对数尺度
            ...     'feat_loss1_mean': np.random.choice([0, 0.01, 0.05, 0.1], 100),
            ...     'feat_delay1_std': np.random.normal(0.1, 0.05, 100),
            ...     'feat_loss1_std': np.random.normal(0.02, 0.01, 100)
            ... })
            >>> thresholds = {}
            >>> pattern_identifier._calculate_thresholds_from_features(features_df, thresholds)
            >>> print(f"从特征列计算的阈值: {thresholds}")
        """
        # 收集所有延迟和丢包特征，用于计算阈值
        delay_std_features = []
        delay_mean_features = []
        loss_std_features = []
        loss_mean_features = []

        for feat in self.feature_columns:
            if "delay" in feat:
                if "std" in feat:
                    delay_std_features.append(feat)
                elif "mean" in feat:
                    delay_mean_features.append(feat)
            elif "loss" in feat:
                if "std" in feat:
                    loss_std_features.append(feat)
                elif "mean" in feat:
                    loss_mean_features.append(feat)

        # 计算延迟相关阈值 - 基于对数尺度特征
        if delay_mean_features:
            # 合并所有延迟均值特征数据
            all_delay_means = np.concatenate(
                [features_df[feat].values for feat in delay_mean_features]
            )
            # 从对数尺度特征分布中计算阈值
            thresholds["delay_mean_low"] = np.percentile(all_delay_means, 25)
            thresholds["delay_mean_high"] = np.percentile(all_delay_means, 75)
            logger.debug(
                f"基于对数尺度特征计算延迟阈值: delay_mean_low={thresholds['delay_mean_low']:.4f}, delay_mean_high={thresholds['delay_mean_high']:.4f}"
            )

        if delay_std_features:
            # 合并所有延迟标准差特征数据
            all_delay_stds = np.concatenate(
                [features_df[feat].values for feat in delay_std_features]
            )
            thresholds["delay_std_high"] = np.percentile(all_delay_stds, 75)
            logger.debug(
                f"基于对数尺度特征计算延迟标准差阈值: delay_std_high={thresholds['delay_std_high']:.4f}"
            )

        # 计算丢包相关阈值
        if loss_mean_features:
            # 合并所有丢包均值特征数据
            all_loss_means = np.concatenate(
                [features_df[feat].values for feat in loss_mean_features]
            )
            thresholds["loss_mean_low"] = np.percentile(all_loss_means, 25)
            thresholds["loss_mean_high"] = np.percentile(all_loss_means, 75)
            logger.debug(
                f"计算丢包率均值阈值: loss_mean_low={thresholds['loss_mean_low']:.4f}, loss_mean_high={thresholds['loss_mean_high']:.4f}"
            )

        if loss_std_features:
            # 合并所有丢包标准差特征数据
            all_loss_stds = np.concatenate(
                [features_df[feat].values for feat in loss_std_features]
            )
            thresholds["loss_std_high"] = np.percentile(all_loss_stds, 75)
            logger.debug(
                f"计算丢包率标准差阈值: loss_std_high={thresholds['loss_std_high']:.4f}"
            )

    def _set_default_thresholds(self, thresholds: Dict) -> None:
        """设置默认阈值

        Args:
            thresholds (Dict): 阈值字典，用于存储默认值

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> config = {
            ...     'DEFAULT_DELAY_MEAN_LOW': 50,
            ...     'DEFAULT_DELAY_MEAN_HIGH': 200,
            ...     'DEFAULT_DELAY_STD_HIGH': 100,
            ...     'DEFAULT_LOSS_MEAN_LOW': 0.01,
            ...     'DEFAULT_LOSS_MEAN_HIGH': 0.1,
            ...     'DEFAULT_LOSS_STD_HIGH': 0.05,
            ...     /* 其他配置参数 */
            ... }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> thresholds = {'delay_mean_low': 45}
            >>> pattern_identifier._set_default_thresholds(thresholds)
            >>> print(f"设置默认阈值后: {thresholds}")
        """
        thresholds.setdefault("delay_mean_low", self.DEFAULT_DELAY_MEAN_LOW)
        thresholds.setdefault("delay_mean_high", self.DEFAULT_DELAY_MEAN_HIGH)
        thresholds.setdefault("delay_std_high", self.DEFAULT_DELAY_STD_HIGH)
        thresholds.setdefault("loss_mean_low", self.DEFAULT_LOSS_MEAN_LOW)
        thresholds.setdefault("loss_mean_high", self.DEFAULT_LOSS_MEAN_HIGH)
        thresholds.setdefault("loss_std_high", self.DEFAULT_LOSS_STD_HIGH)

    def _detect_behavior_without_raw_data(
        self, row: pd.Series, available_features: list, thresholds: Dict[str, float]
    ) -> int:
        """无原始数据时检测行为

        注意：
        1. 延迟特征（如 feat_delay1_mean）基于对数尺度 (log(1+delay)) 计算
        2. 延迟相关阈值（如 delay_mean_low）也基于对数尺度
        3. 因此直接比较特征值和阈值是合理的，无需反变换

        Args:
            row (pd.Series): 特征数据行
            available_features (list): 可用特征列表
            thresholds (Dict[str, float]): 动态计算的阈值字典

        Returns:
            int: 行为标签

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import pandas as pd
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> row = pd.Series({
            ...     'feat_loss1_mean': 0.08,
            ...     'feat_delay1_std': 0.15,
            ...     'feat_loss1_std': 0.03,
            ...     'feat_max_congestion_run1': 5
            ... })
            >>> available_features = ['feat_loss1_mean', 'feat_delay1_std', 'feat_loss1_std', 'feat_max_congestion_run1']
            >>> thresholds = {
            ...     'loss_mean_high': 0.1,
            ...     'delay_std_high': 0.2,
            ...     'loss_std_high': 0.05
            ... }
            >>> behavior_label = pattern_identifier._detect_behavior_without_raw_data(row, available_features, thresholds)
            >>> print(f"检测到的行为标签: {behavior_label}")
        """
        # 注意：特征值可能已标准化，部分行为判定受限

        # 检查是否存在上下行特征
        has_up_down_features = any(
            col.endswith("1") or col.endswith("2") for col in available_features
        )

        # 选择要使用的特征列
        if has_up_down_features:
            # 优先使用上行特征进行单通道检测（可扩展为同时检查上下行）
            loss_mean_col = (
                "feat_loss1_mean"
                if "feat_loss1_mean" in available_features
                else "feat_loss2_mean"
            )
            delay_std_col = (
                "feat_delay1_std"
                if "feat_delay1_std" in available_features
                else "feat_delay2_std"
            )
            loss_std_col = (
                "feat_loss1_std"
                if "feat_loss1_std" in available_features
                else "feat_loss2_std"
            )
            # 基于绝对延迟值的行为判断必须使用原始尺度
            # 只使用原始尺度延迟均值，不 fallback 到对数尺度
            delay_mean_col = None
            if "feat_delay1_raw_mean" in available_features:
                delay_mean_col = "feat_delay1_raw_mean"
            elif "feat_delay2_raw_mean" in available_features:
                delay_mean_col = "feat_delay2_raw_mean"
            congestion_run_col = (
                "feat_max_congestion_run1"
                if "feat_max_congestion_run1" in available_features
                else "feat_max_congestion_run2"
            )
            loss_nonzero_ratio_col = (
                "feat_loss1_nonzero_ratio"
                if "feat_loss1_nonzero_ratio" in available_features
                else "feat_loss2_nonzero_ratio"
            )
        else:
            # 使用单通道特征
            loss_mean_col = "feat_loss_mean"
            delay_std_col = "feat_delay_std"
            loss_std_col = "feat_loss_std"
            # 基于绝对延迟值的行为判断必须使用原始尺度
            # 只使用原始尺度延迟均值，不 fallback 到对数尺度
            delay_mean_col = (
                "feat_delay_raw_mean"
                if "feat_delay_raw_mean" in available_features
                else None
            )
            congestion_run_col = "feat_max_congestion_run"
            loss_nonzero_ratio_col = "feat_loss_nonzero_ratio"

        # 检测行为 5: 持续高丢包
        if (
            loss_mean_col in available_features
            and row[loss_mean_col] > thresholds["loss_mean_high"]
        ):
            return self.BEHAVIOR_LABELS["HIGH_LOSS_STEADY"]
        # 检测行为 6: 频繁波动
        elif (
            delay_std_col in available_features
            and row[delay_std_col] > thresholds["delay_std_high"]
        ) and (
            loss_std_col in available_features
            and row[loss_std_col] > thresholds["loss_std_high"]
        ):
            return self.BEHAVIOR_LABELS["FREQUENT_FLUCTUATION"]
        # 检测行为 7: 低延迟高丢包 - 只有当原始尺度延迟均值存在时才检测
        elif (
            delay_mean_col is not None
            and delay_mean_col in available_features
            and row[delay_mean_col] < thresholds["delay_mean_low"]
        ) and (
            loss_mean_col in available_features
            and row[loss_mean_col] > thresholds["loss_mean_high"]
        ):
            return self.BEHAVIOR_LABELS["LOW_DELAY_HIGH_LOSS"]
        # 检测行为 2: Strong Burst
        elif (
            congestion_run_col in available_features
            and row[congestion_run_col] >= self.STRONG_BURST_MIN_RUN
        ):
            return self.BEHAVIOR_LABELS["STRONG_BURST"]
        # 检测行为 1: Weak Burst
        elif (
            loss_nonzero_ratio_col in available_features
            and row[loss_nonzero_ratio_col] > self.WEAK_BURST_LOSS_NONZERO_RATIO
        ):
            return self.BEHAVIOR_LABELS["WEAK_BURST"]
        # 默认为Stable
        # 注意：在无原始数据时，行为3（瞬时峰值）和行为4（高延迟无丢包）未被检测
        # 这些行为需要原始数据的绝对尺度，特征值可能已标准化
        else:
            return self.BEHAVIOR_LABELS["STABLE"]

    def _perform_rule_based_detection(
        self,
        features_df: pd.DataFrame,
        raw_data_df: pd.DataFrame = None,
        thresholds: Dict[str, float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """执行规则事件检测，识别网络行为模式

        重要说明：
        - 该方法返回上下行独立的行为标签数组，这是因为网络的上行和下行可能表现出不同的行为模式
        - 建议用户分别处理上下行标签，以获得更准确的网络状态分析
        - 如果上下行特征相同，上下行标签也会相同

        Args:
            features_df (pd.DataFrame): 包含特征数据的DataFrame，当提供raw_data_df时，必须包含window_start和window_end列
            raw_data_df (pd.DataFrame, optional): 原始数据DataFrame，可选，其索引必须是连续的行号[0, 1, 2, ...]
            thresholds (Dict[str, float], optional): 动态计算的阈值字典

        Returns:
            tuple[np.ndarray, np.ndarray]: 上行行为标签数组和下行行为标签数组
            - 第一个数组为上行行为标签
            - 第二个数组为下行行为标签

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import pandas as pd
            >>> import numpy as np
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> features_df = pd.DataFrame({
            ...     'feat_delay1_mean': np.random.normal(3.9, 0.2, 100),
            ...     'feat_loss1_mean': np.random.choice([0, 0.01, 0.05], 100),
            ...     'feat_delay2_mean': np.random.normal(4.0, 0.25, 100),
            ...     'feat_loss2_mean': np.random.choice([0, 0.01, 0.05, 0.1], 100)
            ... })
            >>> labels_up, labels_down = pattern_identifier._perform_rule_based_detection(features_df)
            >>> print(f"上行标签数量: {len(labels_up)}, 下行标签数量: {len(labels_down)}")
        """
        if thresholds is None:
            # 如果未提供阈值，重新计算
            thresholds = self._calculate_dynamic_thresholds(features_df, raw_data_df)
        logger.info("开始规则事件检测")

        # 获取可用的特征列
        available_features = features_df.columns.tolist()

        # 逐个窗口进行行为判定
        if raw_data_df is not None:
            # 有原始数据时，需要逐窗口处理原始数据
            labels_up, labels_down = self._detect_with_raw_data(
                features_df, raw_data_df, available_features, thresholds
            )
        else:
            # 无原始数据时，使用向量化操作进行批量处理，提高性能
            logger.warning(
                "未提供 raw_data_df，部分行为（如 INSTANT_SPIKE, HIGH_DELAY_NO_LOSS, LOW_DELAY_HIGH_LOSS）无法检测"
            )
            labels_up, labels_down = self._detect_without_raw_data(
                features_df, available_features, thresholds
            )

        logger.info("规则事件检测完成")

        # 使用np.unique替代np.bincount，处理非连续标签
        # 上行标签统计
        unique_labels_up, counts_up = np.unique(labels_up, return_counts=True)
        label_stats_up = {
            int(lbl): int(cnt) for lbl, cnt in zip(unique_labels_up, counts_up)
        }
        logger.info(f"上行行为标签统计: {label_stats_up}")

        # 下行标签统计
        unique_labels_down, counts_down = np.unique(labels_down, return_counts=True)
        label_stats_down = {
            int(lbl): int(cnt) for lbl, cnt in zip(unique_labels_down, counts_down)
        }
        logger.info(f"下行行为标签统计: {label_stats_down}")

        return labels_up, labels_down

    def _detect_with_raw_data(
        self,
        features_df: pd.DataFrame,
        raw_data_df: pd.DataFrame,
        available_features: list,
        thresholds: Dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """使用原始数据进行行为检测

        Args:
            features_df (pd.DataFrame): 包含特征数据的DataFrame
            raw_data_df (pd.DataFrame): 原始数据DataFrame
            available_features (list): 可用特征列表
            thresholds (Dict[str, float]): 动态计算的阈值字典

        Returns:
            tuple[np.ndarray, np.ndarray]: 上行行为标签数组和下行行为标签数组

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import pandas as pd
            >>> import numpy as np
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> features_df = pd.DataFrame({
            ...     'window_start': [0, 50, 100],
            ...     'window_end': [50, 100, 150],
            ...     'feat_delay1_mean': [3.9, 4.2, 3.8],
            ...     'feat_loss1_mean': [0.01, 0.25, 0.02]
            ... })
            >>> raw_data_df = pd.DataFrame({
            ...     'delay1': np.random.normal(50, 10, 150),
            ...     'loss_rate1': np.random.choice([0, 0.01, 0.25], 150),
            ...     'delay2': np.random.normal(60, 15, 150),
            ...     'loss_rate2': np.random.choice([0, 0.01, 0.05], 150)
            ... })
            >>> labels_up, labels_down = pattern_identifier._detect_with_raw_data(
            ...     features_df, raw_data_df, features_df.columns.tolist(), {})
            >>> print(f"上行标签: {labels_up}, 下行标签: {labels_down}")
        """
        # 初始化上下行行为标签数组，默认为STABLE
        labels_up = np.full(len(features_df), self.BEHAVIOR_LABELS["STABLE"], dtype=int)
        labels_down = np.full(
            len(features_df), self.BEHAVIOR_LABELS["STABLE"], dtype=int
        )

        # 假设所有数据都包含完整的上下行数据
        logger.debug("检测到完整的上下行数据")

        # 有原始数据时，需要逐窗口处理原始数据
        for i, (_, row) in enumerate(features_df.iterrows()):
            # 获取当前窗口的原始数据，用于计算更详细的统计信息
            window_start = int(row["window_start"])
            window_end = int(row["window_end"])

            # 检查窗口有效性
            if window_start >= window_end or window_end > len(raw_data_df):
                logger.warning(
                    f"窗口 {i} 无效: window_start={window_start}, window_end={window_end}, raw_data_length={len(raw_data_df)}"
                )
                labels_up[i] = self.BEHAVIOR_LABELS["INVALID"]
                labels_down[i] = self.BEHAVIOR_LABELS["INVALID"]
                continue  # 直接跳过无效窗口，避免冗余计算

            # 提取窗口数据
            window_data = raw_data_df.iloc[window_start:window_end]

            # 上行数据
            delay1 = window_data["delay1"].values
            loss_rate1 = window_data["loss_rate1"].values

            # 下行数据
            delay2 = window_data["delay2"].values
            loss_rate2 = window_data["loss_rate2"].values

            # 检查窗口数据是否为空
            if (
                len(delay1) == 0
                or len(loss_rate1) == 0
                or len(delay2) == 0
                or len(loss_rate2) == 0
            ):
                logger.warning(f"窗口 {i} 数据为空，标记为无效窗口")
                labels_up[i] = self.BEHAVIOR_LABELS["INVALID"]
                labels_down[i] = self.BEHAVIOR_LABELS["INVALID"]
                continue  # 直接跳过空数据窗口

            # 对上下行数据分别检测行为
            # 上行行为
            behavior1 = self._detect_behavior_with_raw_data(
                i, row, delay1, loss_rate1, available_features, thresholds
            )

            # 下行行为
            behavior2 = self._detect_behavior_with_raw_data(
                i, row, delay2, loss_rate2, available_features, thresholds
            )

            # 分别存储上下行行为标签
            labels_up[i] = behavior1
            labels_down[i] = behavior2

        return labels_up, labels_down

    def _detect_without_raw_data(
        self,
        features_df: pd.DataFrame,
        available_features: list,
        thresholds: Dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """不使用原始数据进行行为检测（向量化操作）

        该函数使用向量化操作，基于提取的特征直接进行行为检测，无需访问原始数据。
        支持上下行数据的单独检测和综合分析。

        Args:
            features_df (pd.DataFrame): 包含特征数据的DataFrame
            available_features (list): 可用特征列表
            thresholds (Dict[str, float]): 动态计算的阈值字典，用于行为检测

        Returns:
            tuple[np.ndarray, np.ndarray]: 上行行为标签数组和下行行为标签数组

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import pandas as pd
            >>> import numpy as np
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> features_df = pd.DataFrame({
            ...     'feat_loss1_mean': [0.01, 0.25, 0.02],
            ...     'feat_delay1_std': [0.1, 0.3, 0.15],
            ...     'feat_loss1_std': [0.005, 0.05, 0.01],
            ...     'feat_max_congestion_run1': [0, 15, 0],
            ...     'feat_loss2_mean': [0.02, 0.03, 0.15],
            ...     'feat_delay2_std': [0.12, 0.18, 0.25],
            ...     'feat_loss2_std': [0.01, 0.015, 0.03],
            ...     'feat_max_congestion_run2': [0, 0, 8]
            ... })
            >>> available_features = features_df.columns.tolist()
            >>> thresholds = {
            ...     'loss_mean_high': 0.1,
            ...     'delay_std_high': 0.2,
            ...     'loss_std_high': 0.05
            ... }
            >>> labels_up, labels_down = pattern_identifier._detect_without_raw_data(features_df, available_features, thresholds)
            >>> print(f"上行标签: {labels_up}, 下行标签: {labels_down}")
        """
        logger.info("无原始数据，使用向量化操作进行批量行为检测")

        # 初始化上下行行为标签
        labels_up = np.full(len(features_df), self.BEHAVIOR_LABELS["STABLE"], dtype=int)
        labels_down = np.full(
            len(features_df), self.BEHAVIOR_LABELS["STABLE"], dtype=int
        )

        # 检测上行行为
        if self._has_up_features(available_features):
            logger.debug("检测到上行特征，开始上行行为检测")
            # 检测行为 5: 持续高丢包
            if "feat_loss1_mean" in available_features:
                high_loss_mask1 = (
                    features_df["feat_loss1_mean"] > thresholds["loss_mean_high"]
                )
                labels_up[high_loss_mask1] = self.BEHAVIOR_LABELS["HIGH_LOSS_STEADY"]

            # 检测行为 6: 频繁波动
            if (
                "feat_delay1_std" in available_features
                and "feat_loss1_std" in available_features
            ):
                frequent_fluctuation_mask1 = (
                    features_df["feat_delay1_std"] > thresholds["delay_std_high"]
                ) & (features_df["feat_loss1_std"] > thresholds["loss_std_high"])
                labels_up[frequent_fluctuation_mask1] = self.BEHAVIOR_LABELS[
                    "FREQUENT_FLUCTUATION"
                ]

            # 检测行为 7: 低延迟高丢包 - 只使用原始尺度延迟均值
            # 基于绝对延迟值的行为判断必须使用原始尺度
            if (
                "feat_delay1_raw_mean" in available_features
                and "feat_loss1_mean" in available_features
            ):
                low_delay_high_loss_mask1 = (
                    features_df["feat_delay1_raw_mean"] < thresholds["delay_mean_low"]
                ) & (features_df["feat_loss1_mean"] > thresholds["loss_mean_high"])
                labels_up[low_delay_high_loss_mask1] = self.BEHAVIOR_LABELS[
                    "LOW_DELAY_HIGH_LOSS"
                ]

            # 检测行为 2: Strong Burst
            if "feat_max_congestion_run1" in available_features:
                strong_burst_mask1 = (
                    features_df["feat_max_congestion_run1"] >= self.STRONG_BURST_MIN_RUN
                )
                labels_up[strong_burst_mask1] = self.BEHAVIOR_LABELS["STRONG_BURST"]

            # 检测行为 1: Weak Burst
            weak_burst_mask1 = self._detect_weak_burst_without_raw_data(
                features_df, available_features, thresholds, direction="up"
            )
            labels_up[weak_burst_mask1] = self.BEHAVIOR_LABELS["WEAK_BURST"]
        else:
            logger.warning("未检测到完整的上行特征，跳过上行行为检测")

        # 检测下行行为
        if self._has_down_features(available_features):
            logger.debug("检测到下行特征，开始下行行为检测")
            # 检测行为 5: 持续高丢包
            if "feat_loss2_mean" in available_features:
                high_loss_mask2 = (
                    features_df["feat_loss2_mean"] > thresholds["loss_mean_high"]
                )
                labels_down[high_loss_mask2] = self.BEHAVIOR_LABELS["HIGH_LOSS_STEADY"]

            # 检测行为 6: 频繁波动
            if (
                "feat_delay2_std" in available_features
                and "feat_loss2_std" in available_features
            ):
                frequent_fluctuation_mask2 = (
                    features_df["feat_delay2_std"] > thresholds["delay_std_high"]
                ) & (features_df["feat_loss2_std"] > thresholds["loss_std_high"])
                labels_down[frequent_fluctuation_mask2] = self.BEHAVIOR_LABELS[
                    "FREQUENT_FLUCTUATION"
                ]

            # 检测行为 7: 低延迟高丢包 - 只使用原始尺度延迟均值
            # 基于绝对延迟值的行为判断必须使用原始尺度
            if (
                "feat_delay2_raw_mean" in available_features
                and "feat_loss2_mean" in available_features
            ):
                low_delay_high_loss_mask2 = (
                    features_df["feat_delay2_raw_mean"] < thresholds["delay_mean_low"]
                ) & (features_df["feat_loss2_mean"] > thresholds["loss_mean_high"])
                labels_down[low_delay_high_loss_mask2] = self.BEHAVIOR_LABELS[
                    "LOW_DELAY_HIGH_LOSS"
                ]

            # 检测行为 2: Strong Burst
            if "feat_max_congestion_run2" in available_features:
                strong_burst_mask2 = (
                    features_df["feat_max_congestion_run2"] >= self.STRONG_BURST_MIN_RUN
                )
                labels_down[strong_burst_mask2] = self.BEHAVIOR_LABELS["STRONG_BURST"]

            # 检测行为 1: Weak Burst
            weak_burst_mask2 = self._detect_weak_burst_without_raw_data(
                features_df, available_features, thresholds, direction="down"
            )
            labels_down[weak_burst_mask2] = self.BEHAVIOR_LABELS["WEAK_BURST"]
        else:
            logger.warning("未检测到完整的下行特征，跳过下行行为检测")

        return labels_up, labels_down

    def _has_up_features(self, available_features: list) -> bool:
        """检查是否存在足够的上行特征用于行为检测

        Args:
            available_features (list): 可用特征列表

        Returns:
            bool: 是否存在足够的上行特征

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> # 有上行特征
            >>> available_features = ['feat_delay1_mean', 'feat_loss1_mean', 'feat_delay2_std']
            >>> has_up = pattern_identifier._has_up_features(available_features)
            >>> print(f"是否有上行特征: {has_up}")  # 是
            >>> # 没有上行特征
            >>> available_features = ['feat_delay2_mean', 'feat_loss2_mean']
            >>> has_up = pattern_identifier._has_up_features(available_features)
            >>> print(f"是否有上行特征: {has_up}")  # 否
        """
        # 检查是否至少存在一个上行特征
        up_features = [
            col
            for col in available_features
            if "1" in col and ("delay" in col or "loss" in col)
        ]
        return len(up_features) > 0

    def _has_down_features(self, available_features: list) -> bool:
        """检查是否存在足够的下行特征用于行为检测

        Args:
            available_features: 可用特征列表

        Returns:
            bool: 是否存在足够的下行特征
        """
        # 检查是否至少存在一个下行特征
        down_features = [
            col
            for col in available_features
            if "2" in col and ("delay" in col or "loss" in col)
        ]
        return len(down_features) > 0

    def _detect_weak_burst_without_raw_data(
        self,
        features_df: pd.DataFrame,
        available_features: list,
        thresholds: Dict[str, float],
        direction: str = None,
    ) -> np.ndarray:
        """无原始数据时检测Weak Burst行为

        Args:
            features_df: 包含特征数据的DataFrame
            available_features: 可用特征列表
            thresholds: 动态计算的阈值字典
            direction: 方向，"up"表示上行，"down"表示下行，None表示单通道

        Returns:
            np.ndarray: Weak Burst行为的掩码数组
        """
        # 确定使用的特征后缀
        suffix = "" if direction is None else "1" if direction == "up" else "2"

        # 获取特征列名
        loss_mean_col = f"feat_loss{suffix}_mean"
        delay_std_col = f"feat_delay{suffix}_std"
        loss_std_col = f"feat_loss{suffix}_std"
        loss_nonzero_col = f"feat_loss{suffix}_nonzero_ratio"
        delay_trend_col = f"feat_delay{suffix}_trend"

        # Stable行为条件：低延迟，低丢包，低波动
        stable_mask = np.ones(len(features_df), dtype=bool)
        if loss_mean_col in available_features:
            stable_mask &= features_df[loss_mean_col] <= thresholds["loss_mean_low"]
        if delay_std_col in available_features:
            stable_mask &= features_df[delay_std_col] < thresholds["delay_std_high"] / 2
        if loss_std_col in available_features:
            stable_mask &= features_df[loss_std_col] < thresholds["loss_std_high"] / 2

        # 收集所有满足的条件，采用与带原始数据版本一致的条件顺序和判定原则
        weak_burst_conditions = []

        # 条件1: 中等丢包率
        if loss_mean_col in available_features:
            condition1 = (features_df[loss_mean_col] >= thresholds["loss_mean_low"]) & (
                features_df[loss_mean_col] < thresholds["loss_mean_high"]
            )
            weak_burst_conditions.append(condition1)

        # 条件2: 延迟上升趋势
        if delay_trend_col in available_features:
            condition2 = features_df[delay_trend_col] > 0.0
            weak_burst_conditions.append(condition2)

        # 条件3: 丢包率波动大
        if loss_std_col in available_features:
            condition3 = features_df[loss_std_col] > thresholds["loss_std_high"] / 2
            weak_burst_conditions.append(condition3)

        # 条件4: 延迟波动大
        if delay_std_col in available_features:
            condition4 = features_df[delay_std_col] > thresholds["delay_std_high"] / 2
            weak_burst_conditions.append(condition4)

        # 条件5: 丢包率非零比例高
        if loss_nonzero_col in available_features:
            condition5 = (
                features_df[loss_nonzero_col] > self.WEAK_BURST_LOSS_NONZERO_RATIO
            )
            weak_burst_conditions.append(condition5)

        # 应用多数条件满足原则（至少2个条件），并且排除Stable窗口
        if weak_burst_conditions:
            return (
                np.sum(weak_burst_conditions, axis=0) >= self.WEAK_BURST_MIN_CONDITIONS
            ) & (~stable_mask)
        else:
            return np.zeros(len(features_df), dtype=bool)

    def _calculate_transition_metrics(self, transition_matrix: np.ndarray) -> Dict:
        """计算行为转移质量指标

        Args:
            transition_matrix: 行为转移矩阵

        Returns:
            Dict: 包含转移熵、转移稀疏性和典型路径的字典
        """
        metrics = {}

        # 平均转移熵
        entropy = 0.0
        for row in transition_matrix:
            # 移除零概率值以避免log(0)
            row = row[row > 0]
            if len(row) > 0:
                entropy -= np.sum(row * np.log2(row))
        metrics["average_transition_entropy"] = float(entropy / len(transition_matrix))

        # 转移稀疏性
        total_elements = transition_matrix.size
        non_zero_elements = np.count_nonzero(transition_matrix)
        metrics["transition_sparsity"] = float(non_zero_elements / total_elements)

        # 典型路径分析 - 找出最可能的转移路径
        # 对于每个状态，找出前3个最可能的下一个状态，跳过全零行
        typical_paths = {}
        num_states = transition_matrix.shape[0]
        for i in range(num_states):
            # 检查该行是否全为零
            if np.all(transition_matrix[i, :] == 0):
                continue  # 跳过全零行

            # 获取前3个最可能的下一个状态
            top_indices = np.argsort(transition_matrix[i, :])[::-1][:3]
            top_probs = transition_matrix[i, top_indices]
            typical_paths[str(i)] = {
                "next_states": [int(idx) for idx in top_indices],
                "probabilities": [float(prob) for prob in top_probs],
            }
        metrics["typical_paths"] = typical_paths

        return metrics

    def _calculate_separation_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """计算行为分离度指标

        Args:
            X: 特征数据
            labels: 行为标签

        Returns:
            Dict: 包含各类别特征均值和标准差的字典
        """
        unique_labels = np.unique(labels)
        separation_metrics = {"label_means": {}, "label_stds": {}}

        for label in unique_labels:
            cluster_data = X[labels == label]
            separation_metrics["label_means"][str(label)] = cluster_data.mean(
                axis=0
            ).tolist()
            separation_metrics["label_stds"][str(label)] = cluster_data.std(
                axis=0
            ).tolist()

        return separation_metrics

    def identify_and_save(
        self,
        input_features_path: Path,
        input_processed_path: Path,
        output_patterns_dir: Path,
    ) -> Dict:
        """从特征数据中识别行为模式并保存结果

        Args:
            input_features_path: 特征数据文件路径
            input_processed_path: 处理后的数据文件路径
            output_patterns_dir: 行为模式结果的输出目录路径

        Returns:
            Dict: 识别结果字典
        """
        logger.info(
            f"开始识别行为模式并保存结果，输入特征: {input_features_path}, 输入处理数据: {input_processed_path}, 输出: {output_patterns_dir}"
        )

        # 加载特征数据
        features_df = pd.read_csv(input_features_path)

        # 加载处理后的数据
        processed_df = pd.read_csv(input_processed_path, parse_dates=["timestamp"])

        # 识别行为模式
        patterns = self.identify(features_df, processed_df)

        # 创建输出目录结构
        output_patterns_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir = output_patterns_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # 添加可视化
        from network_simulation.visualization.visualizer import Visualizer

        visualizer = Visualizer(output_patterns_dir)

        # 提取特征矩阵
        feature_columns = [
            col for col in features_df.columns if col.startswith("feat_")
        ]
        X = features_df[feature_columns].values

        # 生成HTML报告和可视化
        visualizer.generate_html_report(
            patterns, X, feature_columns, raw_data=processed_df, features_df=features_df
        )

        # 保存模式识别结果
        self.save(patterns, output_patterns_dir)

        logger.info(f"行为模式已保存到: {output_patterns_dir}")
        return patterns

    def _calculate_transition_matrix(self, labels: np.ndarray) -> np.ndarray:
        """计算状态转移矩阵

        Args:
            labels: 行为标签数组

        Returns:
            np.ndarray: 归一化的状态转移矩阵，只包含有效行为标签
        """
        # 过滤掉INVALID标签，只保留有效标签
        valid_mask = labels != self.BEHAVIOR_LABELS["INVALID"]
        valid_labels = labels[valid_mask]

        # 显式定义状态空间：所有有效行为标签（排除INVALID）
        # 使用列表推导式获取所有非INVALID的标签值
        valid_behavior_ids = [
            v for k, v in self.BEHAVIOR_LABELS.items() if k != "INVALID"
        ]
        # 按顺序排序，确保一致性
        valid_behavior_ids.sort()
        num_clusters = len(valid_behavior_ids)

        # 创建从标签值到矩阵索引的映射
        label_to_index = {lbl: i for i, lbl in enumerate(valid_behavior_ids)}

        transition_matrix = np.zeros((num_clusters, num_clusters))

        # 只计算有效标签之间的转移
        if len(valid_labels) > 1:
            for i in range(len(valid_labels) - 1):
                current_state = valid_labels[i]
                next_state = valid_labels[i + 1]
                # 确保当前状态和下一状态都在有效行为标签中
                if (
                    current_state in valid_behavior_ids
                    and next_state in valid_behavior_ids
                ):
                    # 将标签映射到索引
                    current_idx = label_to_index[current_state]
                    next_idx = label_to_index[next_state]
                    transition_matrix[current_idx, next_idx] += 1

        # 归一化行以获取概率
        row_sums = transition_matrix.sum(axis=1, keepdims=True)

        # 处理行和为0的情况：保持全零行不变，不引入虚假转移概率
        # 全零行表示该状态不会转移到任何其他状态（吸收态）
        for i in range(len(row_sums)):
            if row_sums[i, 0] != 0:
                transition_matrix[i, :] = transition_matrix[i, :] / row_sums[i, 0]

        return transition_matrix

    def _calculate_behavior_statistics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """计算每个行为的统计信息

        Args:
            X: 特征数据
            labels: 行为标签

        Returns:
            Dict: 包含每个行为统计信息的字典
        """
        behavior_stats = {}
        unique_labels = np.unique(labels)

        for label in unique_labels:
            behavior_data = X[labels == label]
            behavior_stats[str(label)] = {
                "mean": behavior_data.mean(axis=0).tolist(),
                "std": behavior_data.std(axis=0).tolist(),
                "min": behavior_data.min(axis=0).tolist(),
                "max": behavior_data.max(axis=0).tolist(),
                "count": int(len(behavior_data)),
            }

        return behavior_stats

    def save(
        self,
        results: Dict,
        output_dir: Path,
        valid_loss_values: list = None,
        save_all_columns: bool = True,
    ) -> None:
        """保存规则检测结果到文件

        该函数将行为检测的结果保存到指定目录，包括标签文件、行为统计信息、转移矩阵等。
        支持多种检测方法的结果保存。

        Args:
            results: 检测结果字典，包含labels_up、labels_down、method、behavior_stats等字段
            output_dir: 输出目录，用于保存所有结果文件
            valid_loss_values: 合法丢包值列表，可选，用于覆盖自动推导的值
                - 预期格式：[0.0, 0.1, 0.2, ..., 1.0]，表示网络中可能出现的所有丢包率值
                - 使用场景：当自动推导的丢包值不完整或不准确时，可手动指定
                - 示例：valid_loss_values=[0.0, 0.5, 1.0] 表示网络中只有0%、50%和100%三种丢包率
            save_all_columns: 是否保存所有列，包括timestamp等非数值列
                - 默认值：True，保持向后兼容
                - 设置为False时，只保存数值列（delay*、loss_rate*、bandwidth*），节省存储空间

        保存的文件包括：
            - labels_rule_up.npy: 上行行为标签数组
            - labels_rule_down.npy: 下行行为标签数组
            - window_features_rule.npy: 窗口特征矩阵
            - behavior_statistics_up.json: 上行行为统计信息
            - behavior_statistics_down.json: 下行行为统计信息
            - transition_matrix_rule_up.json: 上行行为转移矩阵
            - transition_matrix_rule_down.json: 下行行为转移矩阵
            - valid_loss_values.json: 合法丢包值列表
            - labeled_windows/: 包含每个行为类别的窗口数据，按上下行分离
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        method = results["method"]
        labels_up = np.array(results["labels_up"])
        labels_down = np.array(results["labels_down"])

        # 保存上下行标签为独立的npy文件
        if method == "rule":
            # 保存上行标签
            labels_up_file = output_dir / "labels_rule_up.npy"
            np.save(labels_up_file, labels_up)
            logger.info(f"保存上行标签为npy文件: {labels_up_file}")

            # 保存下行标签
            labels_down_file = output_dir / "labels_rule_down.npy"
            np.save(labels_down_file, labels_down)
            logger.info(f"保存下行标签为npy文件: {labels_down_file}")

            # 保存窗口特征矩阵
            if "features" in results and results["features"]:
                features = np.array(results["features"])
                features_file = output_dir / "window_features_rule.npy"
                np.save(features_file, features)
                logger.info(f"保存窗口特征矩阵: {features_file}")

        # 保存上行行为统计信息
        behavior_stats_up_file = output_dir / "behavior_statistics_up.json"
        with open(behavior_stats_up_file, "w") as f:
            json.dump(results["behavior_stats_up"], f, indent=2)
        logger.info(f"保存上行行为统计信息: {behavior_stats_up_file}")

        # 保存下行行为统计信息
        behavior_stats_down_file = output_dir / "behavior_statistics_down.json"
        with open(behavior_stats_down_file, "w") as f:
            json.dump(results["behavior_stats_down"], f, indent=2)
        logger.info(f"保存下行行为统计信息: {behavior_stats_down_file}")

        # 保存上行转移矩阵
        with open(output_dir / f"behavior_transition_graph_{method}_up.json", "w") as f:
            json.dump(
                {
                    "transition_matrix": results["transition_matrix_up"],
                    "num_behaviors": len(np.unique(labels_up)),
                    "method": method,
                },
                f,
                indent=2,
            )
            # 构建完整的文件路径
            transition_graph_up_file = (
                output_dir / f"behavior_transition_graph_{method}_up.json"
            )
            logger.info(f"保存上行转移矩阵: {transition_graph_up_file}")

        # 保存下行转移矩阵
        with open(
            output_dir / f"behavior_transition_graph_{method}_down.json", "w"
        ) as f:
            json.dump(
                {
                    "transition_matrix": results["transition_matrix_down"],
                    "num_behaviors": len(np.unique(labels_down)),
                    "method": method,
                },
                f,
                indent=2,
            )
            # 构建完整的文件路径
            transition_graph_down_file = (
                output_dir / f"behavior_transition_graph_{method}_down.json"
            )
            logger.info(f"保存下行转移矩阵: {transition_graph_down_file}")

        # 自动从raw_data_df中推导合法丢包值并保存到metadata目录
        if "raw_data" in results and results["raw_data"] is not None:
            raw_data_df = results["raw_data"]
            # 使用工具函数提取合法丢包值
            from network_simulation.utils.loss_utils import (
                extract_directional_valid_loss_values,
            )

            valid_loss_values_up, valid_loss_values_down, valid_loss_values = (
                extract_directional_valid_loss_values(raw_data_df)
            )

            if valid_loss_values is not None:
                metadata_dir = output_dir / "metadata"
                metadata_dir.mkdir(parents=True, exist_ok=True)

                # 保存上下行独立的合法丢包值
                if (
                    valid_loss_values_up is not None
                    and valid_loss_values_down is not None
                ):
                    # 保存上行合法丢包值
                    valid_loss_up_file = metadata_dir / "valid_loss_values_up.json"
                    with open(valid_loss_up_file, "w") as f:
                        json.dump(valid_loss_values_up, f, indent=2)
                    logger.info(f"保存上行合法丢包值: {valid_loss_up_file}")

                    # 保存下行合法丢包值
                    valid_loss_down_file = metadata_dir / "valid_loss_values_down.json"
                    with open(valid_loss_down_file, "w") as f:
                        json.dump(valid_loss_values_down, f, indent=2)
                    logger.info(f"保存下行合法丢包值: {valid_loss_down_file}")

                # 保存合并的合法丢包值（兼容旧版本）
                valid_loss_file = metadata_dir / "valid_loss_values.json"
                with open(valid_loss_file, "w") as f:
                    json.dump(valid_loss_values, f, indent=2)
                logger.info(f"保存合并合法丢包值到metadata目录: {valid_loss_file}")

        # 如果有原始数据，按行为类别保存原始数据片段
        if (
            "raw_data" in results
            and results["raw_data"] is not None
            and results["window_info"] is not None
        ):
            raw_data_df = results["raw_data"]
            window_info = results["window_info"]

            # 创建labeled_windows目录（方案要求的目录名）
            labeled_windows_dir = output_dir / "labeled_windows"
            labeled_windows_dir.mkdir(parents=True, exist_ok=True)

            # 创建上下行子目录
            labeled_windows_up_dir = labeled_windows_dir / "up"
            labeled_windows_up_dir.mkdir(parents=True, exist_ok=True)

            labeled_windows_down_dir = labeled_windows_dir / "down"
            labeled_windows_down_dir.mkdir(parents=True, exist_ok=True)

            # 直接使用window_info中的window_start和window_end，确保与检测阶段完全一致
            for i in range(len(labels_up)):
                # 获取当前窗口的起始和结束索引
                window_start = int(window_info.iloc[i]["window_start"])
                window_end = int(window_info.iloc[i]["window_end"])

                # 检查窗口有效性
                if window_start >= window_end or window_end > len(raw_data_df):
                    logger.warning(
                        f"保存时跳过无效窗口 {i}: window_start={window_start}, window_end={window_end}, raw_data_length={len(raw_data_df)}"
                    )
                    continue

                # 从raw_data_df中提取窗口数据
                window_data = raw_data_df.iloc[window_start:window_end].copy()

                # 检查是否包含上下行数据
                has_up_down_data = all(
                    col in window_data.columns
                    for col in ["delay1", "loss_rate1", "delay2", "loss_rate2"]
                )

                # 根据save_all_columns参数过滤列
                if not save_all_columns:
                    # 只保存指定前缀的列
                    allowed_cols = [
                        col
                        for col in window_data.columns
                        if any(
                            col.startswith(prefix)
                            for prefix in self.SAVE_ALLOWED_PREFIXES
                        )
                    ]

                    # 不符合条件的处理：至少保留timestamp列（如果存在）
                    if not allowed_cols:
                        # 至少保留timestamp列（如果存在）
                        if "timestamp" in window_data.columns:
                            allowed_cols = ["timestamp"]
                        else:
                            logger.debug(
                                f"窗口 {i} 没有找到匹配前缀的列且无timestamp列，跳过保存"
                            )
                            continue

                    window_data = window_data[allowed_cols]

                # 保存上行标签对应的窗口数据
                label_up = labels_up[i]
                # 跳过无效窗口
                if label_up != self.BEHAVIOR_LABELS["INVALID"]:
                    # 如果该标签的目录不存在，则创建
                    label_up_dir = labeled_windows_up_dir / str(label_up)
                    label_up_dir.mkdir(parents=True, exist_ok=True)

                    # 为上行数据选择对应列
                    if has_up_down_data:
                        # 上下行数据分离，只保存上行相关列
                        up_columns = [
                            col
                            for col in window_data.columns
                            if col.endswith("1") or not col.endswith(("1", "2"))
                        ]
                        window_data_up = window_data[up_columns]
                    else:
                        # 单通道数据，直接使用
                        window_data_up = window_data.copy()

                    # 将窗口数据保存到CSV文件
                    window_file_up = label_up_dir / f"window_{i}.csv"
                    window_data_up.to_csv(window_file_up, index=False)

                # 保存下行标签对应的窗口数据
                label_down = labels_down[i]
                # 跳过无效窗口
                if label_down != self.BEHAVIOR_LABELS["INVALID"]:
                    # 如果该标签的目录不存在，则创建
                    label_down_dir = labeled_windows_down_dir / str(label_down)
                    label_down_dir.mkdir(parents=True, exist_ok=True)

                    # 为下行数据选择对应列
                    if has_up_down_data:
                        # 上下行数据分离，只保存下行相关列
                        down_columns = [
                            col
                            for col in window_data.columns
                            if col.endswith("2") or not col.endswith(("1", "2"))
                        ]
                        window_data_down = window_data[down_columns]
                    else:
                        # 单通道数据，直接使用
                        window_data_down = window_data.copy()

                    # 将窗口数据保存到CSV文件
                    window_file_down = label_down_dir / f"window_{i}.csv"
                    window_data_down.to_csv(window_file_down, index=False)
