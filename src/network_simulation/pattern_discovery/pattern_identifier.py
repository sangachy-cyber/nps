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

from config import CONGESTION_DELAY_THRESHOLD, CONGESTION_LOSS_THRESHOLD
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PatternIdentifier:
    """使用规则事件检测引擎识别网络行为模式"""

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
        "INVALID": -1  # 无效窗口标签
    }

    # 魔法数字常量定义
    STRONG_BURST_DELAY_THRESHOLD: float = CONGESTION_DELAY_THRESHOLD  # Strong Burst的延迟阈值
    STRONG_BURST_LOSS_THRESHOLD: float = CONGESTION_LOSS_THRESHOLD  # Strong Burst的丢包率阈值
    STRONG_BURST_MIN_RUN: int = 15  # Strong Burst的最小连续点数量

    INSTANT_SPIKE_DELAY_THRESHOLD: float = 800.0  # 瞬时峰值的延迟阈值
    INSTANT_SPIKE_LOSS_THRESHOLD: float = 0.8  # 瞬时峰值的丢包率阈值
    INSTANT_SPIKE_MAX_COUNT: int = 5  # 瞬时峰值的最大峰值点数量
    INSTANT_SPIKE_MAX_RATIO: float = 0.1  # 瞬时峰值的最大峰值占比

    WEAK_BURST_LOSS_NONZERO_RATIO: float = 0.1  # Weak Burst的丢包率非零比例阈值
    WEAK_BURST_MIN_CONDITIONS: int = 2  # Weak Burst的最小满足条件数量

    DYNAMIC_THRESHOLD_WINDOW_SIZE: int = 10  # 动态阈值计算的窗口大小

    # 默认阈值常量定义
    DEFAULT_DELAY_MEAN_LOW: float = 100.0  # 默认低延迟阈值
    DEFAULT_DELAY_MEAN_HIGH: float = 300.0  # 默认高延迟阈值
    DEFAULT_DELAY_STD_HIGH: float = 200.0  # 默认高延迟标准差阈值
    DEFAULT_LOSS_MEAN_LOW: float = 0.02  # 默认低丢包率阈值
    DEFAULT_LOSS_MEAN_HIGH: float = 0.5  # 默认高丢包率阈值
    DEFAULT_LOSS_STD_HIGH: float = 0.3  # 默认高丢包率标准差阈值

    def __init__(
        self, method: str = "rule"
    ):
        self.method = method
        # 初始化为空列表，将在identify方法中动态获取
        self.feature_columns: List[str] = []

    def identify(
        self, features_df: pd.DataFrame, raw_data_df: pd.DataFrame = None
    ) -> Dict:
        """使用规则事件检测引擎识别网络行为模式

        Args:
            features_df: 包含特征数据的DataFrame
            raw_data_df: 原始数据DataFrame，可选

        Returns:
            Dict: 包含识别结果的字典，包括标签、转移矩阵和各种指标
        """
        # 检查输入DataFrame是否为空
        if features_df.empty:
            raise ValueError("输入特征DataFrame为空")

        # 检查必要列
        if raw_data_df is not None:
            required_cols = ["window_start", "window_end"]
            missing_cols = [col for col in required_cols if col not in features_df.columns]
            if missing_cols:
                raise ValueError(f"raw_data_df提供时，features_df必须包含以下列: {required_cols}。缺少: {missing_cols}")

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
            raise ValueError(
                "未找到特征列。期望列名以'feat_'开头")

        # 提取特征数据
        X = features_df[self.feature_columns].values

        # 计算动态阈值
        thresholds = self._calculate_dynamic_thresholds(features_df, raw_data_df)

        # 执行规则事件检测
        if self.method == "rule":
            labels_up, labels_down = self._perform_rule_based_detection(features_df, raw_data_df, thresholds)
        else:
            raise ValueError(f"未知的检测方法: {self.method}")

        # 计算上行转移矩阵和指标
        transition_matrix_up = self._calculate_transition_matrix(labels_up)
        transition_metrics_up = self._calculate_transition_metrics(transition_matrix_up)
        separation_metrics_up = self._calculate_separation_metrics(X, labels_up)
        behavior_stats_up = self._calculate_behavior_statistics(X, labels_up)

        # 计算下行转移矩阵和指标
        transition_matrix_down = self._calculate_transition_matrix(labels_down)
        transition_metrics_down = self._calculate_transition_metrics(transition_matrix_down)
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

        # Store raw data for later saving if provided
        if raw_data_df is not None:
            results["raw_data"] = raw_data_df
            # 存储必要的features_df列，用于save()方法保存labeled_windows
            results["window_info"] = features_df[["window_start", "window_end"]].copy()
        else:
            results["raw_data"] = None
            results["window_info"] = None

        return results

    def _calculate_window_stats(self, delays: np.ndarray, loss_rates: np.ndarray, window_idx: int = None) -> tuple:
        """计算窗口统计信息

        Args:
            delays: 延迟数据数组
            loss_rates: 丢包率数据数组
            window_idx: 窗口索引，可选，用于日志输出

        Returns:
            tuple: 包含延迟均值、延迟标准差、丢包率均值、丢包率标准差的元组；若窗口为空返回None
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
            delays: 延迟数据数组
            loss_rates: 丢包率数据数组

        Returns:
            bool: 是否为Strong Burst行为
        """
        # 检测条件：delay ≥ 400ms且loss ≥ 0.25，连续15个点以上
        congested = (delays >= self.STRONG_BURST_DELAY_THRESHOLD) & (loss_rates >= self.STRONG_BURST_LOSS_THRESHOLD)

        # 处理空数组情况
        if len(congested) == 0:
            return False

        # 使用NumPy向量化方法计算最长连续拥塞点数量
        # 在数组前后添加False，处理边界情况
        padded = np.concatenate(([False], congested, [False]))
        # 计算相邻元素的差异，1表示从False到True的转换，-1表示从True到False的转换
        diffs = np.diff(padded.astype(int))
        # 找到所有转换点的位置
        transition_points = np.where(diffs != 0)[0]

        # 处理边界情况
        if len(transition_points) == 0:
            # 数组全为False或全为True
            max_run = len(congested) if congested.all() else 0
        else:
            # 计算相邻转换点之间的距离，得到连续True的长度
            run_lengths = transition_points[1::2] - transition_points[::2]
            max_run = run_lengths.max()

        return max_run >= self.STRONG_BURST_MIN_RUN

    def _detect_weak_burst(self, loss_mean: float, loss_std: float, delay_std: float,
                          delays: np.ndarray, row: pd.Series, available_features: list, thresholds: Dict[str, float]) -> bool:
        """检测Weak Burst行为（弱突发）

        Args:
            loss_mean: 原始数据计算的丢包率均值
            loss_std: 原始数据计算的丢包率标准差
            delay_std: 原始数据计算的延迟标准差
            delays: 原始延迟数据数组
            row: 特征数据行
            available_features: 可用特征列表
            thresholds: 基于原始数据计算的阈值字典

        Returns:
            bool: 是否为Weak Burst行为
        """
        # Stable行为的条件：低延迟，低丢包，低波动
        if (loss_mean <= thresholds["loss_mean_low"] and
            delay_std < thresholds["delay_std_high"] / 2 and
            loss_std < thresholds["loss_std_high"] / 2):
            return False

        # 收集所有满足的条件，采用多数条件满足原则（至少2个条件）
        conditions = []

        # 条件1: 中等丢包率
        conditions.append(loss_mean >= thresholds["loss_mean_low"] and loss_mean < thresholds["loss_mean_high"])

        # 条件2: 延迟上升速率较快
        diffs = np.diff(delays) if len(delays) > 1 else np.array([])
        positive_ramps = diffs[diffs > 0]
        ramp_up = positive_ramps.max() if len(positive_ramps) > 0 else 0
        conditions.append(ramp_up > thresholds["delay_std_high"] / 2)

        # 条件3: 丢包率波动大
        conditions.append(loss_std > thresholds["loss_std_high"] / 2)

        # 条件4: 延迟波动大
        conditions.append(delay_std > thresholds["delay_std_high"] / 2)

        # 条件5: 使用现有特征检测（仅使用非标准化特征或确保阈值一致）
        if "feat_loss_nonzero_ratio" in available_features:
            conditions.append(row["feat_loss_nonzero_ratio"] > self.WEAK_BURST_LOSS_NONZERO_RATIO)

        # 多数条件满足原则：至少满足2个条件才判定为Weak Burst
        return sum(conditions) >= self.WEAK_BURST_MIN_CONDITIONS

    def _detect_instant_spike(self, delays: np.ndarray, loss_rates: np.ndarray) -> bool:
        """检测瞬时峰值行为

        Args:
            delays: 延迟数据数组
            loss_rates: 丢包率数据数组

        Returns:
            bool: 是否为瞬时峰值行为
        """
        # ≤5个点满足：loss ≥ 0.8或delay ≥ 800ms，且占比小于10%
        instant_spike_count = np.sum((delays >= self.INSTANT_SPIKE_DELAY_THRESHOLD) | (loss_rates >= self.INSTANT_SPIKE_LOSS_THRESHOLD))
        total_points = len(delays)
        spike_ratio = instant_spike_count / total_points if total_points > 0 else 0
        return 0 < instant_spike_count <= self.INSTANT_SPIKE_MAX_COUNT and spike_ratio < self.INSTANT_SPIKE_MAX_RATIO

    def _detect_behavior_with_raw_data(self, i: int, row: pd.Series, delays: np.ndarray,
                                      loss_rates: np.ndarray, available_features: list, thresholds: Dict[str, float]) -> int:
        """使用原始数据检测行为

        Args:
            i: 窗口索引
            row: 特征数据行
            delays: 延迟数据数组
            loss_rates: 丢包率数据数组
            available_features: 可用特征列表
            thresholds: 动态计算的阈值字典

        Returns:
            int: 行为标签
        """
        # 计算窗口统计信息
        stats = self._calculate_window_stats(delays, loss_rates, window_idx=i)

        # 如果窗口无效，标记为INVALID
        if stats is None:
            return self.BEHAVIOR_LABELS["INVALID"]

        delay_mean, delay_std, loss_mean, loss_std = stats

        # 按严重性优先级检测行为：
        # 1. 瞬时峰值 (最严重的突发)
        if self._detect_instant_spike(delays, loss_rates):
            return self.BEHAVIOR_LABELS["INSTANT_SPIKE"]

        # 2. Strong Burst (强突发)
        if self._detect_strong_burst(delays, loss_rates):
            return self.BEHAVIOR_LABELS["STRONG_BURST"]

        # 3. 持续高丢包 (持续严重丢包)
        if loss_mean > thresholds["loss_mean_high"] and loss_std < thresholds["loss_std_high"]:
            return self.BEHAVIOR_LABELS["HIGH_LOSS_STEADY"]

        # 4. 高延迟无丢包 (高延迟但无丢包)
        if delay_mean > thresholds["delay_mean_high"] and loss_mean <= thresholds["loss_mean_low"]:
            return self.BEHAVIOR_LABELS["HIGH_DELAY_NO_LOSS"]

        # 5. 低延迟高丢包 (低延迟但高丢包)
        if delay_mean < thresholds["delay_mean_low"] and loss_mean > thresholds["loss_mean_high"]:
            return self.BEHAVIOR_LABELS["LOW_DELAY_HIGH_LOSS"]

        # 6. Weak Burst (弱突发)
        if self._detect_weak_burst(loss_mean, loss_std, delay_std, delays, row, available_features, thresholds):
            return self.BEHAVIOR_LABELS["WEAK_BURST"]

        # 7. 频繁波动 (不稳定网络) - 放在Weak Burst之后，避免误判
        if delay_std > thresholds["delay_std_high"] and loss_std > thresholds["loss_std_high"]:
            return self.BEHAVIOR_LABELS["FREQUENT_FLUCTUATION"]

        # 8. 默认行为：稳定
        return self.BEHAVIOR_LABELS["STABLE"]

    def _calculate_dynamic_thresholds(self, features_df: pd.DataFrame, raw_data_df: pd.DataFrame = None) -> Dict[str, float]:
        """计算动态阈值

        Args:
            features_df: 包含特征数据的DataFrame
            raw_data_df: 原始数据DataFrame，可选

        Returns:
            Dict[str, float]: 计算得到的阈值字典
        """
        thresholds = {}

        if raw_data_df is not None:
            # 使用原始数据计算阈值
            self._calculate_thresholds_from_raw_data(raw_data_df, thresholds)
        else:
            # 使用特征列计算阈值
            self._calculate_thresholds_from_features(features_df, thresholds)

        # 设置默认阈值，确保所有必要的阈值都存在
        self._set_default_thresholds(thresholds)

        logger.info(f"计算动态阈值: {thresholds}")
        return thresholds

    def _calculate_thresholds_from_raw_data(self, raw_data_df: pd.DataFrame, thresholds: Dict) -> None:
        """从原始数据计算阈值

        Args:
            raw_data_df: 原始数据DataFrame
            thresholds: 阈值字典，用于存储计算结果
        """
        # 检查是否包含上下行数据
        has_up_down_data = all(col in raw_data_df.columns for col in ["delay1", "loss_rate1", "delay2", "loss_rate2"])
        logger.debug(f"检测到上下行数据: {has_up_down_data}")

        if has_up_down_data:
            # 合并上下行数据用于计算阈值
            delays = np.concatenate([raw_data_df["delay1"].values, raw_data_df["delay2"].values])
            loss_rates = np.concatenate([raw_data_df["loss_rate1"].values, raw_data_df["loss_rate2"].values])
        else:
            # 兼容旧格式，只处理delay和loss_rate
            delays = raw_data_df["delay"].values
            loss_rates = raw_data_df["loss_rate"].values

        # 计算分位数作为阈值
        thresholds["delay_mean_low"] = np.percentile(delays, 25)  # 低延迟阈值
        thresholds["delay_mean_high"] = np.percentile(delays, 75)  # 高延迟阈值
        thresholds["loss_mean_low"] = np.percentile(loss_rates, 25)  # 低丢包率阈值
        thresholds["loss_mean_high"] = np.percentile(loss_rates, 75)  # 高丢包率阈值

        # 安全计算延迟标准差阈值
        window_size = self.DYNAMIC_THRESHOLD_WINDOW_SIZE
        self._calculate_windowed_std_threshold(delays, window_size, "delay_std_high", thresholds)

        # 安全计算丢包率标准差阈值
        self._calculate_windowed_std_threshold(loss_rates, window_size, "loss_std_high", thresholds)

    def _calculate_windowed_std_threshold(self, data: np.ndarray, window_size: int, threshold_key: str, thresholds: Dict) -> None:
        """计算窗口标准差阈值

        Args:
            data: 输入数据数组
            window_size: 窗口大小
            threshold_key: 阈值字典中的键名
            thresholds: 阈值字典，用于存储计算结果
        """
        if len(data) >= window_size:
            # 使用滑动窗口计算标准差，更充分利用数据
            rolling_std = pd.Series(data).rolling(window=window_size, min_periods=1).std().dropna().values
            thresholds[threshold_key] = np.percentile(rolling_std, 75)
        else:
            # 数据不足一个窗口，直接计算整体标准差，处理空数组情况
            thresholds[threshold_key] = np.std(data) if len(data) > 1 else 0.0

    def _calculate_thresholds_from_features(self, features_df: pd.DataFrame, thresholds: Dict) -> None:
        """从特征列计算阈值

        Args:
            features_df: 包含特征数据的DataFrame
            thresholds: 阈值字典，用于存储计算结果
        """
        # 假设特征列已经标准化，使用分位数作为阈值
        for feat in self.feature_columns:
            if "delay" in feat:
                if "mean" in feat:
                    thresholds["delay_mean_low"] = np.percentile(features_df[feat], 25)
                    thresholds["delay_mean_high"] = np.percentile(features_df[feat], 75)
                elif "std" in feat:
                    thresholds["delay_std_high"] = np.percentile(features_df[feat], 75)
            elif "loss" in feat:
                if "mean" in feat:
                    thresholds["loss_mean_low"] = np.percentile(features_df[feat], 25)
                    thresholds["loss_mean_high"] = np.percentile(features_df[feat], 75)
                elif "std" in feat:
                    thresholds["loss_std_high"] = np.percentile(features_df[feat], 75)

    def _set_default_thresholds(self, thresholds: Dict) -> None:
        """设置默认阈值

        Args:
            thresholds: 阈值字典，用于存储默认值
        """
        thresholds.setdefault("delay_mean_low", self.DEFAULT_DELAY_MEAN_LOW)
        thresholds.setdefault("delay_mean_high", self.DEFAULT_DELAY_MEAN_HIGH)
        thresholds.setdefault("delay_std_high", self.DEFAULT_DELAY_STD_HIGH)
        thresholds.setdefault("loss_mean_low", self.DEFAULT_LOSS_MEAN_LOW)
        thresholds.setdefault("loss_mean_high", self.DEFAULT_LOSS_MEAN_HIGH)
        thresholds.setdefault("loss_std_high", self.DEFAULT_LOSS_STD_HIGH)

    def _detect_behavior_without_raw_data(self, row: pd.Series, available_features: list, thresholds: Dict[str, float]) -> int:
        """无原始数据时检测行为

        Args:
            row: 特征数据行
            available_features: 可用特征列表
            thresholds: 动态计算的阈值字典

        Returns:
            int: 行为标签
        """
        # 注意：特征值可能已标准化，部分行为判定受限
        # 检测行为 5: 持续高丢包
        if (
            "feat_loss_mean" in available_features
            and row["feat_loss_mean"] > thresholds["loss_mean_high"]
        ):
            return self.BEHAVIOR_LABELS["HIGH_LOSS_STEADY"]
        # 检测行为 6: 频繁波动
        elif (
            "feat_delay_std" in available_features
            and row["feat_delay_std"] > thresholds["delay_std_high"]
        ) and (
            "feat_loss_std" in available_features and row["feat_loss_std"] > thresholds["loss_std_high"]
        ):
            return self.BEHAVIOR_LABELS["FREQUENT_FLUCTUATION"]
        # 检测行为 7: 低延迟高丢包
        elif (
            "feat_delay_mean" in available_features
            and row["feat_delay_mean"] < thresholds["delay_mean_low"]
        ) and (
            "feat_loss_mean" in available_features
            and row["feat_loss_mean"] > thresholds["loss_mean_high"]
        ):
            return self.BEHAVIOR_LABELS["LOW_DELAY_HIGH_LOSS"]
        # 检测行为 2: Strong Burst
        elif (
            "feat_max_congestion_run" in available_features
            and row["feat_max_congestion_run"] >= 15
        ):
            return self.BEHAVIOR_LABELS["STRONG_BURST"]
        # 检测行为 1: Weak Burst
        elif (
            "feat_loss_nonzero_ratio" in available_features
            and row["feat_loss_nonzero_ratio"] > self.WEAK_BURST_LOSS_NONZERO_RATIO
        ):
            return self.BEHAVIOR_LABELS["WEAK_BURST"]
        # 默认为Stable
        # 注意：在无原始数据时，行为3（瞬时峰值）和行为4（高延迟无丢包）未被检测
        # 这些行为需要原始数据的绝对尺度，特征值可能已标准化
        else:
            return self.BEHAVIOR_LABELS["STABLE"]

    def _perform_rule_based_detection(
        self, features_df: pd.DataFrame, raw_data_df: pd.DataFrame = None, thresholds: Dict[str, float] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """执行规则事件检测，识别网络行为模式

        Args:
            features_df: 包含特征数据的DataFrame，当提供raw_data_df时，必须包含window_start和window_end列
            raw_data_df: 原始数据DataFrame，可选，其索引必须是连续的行号[0, 1, 2, ...]
            thresholds: 动态计算的阈值字典

        Returns:
            tuple[np.ndarray, np.ndarray]: 上行行为标签数组和下行行为标签数组
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
            labels_up, labels_down = self._detect_with_raw_data(features_df, raw_data_df, available_features, thresholds)
        else:
            # 无原始数据时，使用向量化操作进行批量处理，提高性能
            logger.warning("未提供 raw_data_df，部分行为（如 INSTANT_SPIKE, HIGH_DELAY_NO_LOSS, LOW_DELAY_HIGH_LOSS）无法检测")
            labels_up, labels_down = self._detect_without_raw_data(features_df, available_features, thresholds)

        logger.info("规则事件检测完成")

        # 使用np.unique替代np.bincount，处理非连续标签
        # 上行标签统计
        unique_labels_up, counts_up = np.unique(labels_up, return_counts=True)
        label_stats_up = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels_up, counts_up)}
        logger.info(f"上行行为标签统计: {label_stats_up}")

        # 下行标签统计
        unique_labels_down, counts_down = np.unique(labels_down, return_counts=True)
        label_stats_down = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels_down, counts_down)}
        logger.info(f"下行行为标签统计: {label_stats_down}")

        return labels_up, labels_down

    def _detect_with_raw_data(self, features_df: pd.DataFrame, raw_data_df: pd.DataFrame,
                              available_features: list, thresholds: Dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """使用原始数据进行行为检测

        Args:
            features_df: 包含特征数据的DataFrame
            raw_data_df: 原始数据DataFrame
            available_features: 可用特征列表
            thresholds: 动态计算的阈值字典

        Returns:
            tuple[np.ndarray, np.ndarray]: 上行行为标签数组和下行行为标签数组
        """
        # 初始化上下行行为标签数组，默认为STABLE
        labels_up = np.full(len(features_df), self.BEHAVIOR_LABELS["STABLE"], dtype=int)
        labels_down = np.full(len(features_df), self.BEHAVIOR_LABELS["STABLE"], dtype=int)

        # 增强列检查，确保包含所有必要的上下行数据列
        required_cols = {"delay1", "loss_rate1", "delay2", "loss_rate2"}
        missing_cols = required_cols - set(raw_data_df.columns)
        if missing_cols:
            raise ValueError(f"raw_data_df 缺少必要的上下行列: {missing_cols}")

        has_up_down_data = True
        logger.debug("检测到完整的上下行数据")

        # 有原始数据时，需要逐窗口处理原始数据
        for i, (_, row) in enumerate(features_df.iterrows()):
            # 获取当前窗口的原始数据，用于计算更详细的统计信息
            window_start = int(row["window_start"])
            window_end = int(row["window_end"])

            # 检查窗口有效性
            if window_start >= window_end or window_end > len(raw_data_df):
                logger.warning(f"窗口 {i} 无效: window_start={window_start}, window_end={window_end}, raw_data_length={len(raw_data_df)}")
                labels_up[i] = self.BEHAVIOR_LABELS["INVALID"]
                labels_down[i] = self.BEHAVIOR_LABELS["INVALID"]
                continue  # 直接跳过无效窗口，避免冗余计算

            # 提取窗口数据
            window_data = raw_data_df.iloc[window_start:window_end]

            if has_up_down_data:
                # 上下行数据
                # 上行数据
                delay1 = window_data["delay1"].values
                loss_rate1 = window_data["loss_rate1"].values

                # 下行数据
                delay2 = window_data["delay2"].values
                loss_rate2 = window_data["loss_rate2"].values

                # 检查窗口数据是否为空
                if len(delay1) == 0 or len(loss_rate1) == 0 or len(delay2) == 0 or len(loss_rate2) == 0:
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
            else:
                # 兼容旧格式，只处理delay和loss_rate
                delays = window_data["delay"].values
                loss_rates = window_data["loss_rate"].values

                # 检查窗口数据是否为空
                if len(delays) == 0 or len(loss_rates) == 0:
                    logger.warning(f"窗口 {i} 数据为空，标记为无效窗口")
                    labels_up[i] = self.BEHAVIOR_LABELS["INVALID"]
                    labels_down[i] = self.BEHAVIOR_LABELS["INVALID"]
                    continue  # 直接跳过空数据窗口

                # 使用原始数据检测行为
                behavior = self._detect_behavior_with_raw_data(
                    i, row, delays, loss_rates, available_features, thresholds
                )

                # 旧格式数据同时存储到上下行标签数组中
                labels_up[i] = behavior
                labels_down[i] = behavior

        return labels_up, labels_down

    def _detect_without_raw_data(self, features_df: pd.DataFrame,
                                 available_features: list, thresholds: Dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """不使用原始数据进行行为检测（向量化操作）

        该函数使用向量化操作，基于提取的特征直接进行行为检测，无需访问原始数据。
        支持上下行数据的单独检测和综合分析。

        Args:
            features_df: 包含特征数据的DataFrame
            available_features: 可用特征列表
            thresholds: 动态计算的阈值字典，用于行为检测

        Returns:
            tuple[np.ndarray, np.ndarray]: 上行行为标签数组和下行行为标签数组
        """
        logger.info("无原始数据，使用向量化操作进行批量行为检测")

        # 检查是否包含上下行特征
        has_up_down_features = any("delay1" in col or "loss1" in col or "delay2" in col or "loss2" in col for col in available_features)
        logger.debug(f"检测到上下行特征: {has_up_down_features}")

        if has_up_down_features:
            # 上下行特征行为检测
            # 初始化上下行行为标签
            labels_up = np.full(len(features_df), self.BEHAVIOR_LABELS["STABLE"], dtype=int)
            labels_down = np.full(len(features_df), self.BEHAVIOR_LABELS["STABLE"], dtype=int)

            # 上行行为检测
            # 检测行为 5: 持续高丢包
            if "feat_loss1_mean" in available_features:
                high_loss_mask1 = features_df["feat_loss1_mean"] > thresholds["loss_mean_high"]
                labels_up[high_loss_mask1] = self.BEHAVIOR_LABELS["HIGH_LOSS_STEADY"]

            # 检测行为 6: 频繁波动
            if "feat_delay1_std" in available_features and "feat_loss1_std" in available_features:
                frequent_fluctuation_mask1 = (features_df["feat_delay1_std"] > thresholds["delay_std_high"]) & (features_df["feat_loss1_std"] > thresholds["loss_std_high"])
                labels_up[frequent_fluctuation_mask1] = self.BEHAVIOR_LABELS["FREQUENT_FLUCTUATION"]

            # 检测行为 7: 低延迟高丢包
            if "feat_delay1_mean" in available_features and "feat_loss1_mean" in available_features:
                low_delay_high_loss_mask1 = (features_df["feat_delay1_mean"] < thresholds["delay_mean_low"]) & (features_df["feat_loss1_mean"] > thresholds["loss_mean_high"])
                labels_up[low_delay_high_loss_mask1] = self.BEHAVIOR_LABELS["LOW_DELAY_HIGH_LOSS"]

            # 检测行为 2: Strong Burst
            if "feat_max_congestion_run1" in available_features:
                strong_burst_mask1 = features_df["feat_max_congestion_run1"] >= self.STRONG_BURST_MIN_RUN
                labels_up[strong_burst_mask1] = self.BEHAVIOR_LABELS["STRONG_BURST"]

            # 检测行为 1: Weak Burst
            weak_burst_mask1 = self._detect_weak_burst_without_raw_data(features_df, available_features, thresholds, direction="up")
            labels_up[weak_burst_mask1] = self.BEHAVIOR_LABELS["WEAK_BURST"]

            # 下行行为检测
            # 检测行为 5: 持续高丢包
            if "feat_loss2_mean" in available_features:
                high_loss_mask2 = features_df["feat_loss2_mean"] > thresholds["loss_mean_high"]
                labels_down[high_loss_mask2] = self.BEHAVIOR_LABELS["HIGH_LOSS_STEADY"]

            # 检测行为 6: 频繁波动
            if "feat_delay2_std" in available_features and "feat_loss2_std" in available_features:
                frequent_fluctuation_mask2 = (features_df["feat_delay2_std"] > thresholds["delay_std_high"]) & (features_df["feat_loss2_std"] > thresholds["loss_std_high"])
                labels_down[frequent_fluctuation_mask2] = self.BEHAVIOR_LABELS["FREQUENT_FLUCTUATION"]

            # 检测行为 7: 低延迟高丢包
            if "feat_delay2_mean" in available_features and "feat_loss2_mean" in available_features:
                low_delay_high_loss_mask2 = (features_df["feat_delay2_mean"] < thresholds["delay_mean_low"]) & (features_df["feat_loss2_mean"] > thresholds["loss_mean_high"])
                labels_down[low_delay_high_loss_mask2] = self.BEHAVIOR_LABELS["LOW_DELAY_HIGH_LOSS"]

            # 检测行为 2: Strong Burst
            if "feat_max_congestion_run2" in available_features:
                strong_burst_mask2 = features_df["feat_max_congestion_run2"] >= self.STRONG_BURST_MIN_RUN
                labels_down[strong_burst_mask2] = self.BEHAVIOR_LABELS["STRONG_BURST"]

            # 检测行为 1: Weak Burst
            weak_burst_mask2 = self._detect_weak_burst_without_raw_data(features_df, available_features, thresholds, direction="down")
            labels_down[weak_burst_mask2] = self.BEHAVIOR_LABELS["WEAK_BURST"]

            return labels_up, labels_down
        else:
            # 兼容旧格式，只处理单通道特征
            # 初始化单通道行为标签
            labels_single = np.full(len(features_df), self.BEHAVIOR_LABELS["STABLE"], dtype=int)

            # 检测行为 5: 持续高丢包
            if "feat_loss_mean" in available_features:
                high_loss_mask = features_df["feat_loss_mean"] > thresholds["loss_mean_high"]
                labels_single[high_loss_mask] = self.BEHAVIOR_LABELS["HIGH_LOSS_STEADY"]

            # 检测行为 6: 频繁波动
            if "feat_delay_std" in available_features and "feat_loss_std" in available_features:
                frequent_fluctuation_mask = (features_df["feat_delay_std"] > thresholds["delay_std_high"]) & (features_df["feat_loss_std"] > thresholds["loss_std_high"])
                labels_single[frequent_fluctuation_mask] = self.BEHAVIOR_LABELS["FREQUENT_FLUCTUATION"]

            # 检测行为 7: 低延迟高丢包
            if "feat_delay_mean" in available_features and "feat_loss_mean" in available_features:
                low_delay_high_loss_mask = (features_df["feat_delay_mean"] < thresholds["delay_mean_low"]) & (features_df["feat_loss_mean"] > thresholds["loss_mean_high"])
                labels_single[low_delay_high_loss_mask] = self.BEHAVIOR_LABELS["LOW_DELAY_HIGH_LOSS"]

            # 检测行为 2: Strong Burst
            if "feat_max_congestion_run" in available_features:
                strong_burst_mask = features_df["feat_max_congestion_run"] >= self.STRONG_BURST_MIN_RUN
                labels_single[strong_burst_mask] = self.BEHAVIOR_LABELS["STRONG_BURST"]

            # 检测行为 1: Weak Burst
            weak_burst_mask = self._detect_weak_burst_without_raw_data(features_df, available_features, thresholds)
            labels_single[weak_burst_mask] = self.BEHAVIOR_LABELS["WEAK_BURST"]

            # 单通道特征时，上下行标签相同
            return labels_single, labels_single

    def _detect_weak_burst_without_raw_data(self, features_df: pd.DataFrame,
                                            available_features: list, thresholds: Dict[str, float], direction: str = None) -> np.ndarray:
        """无原始数据时检测Weak Burst行为

        Args:
            features_df: 包含特征数据的DataFrame
            available_features: 可用特征列表
            thresholds: 动态计算的阈值字典
            direction: 方向，"up"表示上行，"down"表示下行，None表示单通道

        Returns:
            np.ndarray: Weak Burst行为的掩码数组
        """
        # 先检测Stable行为
        stable_mask = np.ones(len(features_df), dtype=bool)

        # 确定使用的特征后缀
        suffix = "" if direction is None else "1" if direction == "up" else "2"

        # Stable行为条件：低延迟，低丢包，低波动
        loss_mean_col = f"feat_loss{suffix}_mean"
        delay_std_col = f"feat_delay{suffix}_std"
        loss_std_col = f"feat_loss{suffix}_std"

        if loss_mean_col in available_features:
            stable_mask &= (features_df[loss_mean_col] <= thresholds["loss_mean_low"])
        if delay_std_col in available_features:
            stable_mask &= (features_df[delay_std_col] < thresholds["delay_std_high"] / 2)
        if loss_std_col in available_features:
            stable_mask &= (features_df[loss_std_col] < thresholds["loss_std_high"] / 2)

        # 收集所有满足的条件，采用多数条件满足原则（至少2个条件）
        weak_burst_conditions = []

        # 条件1: 中等丢包率
        if loss_mean_col in available_features:
            condition1 = (features_df[loss_mean_col] >= thresholds["loss_mean_low"]) & (features_df[loss_mean_col] < thresholds["loss_mean_high"])
            weak_burst_conditions.append(condition1)

        # 条件2: 延迟波动大
        if delay_std_col in available_features:
            condition2 = features_df[delay_std_col] > thresholds["delay_std_high"] / 2
            weak_burst_conditions.append(condition2)

        # 条件3: 丢包率波动大
        if loss_std_col in available_features:
            condition3 = features_df[loss_std_col] > thresholds["loss_std_high"] / 2
            weak_burst_conditions.append(condition3)

        # 条件4: 丢包率非零比例高
        loss_nonzero_col = f"feat_loss{suffix}_nonzero_ratio"
        if loss_nonzero_col in available_features:
            condition4 = features_df[loss_nonzero_col] > self.WEAK_BURST_LOSS_NONZERO_RATIO
            weak_burst_conditions.append(condition4)

        # 应用多数条件满足原则，并且排除Stable窗口
        if weak_burst_conditions:
            return (np.sum(weak_burst_conditions, axis=0) >= self.WEAK_BURST_MIN_CONDITIONS) & (~stable_mask)
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
        valid_behavior_ids = [v for k, v in self.BEHAVIOR_LABELS.items() if k != "INVALID"]
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
                if current_state in valid_behavior_ids and next_state in valid_behavior_ids:
                    # 将标签映射到索引
                    current_idx = label_to_index[current_state]
                    next_idx = label_to_index[next_state]
                    transition_matrix[current_idx, next_idx] += 1

        # 归一化行以获取概率
        row_sums = transition_matrix.sum(axis=1, keepdims=True)

        # 处理行和为0的情况：将该行设置为均匀分布
        for i in range(len(row_sums)):
            if row_sums[i, 0] == 0:
                transition_matrix[i, :] = 1.0 / num_clusters
            else:
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
            transition_graph_up_file = output_dir / f"behavior_transition_graph_{method}_up.json"
            logger.info(f"保存上行转移矩阵: {transition_graph_up_file}")

        # 保存下行转移矩阵
        with open(output_dir / f"behavior_transition_graph_{method}_down.json", "w") as f:
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
            transition_graph_down_file = output_dir / f"behavior_transition_graph_{method}_down.json"
            logger.info(f"保存下行转移矩阵: {transition_graph_down_file}")

        # 自动从raw_data_df中推导合法丢包值并保存到metadata目录
        if "raw_data" in results and results["raw_data"] is not None:
            raw_data_df = results["raw_data"]
            # 检查是否包含上下行丢包率数据
            has_up_down_loss = all(col in raw_data_df.columns for col in ["loss_rate1", "loss_rate2"])
            valid_loss_values_up = None
            valid_loss_values_down = None
            valid_loss_values = None

            if has_up_down_loss:
                # 分离上下行丢包率数据
                loss_rates_up = raw_data_df["loss_rate1"].values
                loss_rates_down = raw_data_df["loss_rate2"].values

                # 推导上下行独立的合法丢包值
                valid_loss_values_up = sorted(list(set(loss_rates_up)))
                valid_loss_values_down = sorted(list(set(loss_rates_down)))

                # 同时保存合并的合法丢包值（兼容旧版本）
                loss_rates = np.concatenate([loss_rates_up, loss_rates_down])
                valid_loss_values = sorted(list(set(loss_rates)))
            elif "loss_rate" in raw_data_df.columns:
                # 兼容旧格式，只处理loss_rate
                if valid_loss_values is None:
                    valid_loss_values = sorted(list(set(raw_data_df["loss_rate"].unique())))
                # 上下行使用相同的合法丢包值
                valid_loss_values_up = valid_loss_values
                valid_loss_values_down = valid_loss_values
            else:
                logger.warning("原始数据中未找到丢包率列，无法自动推导合法丢包值")
                valid_loss_values = None
                valid_loss_values_up = None
                valid_loss_values_down = None

            if valid_loss_values is not None:
                metadata_dir = output_dir / "metadata"
                metadata_dir.mkdir(parents=True, exist_ok=True)

                # 保存上下行独立的合法丢包值
                if valid_loss_values_up is not None and valid_loss_values_down is not None:
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
        if "raw_data" in results and results["raw_data"] is not None and results["window_info"] is not None:
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
                    logger.warning(f"保存时跳过无效窗口 {i}: window_start={window_start}, window_end={window_end}, raw_data_length={len(raw_data_df)}")
                    continue

                # 从raw_data_df中提取窗口数据
                window_data = raw_data_df.iloc[window_start:window_end].copy()

                # 保存上行标签对应的窗口数据
                label_up = labels_up[i]
                # 跳过无效窗口
                if label_up != self.BEHAVIOR_LABELS["INVALID"]:
                    # 如果该标签的目录不存在，则创建
                    label_up_dir = labeled_windows_up_dir / str(label_up)
                    label_up_dir.mkdir(parents=True, exist_ok=True)

                    # 将窗口数据保存到CSV文件
                    window_file_up = label_up_dir / f"window_{i}.csv"
                    window_data.to_csv(window_file_up, index=False)

                # 保存下行标签对应的窗口数据
                label_down = labels_down[i]
                # 跳过无效窗口
                if label_down != self.BEHAVIOR_LABELS["INVALID"]:
                    # 如果该标签的目录不存在，则创建
                    label_down_dir = labeled_windows_down_dir / str(label_down)
                    label_down_dir.mkdir(parents=True, exist_ok=True)

                    # 将窗口数据保存到CSV文件
                    window_file_down = label_down_dir / f"window_{i}.csv"
                    window_data.to_csv(window_file_down, index=False)
