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
    """使用规则事件检测引擎识别网络行为模式"""

    def __init__(self, method: str = "rule", config: dict = None, use_cache: bool = True):
        self.method = method
        self.model = None
        # 初始化为空列表，将在identify方法中动态获取
        self.feature_columns = []
        # 加载配置
        self.config = config or {}
        # 缓存配置
        self.use_cache = use_cache
        self.cache_dir = Path(".cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_features(self, file_path: Path) -> pd.DataFrame:
        """Load extracted features"""
        return pd.read_csv(file_path)

    def identify(
        self, features_df: pd.DataFrame, raw_data_df: pd.DataFrame = None
    ) -> Dict:
        """Discover network behavior patterns using rule-based event detection"""


        # Dynamically get feature columns from the dataframe
        self.feature_columns = [col for col in features_df.columns if col.startswith('feat_')]
        logger.info(f"动态获取特征列: {self.feature_columns}")
        
        # Check if we have any feature columns
        if not self.feature_columns:
            raise ValueError("No feature columns found in the dataframe. Expected columns starting with 'feat_'")
            
        # Extract feature columns
        X = features_df[self.feature_columns].values

        # Perform rule-based event detection
        if self.method == "rule":
            labels = self._perform_rule_based_detection(features_df, raw_data_df)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")

        # Calculate transition matrix
        transition_matrix = self._calculate_transition_matrix(labels)

        # Calculate transition metrics
        transition_metrics = self._calculate_transition_metrics(transition_matrix)

        # Calculate separation metrics
        separation_metrics = self._calculate_separation_metrics(X, labels)

        # Get behavior statistics
        behavior_stats = self._calculate_behavior_statistics(X, labels)

        # Prepare results
        results = {
            "method": self.method,
            "labels": labels.tolist(),
            "metrics": transition_metrics,
            "transition_matrix": transition_matrix.tolist(),
            "behavior_stats": behavior_stats,
            "separation_metrics": separation_metrics,
            "model": self.model,
        }

        # Store raw data for later saving if provided
        if raw_data_df is not None:
            results["raw_data"] = raw_data_df
        else:
            results["raw_data"] = None



        return results

    def _perform_rule_based_detection(self, features_df: pd.DataFrame, raw_data_df: pd.DataFrame = None) -> np.ndarray:
        """Perform rule-based event detection to identify network behavior patterns"""
        logger.info("开始规则事件检测")
        
        # 初始化标签数组，默认为0（Stable）
        labels = np.zeros(len(features_df), dtype=int)
        
        # 获取可用的特征列
        available_features = features_df.columns.tolist()
        
        # 逐个窗口进行行为判定
        for i, (_, row) in enumerate(features_df.iterrows()):
            # 获取当前窗口的原始数据，用于计算更详细的统计信息
            window_start = int(row['window_start'])
            window_end = int(row['window_end'])
            
            if raw_data_df is not None:
                # 确保window_end不超过raw_data_df的长度
                window_end = min(window_end, len(raw_data_df))
                window_data = raw_data_df.iloc[window_start:window_end]
                delays = window_data['delay'].values
                loss_rates = window_data['loss_rate'].values
                
                # 计算窗口内的统计信息
                delay_mean = np.mean(delays)
                delay_std = np.std(delays)
                delay_min = np.min(delays)
                delay_max = np.max(delays)
                loss_mean = np.mean(loss_rates)
                loss_std = np.std(loss_rates)
                loss_max = np.max(loss_rates)
                
                # 检测行为 4: 高延迟无丢包 (放宽条件)
                # delay_mean > 500ms且loss_mean < 0.02
                if delay_mean > 500 and loss_mean < 0.02:
                    labels[i] = 4
                    continue
                
                # 检测行为 5: 持续高丢包
                # loss_mean > 0.5且loss_std < 0.2
                if loss_mean > 0.5 and loss_std < 0.2:
                    labels[i] = 5
                    continue
                
                # 检测行为 6: 频繁波动
                # delay_std > 300且loss_std > 0.3
                if delay_std > 300 and loss_std > 0.3:
                    labels[i] = 6
                    continue
                
                # 检测行为 7: 低延迟高丢包
                # delay_mean < 100ms且loss_mean > 0.3
                if delay_mean < 100 and loss_mean > 0.3:
                    labels[i] = 7
                    continue
                
                # 检测行为 3: 瞬时峰值 (调整阈值)
                # ≤5个点满足：loss ≥ 0.8或delay ≥ 800ms
                instant_spike_count = np.sum((delays >= 800) | (loss_rates >= 0.8))
                if instant_spike_count > 0 and instant_spike_count <= 5:
                    labels[i] = 3
                    continue
                
                # 检测行为 2: Strong Burst（持续拥塞，使用原始数据计算）
                # 计算连续拥塞点数量：delay ≥ 400ms且loss ≥ 0.25
                congestion_run = 0
                max_congestion_run = 0
                for delay, loss in zip(delays, loss_rates):
                    if delay >= 400 and loss >= 0.25:
                        congestion_run += 1
                        max_congestion_run = max(max_congestion_run, congestion_run)
                    else:
                        congestion_run = 0
                
                if max_congestion_run >= 15:
                    labels[i] = 2
                    continue
                
                # 检测行为 1: Weak Burst (调整条件，增加更多判定依据)
                # 条件1: loss_mean ∈ [0.05, 0.5) 或
                # 条件2: ramp_up > 80ms 或
                # 条件3: loss_std > 0.2 或
                # 条件4: delay_std > 200
                weak_burst = False
                
                # 条件1: 中等丢包率
                if (loss_mean >= 0.05 and loss_mean < 0.5):
                    weak_burst = True
                
                # 条件2: 延迟上升速率较快
                if len(delays) > 1:
                    delay_diff = delays[1:] - delays[:-1]
                    ramp_up = np.max(delay_diff) if len(delay_diff) > 0 else 0
                    if ramp_up > 80:
                        weak_burst = True
                
                # 条件3: 丢包率波动大
                if loss_std > 0.2:
                    weak_burst = True
                
                # 条件4: 延迟波动大
                if delay_std > 200:
                    weak_burst = True
                
                # 条件5: 使用现有特征检测
                if 'feat_loss_nonzero_ratio' in available_features:
                    if row['feat_loss_nonzero_ratio'] > 0.1:
                        weak_burst = True
                
                if 'feat_delay_std' in available_features:
                    if row['feat_delay_std'] > 1.0:
                        weak_burst = True
                
                if weak_burst:
                    labels[i] = 1
                    continue
                
                # 默认行为 0: Stable
                labels[i] = 0
            else:
                # 如果没有原始数据，只使用可用的特征进行判定
                # 检测行为 5: 持续高丢包
                if 'feat_loss_mean' in available_features and row['feat_loss_mean'] > 0.5:
                    labels[i] = 5
                # 检测行为 6: 频繁波动
                elif ('feat_delay_std' in available_features and row['feat_delay_std'] > 3.0) and \
                     ('feat_loss_std' in available_features and row['feat_loss_std'] > 0.3):
                    labels[i] = 6
                # 检测行为 7: 低延迟高丢包
                elif ('feat_delay_mean' in available_features and row['feat_delay_mean'] < 0.5) and \
                     ('feat_loss_mean' in available_features and row['feat_loss_mean'] > 0.3):
                    labels[i] = 7
                # 检测行为 4: 高延迟无丢包
                elif 'feat_delay_mean' in available_features and row['feat_delay_mean'] > 2.0 and \
                     ('feat_loss_mean' in available_features and row['feat_loss_mean'] < 0.05):
                    labels[i] = 4
                # 检测行为 2: Strong Burst
                elif 'feat_max_congestion_run' in available_features and row['feat_max_congestion_run'] >= 15:
                    labels[i] = 2
                # 检测行为 3: Instant Spike
                elif 'feat_delay_std' in available_features and row['feat_delay_std'] > 2.0:
                    labels[i] = 3
                # 检测行为 1: Weak Burst
                elif 'feat_loss_nonzero_ratio' in available_features and row['feat_loss_nonzero_ratio'] > 0.3:
                    labels[i] = 1
                # 默认为Stable
                else:
                    labels[i] = 0
        
        logger.info("规则事件检测完成")
        logger.info(f"行为标签统计: {np.bincount(labels)}")
        
        return labels

    def _calculate_transition_metrics(self, transition_matrix: np.ndarray) -> Dict:
        """Calculate behavior transition quality metrics"""
        metrics = {}

        # Average transition entropy
        entropy = 0.0
        for row in transition_matrix:
            # Remove zero probabilities to avoid log(0)
            row = row[row > 0]
            if len(row) > 0:
                entropy -= np.sum(row * np.log2(row))
        metrics["average_transition_entropy"] = float(entropy / len(transition_matrix))

        # Transition sparsity
        total_elements = transition_matrix.size
        non_zero_elements = np.count_nonzero(transition_matrix)
        metrics["transition_sparsity"] = float(non_zero_elements / total_elements)

        # Typical path analysis - Find most likely transition paths
        # For each state, find the top 3 most likely next states
        typical_paths = {}
        num_states = transition_matrix.shape[0]
        for i in range(num_states):
            # Get top 3 most likely next states
            top_indices = np.argsort(transition_matrix[i, :])[::-1][:3]
            top_probs = transition_matrix[i, top_indices]
            typical_paths[str(i)] = {
                "next_states": [int(idx) for idx in top_indices],
                "probabilities": [float(prob) for prob in top_probs]
            }
        metrics["typical_paths"] = typical_paths

        return metrics

    def _calculate_separation_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate behavior separation metrics"""
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
        """Calculate state transition matrix"""
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels)
        
        # Create a mapping from label value to matrix index
        label_to_index = {label: i for i, label in enumerate(sorted(unique_labels))}
        
        transition_matrix = np.zeros((num_clusters, num_clusters))

        for i in range(len(labels) - 1):
            current_state = labels[i]
            next_state = labels[i + 1]
            # Map labels to indices
            current_idx = label_to_index[current_state]
            next_idx = label_to_index[next_state]
            transition_matrix[current_idx, next_idx] += 1

        # Normalize rows to get probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums

        return transition_matrix

    def _calculate_behavior_statistics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate statistics for each behavior"""
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
        features_df: pd.DataFrame = None,
        valid_loss_values: List[float] = None,
    ) -> None:
        """Save rule-based detection results to files with method-specific filenames and raw data segments"""
        output_dir.mkdir(parents=True, exist_ok=True)

        method = results["method"]

        # Save labels as npy file (方案要求的文件名)
        if method == "rule":
            labels_file = output_dir / "_labels_rule.npy"
            np.save(labels_file, np.array(results["labels"]))
            logger.info(f"保存标签为npy文件: {labels_file}")

        # Save behavior statistics as separate JSON file (方案要求的文件名)
        behavior_stats_file = output_dir / "behavior_statistics.json"
        with open(behavior_stats_file, "w") as f:
            json.dump(results["behavior_stats"], f, indent=2)
        logger.info(f"保存行为统计信息: {behavior_stats_file}")

        # Save labels and metadata to JSON with method suffix
        with open(output_dir / f"behavior_labels_{method}.json", "w") as f:
            json.dump(
                {
                    "method": method,
                    "labels": results["labels"],
                    "metrics": results["metrics"],
                    "behavior_stats": results["behavior_stats"],
                },
                f,
                indent=2,
            )

        # Save transition matrix with method suffix
        with open(output_dir / f"behavior_transition_graph_{method}.json", "w") as f:
            json.dump(
                {
                    "transition_matrix": results["transition_matrix"],
                    "num_behaviors": len(np.unique(results["labels"])),
                    "method": method,
                },
                f,
                indent=2,
            )

        # Save valid loss values to metadata directory (方案要求的目录位置)
        if valid_loss_values is not None:
            metadata_dir = output_dir / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            valid_loss_file = metadata_dir / "valid_loss_values.json"
            with open(valid_loss_file, "w") as f:
                json.dump(valid_loss_values, f, indent=2)
            logger.info(f"保存合法丢包值到metadata目录: {valid_loss_file}")

        # Save raw data segments by behavior category if raw data is available
        if "raw_data" in results and features_df is not None:
            raw_data_df = results["raw_data"]
            labels = results["labels"]

            # Create labeled_windows directory (方案要求的目录名)
            labeled_windows_dir = output_dir / "labeled_windows"
            labeled_windows_dir.mkdir(parents=True, exist_ok=True)

            # Use default window parameters from FeatureExtractor
            slide_samples = 50  # Based on slide_step=5.0s, time_granularity=0.1s
            window_samples = 100  # Based on window_size=10.0s, time_granularity=0.1s

            # Calculate window start and end indices based on index
            for i, label in enumerate(labels):
                # Calculate window start and end indices
                window_start = i * slide_samples
                window_end = window_start + window_samples

                # Ensure window_end doesn't exceed raw_data_df length
                window_end = min(window_end, len(raw_data_df))

                # Extract window data from raw_data_df
                window_data = raw_data_df.iloc[window_start:window_end].copy()

                # Create directory for this label if it doesn't exist
                label_dir = labeled_windows_dir / str(label)
                label_dir.mkdir(parents=True, exist_ok=True)

                # Save window data to CSV
                window_file = label_dir / f"window_{i}.csv"
                window_data.to_csv(window_file, index=False)
