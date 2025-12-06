#!/usr/bin/env python3
"""
网络行为模式发现模块
负责使用聚类算法发现网络行为模式
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from hdbscan import HDBSCAN
from typing import Dict, List, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)


# Top-level DTW distance functions to support pickling


def _dtw_distance_fastdtw(a, b):
    """DTW distance using fastdtw"""
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    distance, _ = fastdtw(a.reshape(-1, 1), b.reshape(-1, 1), dist=euclidean)
    return distance


def _dtw_distance_tslearn(a, b):
    """DTW distance using tslearn"""
    from tslearn.metrics import dtw

    return dtw(a.reshape(-1, 1), b.reshape(-1, 1))


class PatternIdentifier:
    """使用聚类算法识别网络行为模式"""

    def __init__(self, method: str = "gmm", config: dict = None, use_cache: bool = True):
        self.method = method
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = [
            "feat_delay_std",
            "feat_loss_nonzero_ratio",
            "feat_loss_high_ratio",
            "feat_max_consec_loss",
            "feat_loss_unique_values",
            "feat_delay_trend",
            "feat_delay_acf_5",
            "feat_loss_mode_encoded",
            "feat_loss_pattern_std",
        ]
        # 加载配置
        self.config = config or {}
        # 导入默认配置
        try:
            from ...config import (
                DEFAULT_MIN_CLUSTER_SIZE,
                DEFAULT_MIN_SAMPLES,
                DEFAULT_CLUSTER_SELECTION_EPSILON
            )
            # 设置默认值
            self.config.setdefault('min_cluster_size', DEFAULT_MIN_CLUSTER_SIZE)
            self.config.setdefault('min_samples', DEFAULT_MIN_SAMPLES)
            self.config.setdefault('cluster_selection_epsilon', DEFAULT_CLUSTER_SELECTION_EPSILON)
        except ImportError:
            pass
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
        """Discover network behavior patterns and optionally save raw data segments"""
        import hashlib
        import pickle

        # Generate cache key based on features and parameters
        def generate_cache_key():
            # Create a hash of the features and parameters
            hash_obj = hashlib.md5()
            # Hash features data
            hash_obj.update(features_df.to_csv().encode('utf-8'))
            # Hash method and config
            hash_obj.update(self.method.encode('utf-8'))
            hash_obj.update(str(self.config).encode('utf-8'))
            return hash_obj.hexdigest()

        cache_key = generate_cache_key()
        cache_file = self.cache_dir / f"clustering_{cache_key}.pkl"

        # Check if cache exists and use it if enabled
        if self.use_cache and cache_file.exists():
            logger.info(f"使用缓存的聚类结果: {cache_file}")
            with open(cache_file, 'rb') as f:
                results = pickle.load(f)
            # Restore scaler
            self.scaler = results["scaler"]
            self.model = results.get("model")
            # Store raw data for later saving if provided
            if raw_data_df is not None:
                results["raw_data"] = raw_data_df
            return results

        # Extract feature columns
        X = features_df[self.feature_columns].values

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Perform clustering based on selected method
        if self.method == "gmm":
            labels, model = self._perform_gmm(X_scaled)
        elif self.method == "kmeans":
            labels, model = self._perform_kmeans(X_scaled)
        elif self.method == "hdbscan":
            labels, model = self._perform_hdbscan(X_scaled)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        self.model = model

        # Calculate clustering metrics
        metrics = self._calculate_clustering_metrics(X_scaled, labels)

        # Calculate transition matrix
        transition_matrix = self._calculate_transition_matrix(labels)

        # Calculate transition metrics
        transition_metrics = self._calculate_transition_metrics(transition_matrix)
        metrics.update(transition_metrics)

        # Calculate separation metrics
        separation_metrics = self._calculate_separation_metrics(X, labels)

        # Get cluster centers and statistics
        cluster_stats = self._calculate_cluster_statistics(X, labels)

        # Prepare results
        results = {
            "method": self.method,
            "labels": labels.tolist(),
            "metrics": metrics,
            "transition_matrix": transition_matrix.tolist(),
            "cluster_stats": cluster_stats,
            "separation_metrics": separation_metrics,
            "scaler": self.scaler,
            "model": model,
        }

        # Store raw data for later saving if provided
        if raw_data_df is not None:
            results["raw_data"] = raw_data_df
        else:
            results["raw_data"] = None

        # Save cache if enabled
        if self.use_cache:
            # Remove raw data from cache to save space
            cache_results = results.copy()
            cache_results.pop("raw_data", None)
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_results, f)
            logger.info(f"保存聚类结果到缓存: {cache_file}")

        return results

    def _perform_gmm(self, X: np.ndarray) -> Tuple[np.ndarray, GaussianMixture]:
        """Perform Gaussian Mixture Model clustering"""
        # Try different numbers of components to find the best one
        best_bic = float("inf")
        best_model = None

        for n_components in range(2, 15):
            model = GaussianMixture(n_components=n_components, random_state=42)
            model.fit(X)
            bic = model.bic(X)

            if bic < best_bic:
                best_bic = bic
                best_model = model

        labels = best_model.predict(X)
        return labels, best_model

    def _perform_kmeans(self, X: np.ndarray) -> Tuple[np.ndarray, KMeans]:
        """Perform K-means clustering"""
        # Try different numbers of clusters to find the best one
        best_silhouette = -1
        best_model = None

        for n_clusters in range(2, 15):
            model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(X)
            silhouette = silhouette_score(X, labels)

            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_model = model

        labels = best_model.predict(X)
        return labels, best_model

    def _perform_hdbscan(self, X: np.ndarray) -> Tuple[np.ndarray, HDBSCAN]:
        """Perform HDBSCAN clustering with DTW distance using tslearn"""
        import warnings

        logger.info("Using tslearn DTW for HDBSCAN clustering")

        # Filter specific FutureWarning about force_all_finite being renamed to ensure_all_finite
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message="'force_all_finite' was renamed to 'ensure_all_finite'",
            )
            # 优化HDBSCAN参数，提高性能和聚类质量
            model = HDBSCAN(
                min_cluster_size=self.config.get('min_cluster_size', 3),
                min_samples=self.config.get('min_samples', 1),
                metric=_dtw_distance_tslearn,
                cluster_selection_method="eom",
                cluster_selection_epsilon=self.config.get('cluster_selection_epsilon', 0.2),
                n_jobs=-1,  # 使用所有可用CPU核心
                gen_min_span_tree=False,  # 禁用最小生成树生成，提高性能
                algorithm='best',  # 选择最佳算法（'best'自动选择最快的实现）
                leaf_size=40,  # 调整叶子大小，提高性能
            )
            labels = model.fit_predict(X)

        return labels, model

    def _calculate_clustering_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate clustering quality metrics"""
        metrics = {}

        # Silhouette Score
        if len(np.unique(labels)) > 1:
            metrics["silhouette_score"] = float(silhouette_score(X, labels))
        else:
            metrics["silhouette_score"] = 0.0

        # Calinski-Harabasz Index
        metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X, labels))

        # Number of clusters
        metrics["num_clusters"] = int(len(np.unique(labels)))

        # Cluster sizes
        metrics["cluster_sizes"] = {}
        for label in np.unique(labels):
            metrics["cluster_sizes"][str(label)] = int(np.sum(labels == label))

        # BIC and AIC for GMM
        if hasattr(self.model, "bic"):
            metrics["bic"] = float(self.model.bic(X))
            metrics["aic"] = float(self.model.aic(X))
            metrics["log_likelihood"] = float(self.model.score(X))

        return metrics

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
        num_clusters = len(np.unique(labels))
        transition_matrix = np.zeros((num_clusters, num_clusters))

        for i in range(len(labels) - 1):
            current_state = labels[i]
            next_state = labels[i + 1]
            transition_matrix[current_state, next_state] += 1

        # Normalize rows to get probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums

        return transition_matrix

    def _calculate_cluster_statistics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate statistics for each cluster"""
        cluster_stats = {}
        unique_labels = np.unique(labels)

        for label in unique_labels:
            cluster_data = X[labels == label]
            cluster_stats[str(label)] = {
                "mean": cluster_data.mean(axis=0).tolist(),
                "std": cluster_data.std(axis=0).tolist(),
                "min": cluster_data.min(axis=0).tolist(),
                "max": cluster_data.max(axis=0).tolist(),
                "count": int(len(cluster_data)),
            }

        return cluster_stats

    def save(
        self,
        results: Dict,
        output_dir: Path,
        features_df: pd.DataFrame = None,
        valid_loss_values: List[float] = None,
    ) -> None:
        """Save clustering results to files with method-specific filenames and raw data segments"""
        output_dir.mkdir(parents=True, exist_ok=True)

        method = results["method"]

        # Save labels and metadata to JSON with method suffix
        with open(output_dir / f"behavior_labels_{method}.json", "w") as f:
            json.dump(
                {
                    "method": method,
                    "labels": results["labels"],
                    "metrics": results["metrics"],
                    "cluster_stats": results["cluster_stats"],
                },
                f,
                indent=2,
            )

        # Save transition matrix with method suffix
        with open(output_dir / f"behavior_transition_graph_{method}.json", "w") as f:
            json.dump(
                {
                    "transition_matrix": results["transition_matrix"],
                    "num_clusters": results["metrics"]["num_clusters"],
                    "method": method,
                },
                f,
                indent=2,
            )

        # Save model and scaler with method suffix for model
        with open(output_dir / f"behavior_{method}_model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        # Scaler is common for all methods, so no suffix needed
        with open(output_dir / "behavior_scaler.pkl", "wb") as f:
            pickle.dump(results["scaler"], f)

        # Save valid loss values to metadata directory
        if valid_loss_values is not None:
            metadata_dir = output_dir / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            with open(metadata_dir / "valid_loss_values.json", "w") as f:
                json.dump(valid_loss_values, f, indent=2)

        # Save raw data segments by behavior category if raw data is available
        if "raw_data" in results and features_df is not None:
            raw_data_df = results["raw_data"]
            labels = results["labels"]

            # Create clustered_data directory
            clustered_data_dir = output_dir / "clustered_data"
            clustered_data_dir.mkdir(parents=True, exist_ok=True)

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
                label_dir = clustered_data_dir / str(label)
                label_dir.mkdir(parents=True, exist_ok=True)

                # Save window data to CSV
                window_file = label_dir / f"window_{i}.csv"
                window_data.to_csv(window_file, index=False)
