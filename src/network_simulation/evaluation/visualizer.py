#!/usr/bin/env python3
"""
Visualization Module
Responsible for visualizing network behavior patterns and evaluation results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List
from pathlib import Path


class Visualizer:
    """Visualizes network behavior patterns and evaluation results"""

    def __init__(self):
        self.feature_columns = [
            "feat_delay_std",
            "feat_loss_burst_ratio",
            "feat_burst_duration",
            "feat_burst_intensity",
            "feat_delay_trend",
            "feat_delay_acf_5",
        ]

        # Set plotting style
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("husl")
        # 设置中文支持，兼容Windows、MacOS和Linux，Linux系统优先使用文泉驿正黑
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    def visualize_evaluation_results(self, results: Dict, output_dir: Path) -> None:
        """Visualize evaluation results"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Visualize clustering metrics
        self._plot_clustering_metrics(results, output_dir)

        # Visualize transition metrics
        if "transition_matrix" in results:
            self._plot_transition_metrics(results, output_dir)

    def visualize_clustering_results(
        self, X: np.ndarray, labels: np.ndarray, output_dir: Path
    ) -> None:
        """Visualize clustering results"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # PCA visualization
        self._plot_pca_scatter(X, labels, output_dir)
        self._plot_3d_pca_scatter(X, labels, output_dir)
        self._plot_pca_variance(X, output_dir)

    def visualize_feature_analysis(
        self, features_df: pd.DataFrame, labels: np.ndarray, output_dir: Path
    ) -> None:
        """Visualize feature analysis"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Feature distribution histograms
        self._plot_feature_distributions(features_df, labels, output_dir)

        # Feature correlation heatmap
        self._plot_feature_correlation(features_df, output_dir)

        # 3D feature scatter plot
        self._plot_3d_feature_scatter(features_df, labels, output_dir)

    def visualize_behavior_transition(
        self, transition_matrix: np.ndarray, output_dir: Path
    ) -> None:
        """Visualize behavior transition"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Interactive state transition graph
        self._plot_interactive_transition_graph(transition_matrix, output_dir)

        # Transition matrix heatmap
        self._plot_transition_matrix_heatmap(transition_matrix, output_dir)

    def visualize_behavior_samples(
        self,
        raw_df: pd.DataFrame,
        window_labels: List,
        window_size: int,
        output_dir: Path,
    ) -> None:
        """Visualize behavior samples"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot sample time series
        self._plot_sample_time_series(raw_df, window_labels, window_size, output_dir)

        # Plot typical samples
        self._plot_typical_samples(raw_df, window_labels, window_size, output_dir)

    def _plot_clustering_metrics(self, results: Dict, output_dir: Path) -> None:
        """Plot clustering metrics"""
        metrics = results.get("metrics", {})

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot silhouette score and Calinski-Harabasz index
        if "silhouette_score" in metrics and "calinski_harabasz_score" in metrics:
            ax.bar(
                ["Silhouette Score", "Calinski-Harabasz"],
                [metrics["silhouette_score"], metrics["calinski_harabasz_score"]],
                color=["skyblue", "salmon"],
            )
            ax.set_ylabel("Score")
            ax.set_title("Clustering Quality Metrics")
            ax.axhline(
                y=0.5, color="gray", linestyle="--", alpha=0.7, label="Silhouette > 0.5"
            )
            ax.legend()

            plt.tight_layout()
            plt.savefig(
                output_dir / "clustering_metrics.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

    def _plot_transition_metrics(self, results: Dict, output_dir: Path) -> None:
        """Plot transition metrics"""
        transition_matrix = np.array(results.get("transition_matrix", []))

        if len(transition_matrix) == 0:
            return

        # Calculate transition metrics
        transition_entropy = self._calculate_transition_entropy(transition_matrix)
        transition_sparsity = self._calculate_transition_sparsity(transition_matrix)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(
            ["Transition Entropy", "Transition Sparsity"],
            [transition_entropy, transition_sparsity],
            color=["lightgreen", "orange"],
        )
        ax.set_ylabel("Value")
        ax.set_title("Behavior Transition Metrics")

        plt.tight_layout()
        plt.savefig(output_dir / "transition_metrics.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_pca_scatter(
        self, X: np.ndarray, labels: np.ndarray, output_dir: Path
    ) -> None:
        """Plot PCA scatter plot"""
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(10, 8))

        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                label=f"Behavior {label}",
                alpha=0.7,
                s=50,
            )

        ax.set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        ax.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        ax.set_title("PCA Scatter Plot of Network Behaviors")
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "pca_scatter.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_pca_variance(self, X: np.ndarray, output_dir: Path) -> None:
        """Plot PCA variance explained"""
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform PCA
        pca = PCA(n_components=6)
        pca.fit(X_scaled)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot explained variance ratio
        ax.bar(
            range(1, 7),
            pca.explained_variance_ratio_,
            alpha=0.7,
            align="center",
            label="Individual explained variance",
        )
        ax.step(
            range(1, 7),
            np.cumsum(pca.explained_variance_ratio_),
            where="mid",
            label="Cumulative explained variance",
        )

        ax.set_xlabel("Principal Components")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("PCA Variance Explained")
        ax.legend()
        ax.set_xticks(range(1, 7))

        plt.tight_layout()
        plt.savefig(output_dir / "pca_variance.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_3d_pca_scatter(
        self, X: np.ndarray, labels: np.ndarray, output_dir: Path
    ) -> None:
        """Plot 3D PCA scatter plot"""
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform PCA
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)

        # Create 3D scatter plot using plotly
        fig = go.Figure()

        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            fig.add_trace(
                go.Scatter3d(
                    x=X_pca[mask, 0],
                    y=X_pca[mask, 1],
                    z=X_pca[mask, 2],
                    mode="markers",
                    name=f"Behavior {label}",
                    marker=dict(size=5, opacity=0.7),
                )
            )

        fig.update_layout(
            title="3D PCA Scatter Plot of Network Behaviors",
            scene=dict(
                xaxis_title=f"PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)",
                yaxis_title=f"PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)",
                zaxis_title=f"PCA 3 ({pca.explained_variance_ratio_[2]:.2%} variance)",
            ),
            legend_title="Behavior Categories",
            width=1000,
            height=800,
        )

        # Save as HTML for interactive viewing
        fig.write_html(output_dir / "pca_scatter_3d.html")

        # Try to save as static image, but skip if there's an error
        try:
            # Also save as static image
            fig.write_image(
                output_dir / "pca_scatter_3d.png", width=1000, height=800, scale=2
            )
        except Exception:
            # Skip saving static image if there's an error (e.g., incompatible Kaleido version)
            pass

    def _plot_feature_distributions(
        self, features_df: pd.DataFrame, labels: np.ndarray, output_dir: Path
    ) -> None:
        """Plot feature distributions by behavior category"""
        n_features = len(self.feature_columns)
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features))

        for i, feature in enumerate(self.feature_columns):
            ax = axes[i]
            for label in np.unique(labels):
                mask = labels == label
                sns.histplot(
                    features_df[feature][mask],
                    ax=ax,
                    kde=True,
                    alpha=0.5,
                    label=f"Behavior {label}",
                )
            ax.set_title(f"Distribution of {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")
            ax.legend()

        plt.tight_layout()
        plt.savefig(
            output_dir / "feature_distributions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_feature_correlation(
        self, features_df: pd.DataFrame, output_dir: Path
    ) -> None:
        """Plot feature correlation heatmap"""
        # Calculate correlation matrix using only numerical columns
        numerical_features = features_df.select_dtypes(include=[np.number])
        corr = numerical_features.corr(method="pearson")

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )

        ax.set_title("Feature Correlation Heatmap (Pearson)")

        plt.tight_layout()
        plt.savefig(
            output_dir / "feature_correlation.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_3d_feature_scatter(
        self, features_df: pd.DataFrame, labels: np.ndarray, output_dir: Path
    ) -> None:
        """Plot 3D feature scatter plot"""
        # Select numerical features
        numerical_features = features_df.select_dtypes(include=[np.number])

        # Use first 3 features for 3D visualization
        feature_names = numerical_features.columns.tolist()[:3]
        X = numerical_features[feature_names].values

        # Create 3D scatter plot using plotly
        fig = go.Figure()

        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            fig.add_trace(
                go.Scatter3d(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    z=X[mask, 2],
                    mode="markers",
                    name=f"Behavior {label}",
                    marker=dict(size=5, opacity=0.7),
                )
            )

        fig.update_layout(
            title=f"3D Feature Scatter Plot ({feature_names[0]} vs {feature_names[1]} vs {feature_names[2]})",
            scene=dict(
                xaxis_title=feature_names[0],
                yaxis_title=feature_names[1],
                zaxis_title=feature_names[2],
            ),
            legend_title="Behavior Categories",
            width=1000,
            height=800,
        )

        # Save as HTML for interactive viewing
        fig.write_html(output_dir / "feature_scatter_3d.html")

        # Try to save as static image, but skip if there's an error
        try:
            # Also save as static image
            fig.write_image(
                output_dir / "feature_scatter_3d.png", width=1000, height=800, scale=2
            )
        except Exception:
            # Skip saving static image if there's an error (e.g., incompatible Kaleido version)
            pass

    def _plot_interactive_transition_graph(
        self, transition_matrix: np.ndarray, output_dir: Path
    ) -> None:
        """Plot interactive state transition graph"""
        G = nx.DiGraph()

        # Add nodes
        for i in range(len(transition_matrix)):
            G.add_node(i, label=f"Behavior {i}")

        # Add edges with weights
        for i in range(len(transition_matrix)):
            for j in range(len(transition_matrix)):
                weight = transition_matrix[i, j]
                if weight > 0.01:  # Only add edges with significant weight
                    G.add_edge(i, j, weight=weight)

        # Create plotly figure
        pos = nx.spring_layout(G, seed=42)

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[f"Behavior {node}" for node in G.nodes()],
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                reversescale=True,
                color=[],
                size=15,
                colorbar=dict(
                    thickness=15,
                    title="Node Connections",
                    xanchor="left",
                    titleside="right",
                ),
                line_width=2,
            ),
        )

        # Color nodes by number of connections
        node_adjacencies = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
        node_trace.marker.color = node_adjacencies

        # Add edge weights as annotations
        edge_weights = []
        for edge in G.edges(data=True):
            edge_weights.append(edge[2]["weight"])

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Interactive State Transition Graph",
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="<br>Edge width reflects transition intensity",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        # Save as HTML
        fig.write_html(output_dir / "interactive_transition_graph.html")

    def _plot_transition_matrix_heatmap(
        self, transition_matrix: np.ndarray, output_dir: Path
    ) -> None:
        """Plot transition matrix heatmap"""
        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(
            transition_matrix,
            annot=True,
            cmap="YlGnBu",
            fmt=".2f",
            xticklabels=[f"Behavior {i}" for i in range(len(transition_matrix))],
            yticklabels=[f"Behavior {i}" for i in range(len(transition_matrix))],
            ax=ax,
        )

        ax.set_title("State Transition Matrix Heatmap")
        ax.set_xlabel("Next State")
        ax.set_ylabel("Current State")

        plt.tight_layout()
        plt.savefig(
            output_dir / "transition_matrix_heatmap.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_sample_time_series(
        self,
        raw_df: pd.DataFrame,
        window_labels: List,
        window_size: int,
        output_dir: Path,
    ) -> None:
        """Plot sample time series for each behavior category"""
        # Get unique labels
        unique_labels = np.unique(window_labels)

        # Randomly select 1-3 samples per label
        selected_windows = {}
        for label in unique_labels:
            label_windows = [i for i, current_label in enumerate(window_labels) if current_label == label]
            n_samples = min(3, len(label_windows))
            selected_windows[label] = np.random.choice(
                label_windows, n_samples, replace=False
            )

        # Plot selected samples
        for label, windows in selected_windows.items():
            for window_idx in windows:
                start_idx = window_idx * window_size
                end_idx = start_idx + window_size

                # Get the window data
                window_df = raw_df.iloc[start_idx:end_idx].copy()

                # Create time index
                window_df["time"] = (
                    np.arange(len(window_df)) * 0.1
                )  # Assuming 100ms intervals

                fig, ax1 = plt.subplots(figsize=(12, 6))

                # Plot delay on primary y-axis
                ax1.set_xlabel("Time (seconds)")
                ax1.set_ylabel("Delay (ms)", color="tab:blue")
                ax1.plot(
                    window_df["time"], window_df["delay"], color="tab:blue", alpha=0.7
                )
                ax1.tick_params(axis="y", labelcolor="tab:blue")

                # Plot loss rate on secondary y-axis
                ax2 = ax1.twinx()
                ax2.set_ylabel("Loss Rate", color="tab:red")
                ax2.plot(
                    window_df["time"],
                    window_df["loss_rate"],
                    color="tab:red",
                    alpha=0.7,
                )
                ax2.tick_params(axis="y", labelcolor="tab:red")
                # Set y-axis limits to 0-1 for loss rate
                ax2.set_ylim(0, 1)

                fig.tight_layout()
                plt.title(f"Behavior {label} - Sample Window {window_idx}")
                plt.savefig(
                    output_dir / f"behavior_{label}_sample_{window_idx}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

    def _plot_typical_samples(
        self,
        raw_df: pd.DataFrame,
        window_labels: List,
        window_size: int,
        output_dir: Path,
    ) -> None:
        """Plot typical samples for each behavior category"""
        # Get unique labels
        unique_labels = np.unique(window_labels)

        # Create a figure with subplots for each behavior
        fig, axes = plt.subplots(
            len(unique_labels), 1, figsize=(15, 4 * len(unique_labels))
        )

        if len(unique_labels) == 1:
            axes = [axes]

        for i, label in enumerate(unique_labels):
            # Find the first occurrence of this label
            window_idx = window_labels.index(label)
            start_idx = window_idx * window_size
            end_idx = start_idx + window_size

            # Get the window data
            window_df = raw_df.iloc[start_idx:end_idx].copy()

            # Create time index (10 seconds window)
            window_df["time"] = np.arange(len(window_df)) * 0.1

            ax1 = axes[i]

            # Plot delay on primary y-axis
            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel("Delay (ms)", color="tab:blue")
            ax1.plot(window_df["time"], window_df["delay"], color="tab:blue", alpha=0.7)
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax1.set_title(f"Typical Sample - Behavior {label}")

            # Plot loss rate on secondary y-axis
            ax2 = ax1.twinx()
            ax2.set_ylabel("Loss Rate", color="tab:red")
            ax2.plot(
                window_df["time"], window_df["loss_rate"], color="tab:red", alpha=0.7
            )
            ax2.tick_params(axis="y", labelcolor="tab:red")
            # Set y-axis limits to 0-1 for loss rate
            ax2.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(output_dir / "typical_samples.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _calculate_transition_entropy(self, transition_matrix: np.ndarray) -> float:
        """Calculate average transition entropy"""
        entropy = 0.0
        num_states = len(transition_matrix)

        for i in range(num_states):
            state_entropy = 0.0
            for j in range(num_states):
                p = transition_matrix[i, j]
                if p > 0:
                    state_entropy -= p * np.log2(p)
            entropy += state_entropy

        return entropy / num_states

    def _calculate_transition_sparsity(self, transition_matrix: np.ndarray) -> float:
        """Calculate transition sparsity"""
        total_elements = transition_matrix.size
        non_zero_elements = np.count_nonzero(transition_matrix)
        return non_zero_elements / total_elements
