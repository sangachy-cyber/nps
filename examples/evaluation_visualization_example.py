#!/usr/bin/env python3
"""
Example script demonstrating the evaluation and visualization functionality
"""

import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from pathlib import Path
from src.network_simulation.pattern_discovery.pattern_identifier import (
    PatternIdentifier,
)
from src.network_simulation.evaluation.evaluator import Evaluator
from src.network_simulation.evaluation.visualizer import Visualizer


def main():
    # Define paths
    data_dir = Path("./data")
    features_path = data_dir / "results" / "features" / "features.csv"
    output_dir = data_dir / "results" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data for sample visualization
    raw_data_path = data_dir / "raw" / "20251130_233044_xke-playback-1.txt"

    print("Loading data...")

    # Load features
    features_df = pd.read_csv(features_path)

    # Load raw data - skip header lines and use correct column names
    # Skip first 13 lines, use column names from line 11
    raw_df = pd.read_csv(
        raw_data_path,
        skiprows=13,
        header=None,
        names=["delay1", "loss1", "bandwidth1", "delay2", "loss2", "bandwidth2"],
    )

    # Create a simple time index (assuming 0.2s intervals as mentioned in header)
    raw_df["time"] = np.arange(len(raw_df)) * 0.2

    # For visualization, we'll use delay1 and loss1 as our main metrics
    raw_df = raw_df[["time", "delay1", "loss1"]].rename(
        columns={"delay1": "delay", "loss1": "loss_rate"}
    )

    print(f"Loaded {len(features_df)} feature windows")
    print(f"Loaded {len(raw_df)} raw data points")

    # Step 1: Perform clustering
    print("\nPerforming clustering...")
    pattern_identifier = PatternIdentifier(method="gmm")
    clustering_results = pattern_identifier.identify(features_df)

    labels = np.array(clustering_results["labels"])
    X = features_df[pattern_identifier.feature_columns].values
    transition_matrix = np.array(clustering_results["transition_matrix"])

    print(
        f"Clustering completed with {clustering_results['metrics']['num_clusters']} clusters"
    )
    print(f"Silhouette Score: {clustering_results['metrics']['silhouette_score']:.4f}")
    print(
        f"Calinski-Harabasz Index: {clustering_results['metrics']['calinski_harabasz_score']:.4f}"
    )

    # Step 2: Evaluate clustering quality
    print("\nEvaluating clustering quality...")
    evaluator = Evaluator()

    # Clustering quality evaluation
    clustering_quality = evaluator.evaluate_clustering_quality(
        X, labels, pattern_identifier.model
    )
    print(f"Clustering Quality: {clustering_quality}")

    # Behavior transition quality evaluation
    transition_quality = evaluator.evaluate_transition_quality(transition_matrix)
    print(f"Transition Quality: {transition_quality}")

    # Behavior separation evaluation
    behavior_separation = evaluator.evaluate_behavior_separation(X, labels)
    print(
        f"Behavior Separation: {len(behavior_separation['behavior_stats'])} behaviors analyzed"
    )

    # Step 3: Visualization
    print("\nGenerating visualizations...")
    visualizer = Visualizer()

    # 1. Evaluation results visualization
    visualizer.visualize_evaluation_results(clustering_results, output_dir)

    # 2. Clustering results visualization
    visualizer.visualize_clustering_results(X, labels, output_dir)

    # 3. Feature analysis visualization
    visualizer.visualize_feature_analysis(features_df, labels, output_dir)

    # 4. Behavior transition visualization
    visualizer.visualize_behavior_transition(transition_matrix, output_dir)

    # 5. Behavior samples visualization (using window size of 100 points = 10 seconds)
    window_size = 100
    visualizer.visualize_behavior_samples(
        raw_df, labels.tolist(), window_size, output_dir
    )

    # Step 4: Save comprehensive evaluation results
    print("\nSaving evaluation results...")

    comprehensive_results = {
        "clustering_quality": clustering_quality,
        "transition_quality": transition_quality,
        "behavior_separation": behavior_separation,
        "clustering_metrics": clustering_results["metrics"],
        "transition_matrix": transition_matrix.tolist(),
    }

    with open(output_dir / "comprehensive_evaluation_results.json", "w") as f:
        import json

        json.dump(comprehensive_results, f, indent=2)

    # Generate comprehensive report using Evaluator.save method
    evaluator.save(comprehensive_results, output_dir)

    print("\nAll evaluation and visualization completed!")
    print(f"Results saved to: {output_dir}")

    # Print summary of key findings
    print("\nKey Findings:")
    print(
        f"- Number of behaviors discovered: {clustering_results['metrics']['num_clusters']}"
    )
    print(
        f"- Clustering quality (Silhouette Score): {'Good (>0.5)' if clustering_quality['silhouette_score'] > 0.5 else 'Needs improvement (<=0.5)'}"
    )
    print(
        f"- Transition complexity: {'Low' if transition_quality['transition_sparsity'] < 0.3 else 'Medium' if transition_quality['transition_sparsity'] < 0.6 else 'High'}"
    )
    print(
        f"- Behavior separation: {'Good' if behavior_separation['separation_metrics']['separation_index'] > 1.0 else 'Moderate' if behavior_separation['separation_metrics']['separation_index'] > 0.5 else 'Poor'}"
    )


if __name__ == "__main__":
    main()
