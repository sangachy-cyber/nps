#!/usr/bin/env python3
"""
Smart Scheduling Module
Responsible for scheduling network behavior patterns
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


class SmartScheduler:
    """Smart scheduler for network behavior patterns"""

    def __init__(self):
        pass

    def load_schedule(self, schedule_path: Path) -> Dict:
        """Load behavior schedule from file"""
        with open(schedule_path, "r") as f:
            schedule = json.load(f)
        return schedule

    def load_patterns(self, patterns_dir: Path) -> Dict:
        """Load discovered patterns from directory"""
        # Load behavior labels
        with open(patterns_dir / "behavior_labels.json", "r") as f:
            behavior_labels = json.load(f)

        # Load transition graph
        with open(patterns_dir / "behavior_transition_graph.json", "r") as f:
            transition_graph = json.load(f)

        patterns = {
            "behavior_labels": behavior_labels,
            "transition_graph": transition_graph,
            "cluster_stats": behavior_labels["cluster_stats"],
        }

        return patterns

    def validate_schedule(self, schedule: Dict, patterns: Dict) -> Tuple[bool, str]:
        """Validate if the schedule is reasonable based on transition probabilities"""
        transition_matrix = np.array(patterns["transition_graph"]["transition_matrix"])
        segments = schedule["segments"]

        # Check if segments are in order and don't overlap
        for i in range(len(segments) - 1):
            if segments[i]["end_time"] > segments[i + 1]["start_time"]:
                return False, f"Segments {i} and {i + 1} overlap"

        # Check if transition probabilities are reasonable
        for i in range(len(segments) - 1):
            current_behavior = segments[i]["behavior_type"]
            next_behavior = segments[i + 1]["behavior_type"]

            # Map behavior types to cluster labels
            behavior_cluster_map = patterns.get(
                "behavior_cluster_map",
                {"stable": 0, "burst_loss": 1, "high_jitter": 2, "recovery": 3},
            )

            current_cluster = behavior_cluster_map.get(current_behavior, 0)
            next_cluster = behavior_cluster_map.get(next_behavior, 0)

            # Check transition probability
            transition_prob = transition_matrix[current_cluster][next_cluster]
            if transition_prob < 0.01:  # Threshold for reasonable transition
                return (
                    False,
                    f"Transition from {current_behavior} to {next_behavior} is unlikely (probability: {transition_prob:.4f})",
                )

        return True, "Schedule is valid"

    def optimize_schedule(self, schedule: Dict, patterns: Dict) -> Dict:
        """Optimize the schedule based on transition probabilities"""
        # For now, just return the original schedule
        # Later, implement more sophisticated optimization
        return schedule

    def insert_transition_segments(self, schedule: Dict, patterns: Dict) -> Dict:
        """Insert transition segments between behavior segments"""
        optimized_segments = []
        segments = schedule["segments"]

        for i in range(len(segments)):
            current_segment = segments[i]
            optimized_segments.append(current_segment)

            # Insert transition segment if not last segment
            if i < len(segments) - 1:
                next_segment = segments[i + 1]
                transition_segment = self._create_transition_segment(
                    current_segment, next_segment, patterns
                )
                optimized_segments.append(transition_segment)

        optimized_schedule = schedule.copy()
        optimized_schedule["segments"] = optimized_segments
        return optimized_schedule

    def _create_transition_segment(
        self, current_segment: Dict, next_segment: Dict, patterns: Dict
    ) -> Dict:
        """Create a transition segment between two behavior segments"""
        # Calculate transition duration (10% of the shorter segment)
        current_duration = current_segment["end_time"] - current_segment["start_time"]
        next_duration = next_segment["end_time"] - next_segment["start_time"]
        transition_duration = min(current_duration, next_duration) * 0.1

        # Create transition segment
        transition_segment = {
            "start_time": current_segment["end_time"],
            "end_time": current_segment["end_time"] + transition_duration,
            "behavior_type": "transition",
            "from_behavior": current_segment["behavior_type"],
            "to_behavior": next_segment["behavior_type"],
        }

        return transition_segment
