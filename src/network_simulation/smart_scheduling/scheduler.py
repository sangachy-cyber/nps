#!/usr/bin/env python3
"""
Smart Scheduling Module
Responsible for scheduling network behavior patterns
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from network_simulation.utils.logger import get_logger

# 初始化日志记录器
logger = get_logger(__name__)


class SmartScheduler:
    """Smart scheduler for network behavior patterns"""

    def __init__(self):
        pass

    def load_schedule(self, schedule_path: Path) -> Dict:
        """Load behavior schedule from file"""
        logger.info(f"从文件加载调度表: {schedule_path}")
        with open(schedule_path, "r") as f:
            schedule = json.load(f)
        logger.debug(f"成功加载调度表，包含 {len(schedule.get('segments', []))} 个行为段")
        return schedule

    def load_patterns(self, patterns_dir: Path) -> Dict:
        """Load discovered patterns from directory"""
        logger.info(f"从目录加载行为模式: {patterns_dir}")

        # Load behavior labels
        with open(patterns_dir / "behavior_labels.json", "r") as f:
            behavior_labels = json.load(f)
        logger.debug(f"成功加载行为标签，包含 {len(behavior_labels.get('cluster_stats', []))} 个簇")

        # Load transition graph
        with open(patterns_dir / "behavior_transition_graph.json", "r") as f:
            transition_graph = json.load(f)
        logger.debug("成功加载行为转移图")

        patterns = {
            "behavior_labels": behavior_labels,
            "transition_graph": transition_graph,
            "cluster_stats": behavior_labels["cluster_stats"],
        }

        return patterns

    def validate_schedule(self, schedule: Dict, patterns: Dict) -> Tuple[bool, str]:
        """Validate if the schedule is reasonable based on transition probabilities"""
        logger.info("开始验证调度表合理性")
        transition_matrix = np.array(patterns["transition_graph"]["transition_matrix"])
        segments = schedule["segments"]

        # Check if segments are in order and don't overlap
        for i in range(len(segments) - 1):
            if segments[i]["end_time"] > segments[i + 1]["start_time"]:
                message = f"分段 {i} 和 {i + 1} 重叠"
                logger.warning(f"调度表验证失败: {message}")
                return False, message

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
            logger.debug(f"从 {current_behavior} 到 {next_behavior} 的转移概率: {transition_prob:.4f}")

            if transition_prob < 0.01:  # Threshold for reasonable transition
                message = f"从 {current_behavior} 到 {next_behavior} 的转移不太可能发生 (概率: {transition_prob:.4f})"
                logger.warning(f"调度表验证失败: {message}")
                return False, message

        logger.info("调度表验证通过")
        return True, "Schedule is valid"

    def optimize_schedule(self, schedule: Dict, patterns: Dict) -> Dict:
        """Optimize the schedule based on transition probabilities"""
        # For now, just return the original schedule
        # Later, implement more sophisticated optimization
        return schedule

    def insert_transition_segments(self, schedule: Dict, patterns: Dict) -> Dict:
        """Insert transition segments between behavior segments"""
        logger.info("开始在行为段之间插入过渡段")
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
                logger.debug(f"插入从 {current_segment['behavior_type']} 到 {next_segment['behavior_type']} 的过渡段")

        optimized_schedule = schedule.copy()
        optimized_schedule["segments"] = optimized_segments
        logger.info(f"过渡段插入完成，优化后的调度表包含 {len(optimized_segments)} 个段（原 {len(segments)} 个）")
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
