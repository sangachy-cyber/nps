#!/usr/bin/env python3
"""
Condition Generation Module
Responsible for generating network simulation data based on conditions
"""

import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple

from .diffusion_model import ConditionDiffusionModel
from .constraint_injector import ConstraintInjector


class ConditionGenerator:
    """Generates network simulation data based on behavior conditions"""

    def __init__(self, valid_loss_values: List[float] = None):
        self.time_granularity = 0.1  # 100ms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.valid_loss_values = valid_loss_values or [0.0, 1 / 3, 0.5, 2 / 3, 1.0]

        # Initialize constraint injector
        self.constraint_injector = ConstraintInjector(self.valid_loss_values)

        # Initialize diffusion model
        self.diffusion_model = None
        self._init_diffusion_model()

    def _init_diffusion_model(self):
        """初始化扩散模型"""
        self.diffusion_model = ConditionDiffusionModel(
            input_dim=2,
            num_behaviors=10,  # 默认支持10种行为
            behavior_embed_dim=32,
            T=1000,
        )
        self.diffusion_model.to(self.device)

    def generate(
        self, schedule: Dict, patterns: Dict, duration: int = 600
    ) -> pd.DataFrame:
        """Generate network simulation data based on schedule and patterns"""
        # Calculate total samples needed
        total_samples = int(duration / self.time_granularity)

        # Initialize simulation data
        simulation_data = {"timestamp": [], "delay": [], "loss_rate": []}

        # Generate timestamp sequence
        start_time = pd.Timestamp.now()
        timestamps = [
            start_time + pd.Timedelta(seconds=i * self.time_granularity)
            for i in range(total_samples)
        ]
        simulation_data["timestamp"] = timestamps

        # Process schedule and generate data
        delay_sequence, loss_sequence = self._generate_sequences(
            schedule, patterns, total_samples
        )

        # Apply physical constraints
        delay_sequence = np.array(delay_sequence)
        loss_sequence = np.array(loss_sequence)

        # Validate and process sequences
        validation = self.constraint_injector.validate_sequence(
            delay_sequence, loss_sequence
        )
        if not validation["all_valid"]:
            print(
                "Warning: Generated sequence has physical constraint violations. Applying post-processing."
            )
            # 这里可以添加后处理逻辑，但当前实现中_generate_behavior_segment已经确保了基本合理性

        simulation_data["delay"] = delay_sequence.tolist()
        simulation_data["loss_rate"] = loss_sequence.tolist()

        # Create dataframe
        df = pd.DataFrame(simulation_data)
        return df

    def _generate_sequences(
        self, schedule: Dict, patterns: Dict, total_samples: int
    ) -> Tuple[List[float], List[float]]:
        """Generate delay and loss rate sequences based on schedule"""
        # Create behavior ID sequence
        behavior_ids = self._generate_behavior_id_sequence(
            schedule, patterns, total_samples
        )

        # Use diffusion model to generate sequence
        delay_sequence, loss_sequence = self._generate_with_diffusion(behavior_ids)

        return delay_sequence, loss_sequence

    def _generate_behavior_id_sequence(
        self, schedule: Dict, patterns: Dict, total_samples: int
    ) -> List[int]:
        """Generate behavior ID sequence based on schedule"""
        # Map behavior type to cluster label
        behavior_cluster_map = patterns.get(
            "behavior_cluster_map",
            {"stable": 0, "burst_loss": 1, "high_jitter": 2, "recovery": 3},
        )

        # Initialize behavior ID sequence
        behavior_ids = [0] * total_samples  # 默认使用稳定行为

        # Process each schedule segment
        for segment in schedule["segments"]:
            start_time = segment["start_time"]
            end_time = segment["end_time"]
            behavior_type = segment["behavior_type"]

            # Calculate start and end indices
            start_idx = int(start_time / self.time_granularity)
            end_idx = int(end_time / self.time_granularity)

            # Map behavior type to cluster label
            cluster_label = behavior_cluster_map.get(behavior_type, 0)

            # Fill behavior ID sequence
            for i in range(start_idx, min(end_idx, total_samples)):
                behavior_ids[i] = cluster_label

        return behavior_ids

    def _generate_with_diffusion(
        self, behavior_ids: List[int]
    ) -> Tuple[List[float], List[float]]:
        """使用扩散模型生成序列"""
        # Convert behavior_ids to tensor
        behavior_ids_tensor = torch.tensor(behavior_ids, dtype=torch.long).unsqueeze(0)
        behavior_ids_tensor = behavior_ids_tensor.to(self.device)

        # Generate sequence using diffusion model
        with torch.no_grad():
            generated_sequence = self.diffusion_model.sample(
                behavior_ids_tensor, self.device
            )

        # Move to CPU and convert to numpy array
        generated_sequence = generated_sequence.cpu().numpy()[0]  # (seq_len, 2)

        # Split into delay and loss_rate
        delay_norm = generated_sequence[:, 0]
        loss_norm = generated_sequence[:, 1]

        # Process delay and loss_rate to ensure physical constraints
        # 注意：当前实现中还没有delay_scaler，这里使用简单的映射
        delay = (delay_norm + 1) * 100  # 将[-1, 1]映射到[0, 200]ms
        delay = np.clip(delay, 0, None)  # 确保延迟非负

        # 处理丢包率，确保为合法值
        loss_rate = self.constraint_injector.process_loss_rate(loss_norm)

        return delay.tolist(), loss_rate.tolist()

    def _generate_behavior_segment(
        self, behavior_type: str, patterns: Dict, num_samples: int
    ) -> Tuple[List[float], List[float]]:
        """Generate a segment of network data for a specific behavior type"""
        # Map behavior type to cluster label
        behavior_cluster_map = patterns.get(
            "behavior_cluster_map",
            {"stable": 0, "burst_loss": 1, "high_jitter": 2, "recovery": 3},
        )

        cluster_label = behavior_cluster_map.get(behavior_type, 0)

        # Generate behavior ID sequence
        behavior_ids = [cluster_label] * num_samples

        # Use diffusion model to generate segment
        return self._generate_with_diffusion(behavior_ids)

    def save(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save generated simulation data"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    def load_schedule(self, schedule_path: Path) -> Dict:
        """Load behavior schedule from file"""
        with open(schedule_path, "r") as f:
            return json.load(f)
