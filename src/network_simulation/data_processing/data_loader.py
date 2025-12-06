#!/usr/bin/env python3
"""
Data Processing Module
Responsible for loading, preprocessing, and saving network data files
"""

import pandas as pd
from pathlib import Path


class DataLoader:
    """Data loader and preprocessor for network simulation data"""

    def __init__(self):
        self.time_granularity = 0.1  # 100ms as per requirements

    def load(self, file_path: Path) -> pd.DataFrame:
        """Load raw network data from file (supports CSV and HoloWAN Recorder File formats)"""
        try:
            with open(file_path, "r") as f:
                first_line = f.readline().strip()

            # Check if it's a HoloWAN Recorder File
            if first_line == "HoloWAN Recorder File (www.msytest.com)":
                return self._load_holowan_file(file_path)
            else:
                # Load as standard CSV
                df = pd.read_csv(
                    file_path,
                    parse_dates=["timestamp"],
                    dtype={"delay": float, "loss_rate": float},
                )

                # Ensure loss_rate is between 0 and 1
                # If values are in percentage (greater than 1), convert to decimal
                if df["loss_rate"].max() > 1:
                    df["loss_rate"] = df["loss_rate"] / 100.0

                # Ensure loss_rate is between 0 and 1
                df["loss_rate"] = df["loss_rate"].clip(0, 1)

                return df
        except Exception as e:
            raise ValueError(f"Failed to load data from {file_path}: {e}")

    def _load_holowan_file(self, file_path: Path) -> pd.DataFrame:
        """Load HoloWAN Recorder File format"""
        # Read metadata
        metadata = {}
        with open(file_path, "r") as f:
            for i in range(12):
                line = f.readline().strip()
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip().strip('"')

        # Extract start time and interval
        start_time = pd.to_datetime(metadata["Start Time"])
        interval = float(metadata["Interval(sec)"])

        # Read data rows, skipping header and separator
        df = pd.read_csv(
            file_path,
            skiprows=13,  # Skip first 13 lines (metadata + separator)
            header=None,
            names=[
                "Delay1(ms)",
                "Loss1(%)",
                "Bandwidth1(Mbps)",
                "Delay2(ms)",
                "Loss2(%)",
                "Bandwidth2(Mbps)",
            ],
        )

        # Generate timestamp sequence
        timestamps = [
            start_time + pd.Timedelta(seconds=i * interval) for i in range(len(df))
        ]

        # Create standard format dataframe with proper loss rate handling
        # When Bandwidth1 is 0, it indicates 100% packet loss
        loss_rate = (
            df["Loss1(%)"].values / 100.0
        )  # Convert percentage to decimal initially
        bandwidth1 = df["Bandwidth1(Mbps)"].values

        # Set loss_rate to 1.0 (100%) when Bandwidth1 is 0
        loss_rate[bandwidth1 == 0] = 1.0

        result_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "delay": df["Delay1(ms)"].values,
                "loss_rate": loss_rate,
            }
        )

        return result_df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw network data"""
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Check for delay > 2000ms and truncate data if found
        # Find the index where delay first exceeds 2000ms
        delay_exceed_idx = df[df["delay"] > 2000].index
        if not delay_exceed_idx.empty:
            # Get the first occurrence index
            cutoff_idx = delay_exceed_idx[0]
            # Keep only data before this index
            df = df.iloc[:cutoff_idx]
            if len(df) == 0:
                return df

        # Resample to 100ms granularity
        df_resampled = (
            df.set_index("timestamp")
            .resample(f"{int(self.time_granularity * 1000)}ms")
            .agg({"delay": "mean", "loss_rate": "mean"})
            .reset_index()
        )

        # Fill missing values using linear interpolation
        df_resampled = df_resampled.interpolate(method="linear")

        # Ensure loss_rate is between 0 and 1
        df_resampled["loss_rate"] = df_resampled["loss_rate"].clip(0, 1)

        return df_resampled

    def save(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save processed data to file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
