#!/usr/bin/env python3
"""
CLI模块测试用例
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.network_simulation.cli import main


@pytest.fixture
def tmp_data_dir(tmp_path):
    """创建临时数据目录"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def sample_raw_file(tmp_data_dir):
    """创建测试原始数据文件"""
    raw_file = tmp_data_dir / "sample_raw.txt"
    with open(raw_file, "w") as f:
        f.write("HoloWAN Recorder File (www.msytest.com)\n")
        f.write("Start Time: 2025-01-01 00:00:00\n")
        f.write("Interval(sec): 0.1\n")
        f.write("Duration(sec): 10\n")
        f.write("Packet Size(bytes): 1500\n")
        f.write("Baud Rate(bps): 1000000000\n")
        f.write("Channel: 1\n")
        f.write("Direction: 1\n")
        f.write("Interface: eth0\n")
        f.write("Bandwidth(Mbps): 1000\n")
        f.write("Mode: 1\n")
        f.write("CRC: 0x1234\n\n")
        f.write("Delay1(ms),Loss1(%),Bandwidth1(Mbps),Delay2(ms),Loss2(%),Bandwidth2(Mbps)\n")
        f.write("--------------------------------------------------------------------------\n")
        f.write("10.0,0.0,1000.0,10.0,0.0,1000.0\n")
        f.write("20.0,50.0,500.0,20.0,50.0,500.0\n")
    return raw_file


@pytest.fixture
def sample_processed_file(tmp_data_dir):
    """创建测试处理后的数据文件"""
    processed_file = tmp_data_dir / "processed_data.csv"
    with open(processed_file, "w") as f:
        f.write("timestamp,delay,loss_rate\n")
        f.write("2025-01-01 00:00:00,10.0,0.0\n")
        f.write("2025-01-01 00:00:00.1,20.0,0.5\n")
        f.write("2025-01-01 00:00:00.2,15.0,0.0\n")
    return processed_file


@pytest.fixture
def sample_features_file(tmp_data_dir):
    """创建测试特征文件"""
    features_file = tmp_data_dir / "features.csv"
    with open(features_file, "w") as f:
        f.write("feat_delay_std,feat_loss_burst_ratio,feat_burst_duration,feat_burst_intensity,feat_delay_trend,feat_delay_acf_5\n")
        f.write("1.0,0.0,0.0,0.0,0.0,0.0\n")
        f.write("2.0,0.5,0.5,0.5,0.5,0.5\n")
        f.write("3.0,1.0,1.0,1.0,1.0,1.0\n")
    return features_file


@pytest.fixture
def sample_schedule_file(tmp_data_dir):
    """创建测试调度文件"""
    schedule_file = tmp_data_dir / "schedule.json"
    with open(schedule_file, "w") as f:
        f.write("{")
        f.write("  \"segments\": [")
        f.write("    {")
        f.write("      \"start_time\": 0,")
        f.write("      \"end_time\": 5,")
        f.write("      \"behavior_type\": \"stable\"")
        f.write("    },")
        f.write("    {")
        f.write("      \"start_time\": 5,")
        f.write("      \"end_time\": 10,")
        f.write("      \"behavior_type\": \"burst\"")
        f.write("    }")
        f.write("  ]")
        f.write("}")
    return schedule_file