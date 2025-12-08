#!/usr/bin/env python3
"""
数据加载器测试用例
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.network_simulation.data_processing.data_loader import DataLoader


@pytest.fixture
def data_loader():
    """初始化数据加载器"""
    return DataLoader()


@pytest.fixture
def sample_csv_data():
    """创建测试CSV数据"""
    timestamps = pd.date_range(start="2025-01-01", periods=100, freq="100ms")
    data = {
        "timestamp": timestamps,
        "delay": np.random.randn(100) * 10 + 20,
        "loss_rate": np.random.choice([0.0, 0.5, 1.0], 100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_data_with_percentage_loss():
    """创建测试CSV数据，包含百分比形式的丢包率"""
    timestamps = pd.date_range(start="2025-01-01", periods=100, freq="100ms")
    data = {
        "timestamp": timestamps,
        "delay": np.random.randn(100) * 10 + 20,
        "loss_rate": np.random.choice([0.0, 50.0, 100.0], 100),  # 百分比形式
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_holowan_data():
    """创建测试HoloWAN Recorder File格式数据"""
    return """HoloWAN Recorder File (www.msytest.com)
Start Time: 2025-01-01 00:00:00
Interval(sec): 0.1
Duration(sec): 10
Packet Size(bytes): 1500
Baud Rate(bps): 1000000000
Channel: 1
Direction: 1
Interface: eth0
Bandwidth(Mbps): 1000
Mode: 1
CRC: 0x1234

Delay1(ms),Loss1(%),Bandwidth1(Mbps),Delay2(ms),Loss2(%),Bandwidth2(Mbps)
--------------------------------------------------------------------------
10.0,0.0,1000.0,10.0,0.0,1000.0
20.0,50.0,500.0,20.0,50.0,500.0
30.0,100.0,0.0,30.0,100.0,0.0
"""


@pytest.fixture
def sample_data_with_large_delay():
    """创建包含大延迟的测试数据"""
    timestamps = pd.date_range(start="2025-01-01", periods=100, freq="100ms")
    delay = np.random.randn(100) * 10 + 20
    delay[50:] = 3000  # 后半部分设置为超过2000ms的延迟
    data = {
        "timestamp": timestamps,
        "delay": delay,
        "loss_rate": np.random.choice([0.0, 0.5, 1.0], 100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def tmp_output_dir(tmp_path):
    """创建临时输出目录"""
    output_dir = tmp_path / "test_data"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def sample_csv_file(tmp_path, sample_csv_data):
    """创建测试CSV文件"""
    csv_file = tmp_path / "sample_data.csv"
    sample_csv_data.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def sample_csv_file_with_percentage_loss(tmp_path, sample_csv_data_with_percentage_loss):
    """创建测试CSV文件，包含百分比形式的丢包率"""
    csv_file = tmp_path / "sample_data_percentage.csv"
    sample_csv_data_with_percentage_loss.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def sample_holowan_file(tmp_path, sample_holowan_data):
    """创建测试HoloWAN文件"""
    holowan_file = tmp_path / "sample_holowan.hlw"
    with open(holowan_file, "w") as f:
        f.write(sample_holowan_data)
    return holowan_file


def test_data_loader_initialization(data_loader):
    """测试数据加载器初始化"""
    assert data_loader is not None
    assert data_loader.time_granularity == 0.1  # 默认100ms


def test_load_csv_data(data_loader, sample_csv_file, sample_csv_data):
    """测试加载CSV格式数据"""
    # 加载数据
    df = data_loader.load(sample_csv_file)

    # 验证数据加载结果
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_csv_data)
    assert list(df.columns) == ["timestamp", "delay", "loss_rate"]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert pd.api.types.is_float_dtype(df["delay"])
    assert pd.api.types.is_float_dtype(df["loss_rate"])

    # 验证丢包率在0-1范围内
    assert df["loss_rate"].min() >= 0
    assert df["loss_rate"].max() <= 1


def test_load_csv_data_with_percentage_loss(data_loader, sample_csv_file_with_percentage_loss):
    """测试加载包含百分比形式丢包率的CSV数据"""
    # 加载数据
    df = data_loader.load(sample_csv_file_with_percentage_loss)

    # 验证数据加载结果
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100

    # 验证丢包率已转换为小数形式且在0-1范围内
    assert df["loss_rate"].min() >= 0
    assert df["loss_rate"].max() <= 1
    # 验证至少有一个转换后的丢包率值
    assert any(df["loss_rate"].isin([0.0, 0.5, 1.0]))


def test_load_holowan_data(data_loader, sample_holowan_file):
    """测试加载HoloWAN Recorder File格式数据"""
    # 加载数据
    df = data_loader.load(sample_holowan_file)

    # 验证数据加载结果
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # 3行数据
    assert list(df.columns) == ["timestamp", "delay", "loss_rate"]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    # 验证丢包率处理
    assert df["loss_rate"].min() >= 0
    assert df["loss_rate"].max() <= 1
    # 验证带宽为0时丢包率设置为1.0
    assert df.iloc[2]["loss_rate"] == 1.0  # 第三行带宽为0


def test_preprocess(data_loader, sample_csv_data):
    """测试数据预处理"""
    # 预处理数据
    df_processed = data_loader.preprocess(sample_csv_data)

    # 验证预处理结果
    assert isinstance(df_processed, pd.DataFrame)
    assert len(df_processed) > 0
    assert list(df_processed.columns) == ["timestamp", "delay", "loss_rate"]

    # 验证丢包率在0-1范围内
    assert df_processed["loss_rate"].min() >= 0
    assert df_processed["loss_rate"].max() <= 1

    # 验证时间戳排序
    assert df_processed["timestamp"].is_monotonic_increasing


def test_preprocess_with_large_delay(data_loader, sample_data_with_large_delay):
    """测试处理包含大延迟的数据"""
    # 预处理数据
    df_processed = data_loader.preprocess(sample_data_with_large_delay)

    # 验证数据被截断
    assert len(df_processed) < len(sample_data_with_large_delay)
    # 验证截断后的数据中没有超过2000ms的延迟
    assert df_processed["delay"].max() <= 2000


def test_save_data(data_loader, sample_csv_data, tmp_output_dir):
    """测试保存处理后的数据"""
    # 预处理数据
    df_processed = data_loader.preprocess(sample_csv_data)

    # 保存数据
    output_path = tmp_output_dir / "processed_data.csv"
    data_loader.save(df_processed, output_path)

    # 验证文件存在
    assert output_path.exists()

    # 验证文件内容
    loaded_df = pd.read_csv(output_path, parse_dates=["timestamp"])
    assert isinstance(loaded_df, pd.DataFrame)
    assert len(loaded_df) == len(df_processed)
    assert list(loaded_df.columns) == list(df_processed.columns)


def test_preprocess_empty_data(data_loader):
    """测试处理空数据"""
    # 创建空数据框
    empty_df = pd.DataFrame(columns=["timestamp", "delay", "loss_rate"])

    # 预处理数据
    df_processed = data_loader.preprocess(empty_df)

    # 验证处理结果为空
    assert len(df_processed) == 0


def test_load_nonexistent_file(data_loader):
    """测试加载不存在的文件"""
    nonexistent_file = Path("/nonexistent/path/to/file.csv")

    # 验证加载不存在的文件会抛出异常
    with pytest.raises(ValueError):
        data_loader.load(nonexistent_file)
