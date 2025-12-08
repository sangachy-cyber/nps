#!/usr/bin/env python3
"""
步骤1.1：处理原始数据
"""

import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.abspath("src"))

import pandas as pd
from pathlib import Path

# 导入日志模块和配置
from network_simulation.utils.logger import get_logger


# 获取日志记录器
logger = get_logger(__name__)

def process_raw_data(input_file: Path, output_dir: Path):
    """处理单个原始网络数据文件

    该函数读取原始网络数据文件，解析延迟和丢包率数据，并将其转换为结构化的CSV格式。

    Args:
        input_file: 原始数据文件路径，包含网络测量数据
        output_dir: 处理后数据的输出目录路径

    Returns:
        Path: 处理后的数据文件路径

    Raises:
        FileNotFoundError: 如果输入文件不存在
        ValueError: 如果数据格式不符合要求

    输入输出示例：
        输入：
            input_file: Path("data/raw/20251203_230356_b6x-playback.txt")
            output_dir: Path("data/processed/")
        输出：
            PosixPath('data/processed/20251203_230356_b6x-playback_processed.csv')
    """
    logger.info(f"正在处理原始数据文件: {input_file}")

    # 读取原始文件内容
    try:
        with open(input_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"原始数据文件不存在: {input_file}")
    except PermissionError:
        raise PermissionError(f"没有读取文件的权限: {input_file}")

    # 跳过前12行元数据和分隔线，只保留数据行
    if len(lines) < 13:
        logger.error(f"文件格式不正确，缺少足够的元数据行: {input_file}")
        raise ValueError(f"文件格式不正确，缺少足够的元数据行: {input_file}")

    data_lines = lines[12:]

    # 提取开始时间
    try:
        start_time_line = [line for line in lines if "Start Time" in line][0]
        start_time_str = start_time_line.split(": ", 1)[1].strip()
        start_time = pd.Timestamp(start_time_str)
    except (IndexError, ValueError) as e:
        raise ValueError(f"无法提取开始时间: {e}")

    # 提取时间间隔（秒）
    try:
        interval_line = [line for line in lines if "Interval(sec)" in line][0]
        interval_sec = float(interval_line.split(": ", 1)[1].strip())
    except (IndexError, ValueError) as e:
        raise ValueError(f"无法提取时间间隔: {e}")

    # 解析数据行
    data = []
    for i, line in enumerate(data_lines):
        line = line.strip()
        if not line:
            continue

        # 分割数据行，按逗号分隔
        parts = line.split(",")
        if len(parts) < 6:
            continue

        try:
            # 提取Delay1、Loss1和Bandwidth1
            delay = float(parts[0])
            loss_rate = float(parts[1]) / 100  # 转换为0-1范围
            bandwidth1 = float(parts[2])
        except ValueError:
            print(f"警告: 第{i}行的数据格式不正确，跳过该行")
            continue

        # 当遇到时延大于2000ms的数据时，忽略它和它之后的所有数据
        if delay > 2000:
            print(f"在第{i}行发现时延大于2000ms，停止处理")
            break

        # 当Bandwidth1为0时，设置loss_rate为1.0（100%丢包）
        if bandwidth1 == 0:
            loss_rate = 1.0

        # 计算该数据点的时间戳
        timestamp = start_time + pd.Timedelta(seconds=interval_sec * i)

        data.append({"timestamp": timestamp, "delay": delay, "loss_rate": loss_rate})

    if not data:
        raise ValueError(f"没有提取到有效数据: {input_file}")

    # 创建DataFrame并打印数据统计信息
    df = pd.DataFrame(data)
    
    # 添加原始文件路径列
    df['file_path'] = str(input_file)
    
    logger.info(f"处理后数据形状: {df.shape}")
    logger.info(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    logger.info(f"时延范围: {df['delay'].min():.2f} 到 {df['delay'].max():.2f} ms")
    logger.info(f"丢包率范围: {df['loss_rate'].min():.4f} 到 {df['loss_rate'].max():.4f}")

    # 保存处理后的数据到CSV文件
    output_file = output_dir / f"{input_file.stem}_processed.csv"
    df.to_csv(output_file, index=False)
    print(f"处理后数据已保存到: {output_file}")

    return output_file


def main():
    """主函数入口

    解析命令行参数，调用process_raw_data函数处理原始网络数据文件。

    命令行参数：
        python scripts/step1_1_process_raw.py <input_raw_file_or_dir> <output_processed_dir>

    参数说明：
        input_raw_file_or_dir: 原始数据文件或目录路径
        output_processed_dir: 处理后数据的输出目录路径
    """
    if len(sys.argv) < 3:
        print(
            "用法: python scripts/step1_1_process_raw.py <input_raw_file_or_dir> <output_processed_dir>"
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理原始数据
    if input_path.is_file():
        # 如果输入是文件，处理单个文件
        process_raw_data(input_path, output_dir)
    elif input_path.is_dir():
        # 如果输入是文件夹，处理所有.txt文件
        raw_files = list(input_path.glob("*.txt"))
        if not raw_files:
            print(f"警告: 在 {input_path} 中未找到 .txt 文件")
            sys.exit(1)

        for raw_file in raw_files:
            print(f"\n正在处理 {raw_file}...")
            process_raw_data(raw_file, output_dir)
    else:
        print(f"错误: 输入 {input_path} 不是文件或目录")
        sys.exit(1)

    print("步骤1.1：处理原始数据完成！")


if __name__ == "__main__":
    main()
