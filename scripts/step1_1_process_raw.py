#!/usr/bin/env python3
"""
步骤1.1：处理原始数据
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(os.path.abspath("src"))

# 导入日志模块和配置
from network_simulation.utils.logger import get_logger
from network_simulation.data_processing.data_processor import DataProcessor


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

    # 初始化数据处理器
    data_processor = DataProcessor()

    # 调用核心模块的处理函数
    output_file = data_processor.process_raw_data(input_file, output_dir)

    logger.info(f"处理后数据已保存到: {output_file}")
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
