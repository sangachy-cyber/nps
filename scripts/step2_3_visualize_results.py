#!/usr/bin/env python3
"""
步骤2.3：可视化结果

该脚本负责调用 ResultsVisualizer 类进行结果可视化，
只包含流程控制逻辑，不包含具体的业务实现。
"""

import sys
import os

# 添加根目录和src目录到Python路径
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

from pathlib import Path
from network_simulation.utils.logger import get_logger
from network_simulation.visualization.results_visualizer import ResultsVisualizer

# 初始化日志记录器
logger = get_logger(__name__)


def main():
    """主函数入口

    解析命令行参数，调用 ResultsVisualizer 类可视化生成结果与原始数据的对比。

    命令行参数：
        python scripts/step2_3_visualize_results.py <input_generation_dir> <output_visualization_dir>

    参数说明：
        input_generation_dir: 生成样本的目录路径，包含original_sample_6000*.csv和generated_sample_6000*.csv文件
        output_visualization_dir: 可视化结果的输出目录路径
    """
    if len(sys.argv) < 3:
        logger.error(
            "用法: python scripts/step2_3_visualize_results.py <input_generation_dir> <output_visualization_dir>"
        )
        sys.exit(1)

    input_generation_dir = Path(sys.argv[1])
    output_visualization_dir = Path(sys.argv[2])

    # 确保输出目录存在
    output_visualization_dir.mkdir(parents=True, exist_ok=True)

    # 创建 ResultsVisualizer 实例
    visualizer = ResultsVisualizer(output_visualization_dir)

    # 批量可视化结果
    visualizer.visualize_batch_results(input_generation_dir)

    logger.info("步骤2.3：可视化结果完成！")


if __name__ == "__main__":
    main()
