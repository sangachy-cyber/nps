#!/usr/bin/env python3
"""
测试可视化中文支持
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# 导入日志模块
from src.network_simulation.utils.logger import get_logger

# 获取日志记录器
logger = get_logger(__name__)

# 设置中文支持，兼容Windows、MacOS和Linux，Linux系统优先使用文泉驿正黑
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def test_chinese_visualization():
    """测试中文可视化"""
    logger.info("开始测试中文可视化...")
    
    # 创建测试数据
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y1 = [1, 3, 2, 4, 3, 5, 4, 6, 5, 7]
    y2 = [2, 4, 3, 5, 4, 6, 5, 7, 6, 8]
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label="曲线1", marker="o")
    plt.plot(x, y2, label="曲线2", marker="s")
    
    # 添加中文标题和标签
    plt.title("中文标题：测试曲线对比")
    plt.xlabel("X轴标签")
    plt.ylabel("Y轴标签")
    
    # 添加中文图例
    plt.legend(title="图例标题")
    
    # 添加网格
    plt.grid(True)
    
    # 保存图表
    output_dir = Path("tmp")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "test_chinese_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"中文可视化测试完成！图表已保存到：{output_file}")
    logger.info("请查看图表确认中文是否正常显示。")


if __name__ == "__main__":
    test_chinese_visualization()
