#!/usr/bin/env python3
"""
检查行为标签文件，确认实际行为类别数量
"""

import json
import numpy as np
from network_simulation.utils.logger import get_logger

# 初始化日志记录器
logger = get_logger(__name__)

# 读取行为标签文件
with open(
    "/Users/xiaotuanzi/PycharmProjects/NPS/data/results/e2e_pipeline/patterns/behavior_labels.json",
    "r",
) as f:
    data = json.load(f)

# 获取标签数组
labels = np.array(data["labels"])

logger.info(f"原始行为标签数量: {len(labels)}")
logger.info(f"唯一行为标签: {np.unique(labels)}")
logger.info(f"行为标签分布: { {label: np.sum(labels == label) for label in np.unique(labels)} }")
