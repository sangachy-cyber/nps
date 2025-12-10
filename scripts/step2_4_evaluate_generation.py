#!/usr/bin/env python3
"""
步骤2.4：评估生成数据质量

该脚本负责调用 Evaluator 类进行生成数据质量评估，
只包含流程控制逻辑，不包含具体的业务实现。
"""

import sys
import os

# 添加根目录和src目录到Python路径
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

from pathlib import Path
from network_simulation.utils.logger import get_logger
from network_simulation.evaluation.evaluator import Evaluator

# 初始化日志记录器
logger = get_logger(__name__)


def main():
    """主函数入口

    解析命令行参数，调用 Evaluator 类评估生成数据的质量。

    命令行参数：
        python scripts/step2_4_evaluate_generation.py <input_generation_dir> [output_evaluation_dir]

    参数说明：
        input_generation_dir: 生成样本的目录路径，包含original_sample_6000*.csv和generated_sample_6000*.csv文件
        output_evaluation_dir: 评估结果的输出目录路径（可选，默认使用input_generation_dir/../evaluation）
    """
    if len(sys.argv) < 2:
        logger.error(
            "用法: python scripts/step2_4_evaluate_generation.py <input_generation_dir> [output_evaluation_dir]"
        )
        sys.exit(1)

    # 获取命令行参数
    input_generation_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_evaluation_dir = Path(sys.argv[2])
    else:
        # 默认输出目录
        output_evaluation_dir = input_generation_dir.parent / "evaluation"
        logger.info(f"使用默认输出目录: {output_evaluation_dir}")

    # 初始化 Evaluator 类
    evaluator = Evaluator()

    # 批量评估生成样本
    evaluator.evaluate_batch_samples(input_generation_dir, output_evaluation_dir)

    logger.info(
        f"\n所有样本评估完成！汇总报告已保存到: {output_evaluation_dir / 'summary_report.txt'}"
    )


if __name__ == "__main__":
    main()
