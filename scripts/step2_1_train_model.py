#!/usr/bin/env python3
"""
步骤2.1：训练条件扩散模型
"""

import sys
import os

# 添加当前目录到Python路径，以便找到config模块
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

from pathlib import Path

from network_simulation.utils.logger import get_logger


# 获取日志记录器
logger = get_logger(__name__)

def train_model(input_preprocess_dir: Path, output_train_dir: Path, max_samples: int = None):
    """训练条件扩散模型

    该函数使用预处理数据训练条件扩散模型，用于生成网络状态样本数据。

    Args:
        input_preprocess_dir: 预处理数据目录路径，包含preprocess_data.npz文件
        output_train_dir: 训练结果的输出目录路径
        max_samples: 最大使用的样本数量，用于控制内存使用

    Returns:
        Path: 最终训练模型的文件路径

    Raises:
        FileNotFoundError: 如果预处理文件不存在
        ValueError: 如果数据格式不符合要求

    输入输出示例：
        输入：
            input_preprocess_dir: Path("data/results/preprocessed/")
            output_train_dir: Path("output/train_results/")
        输出：
            PosixPath('output/train_results/diffusion_model_final.pth')
    """
    logger.info(f"正在从 {input_preprocess_dir} 训练条件扩散模型")

    # 导入TrainingManager
    from network_simulation.condition_generation.training_manager import TrainingManager

    # 初始化训练管理器
    training_manager = TrainingManager(input_preprocess_dir, output_train_dir, max_samples)

    # 执行训练
    final_model_path = training_manager.train()

    print("训练完成！")
    print(f"最终模型已保存到: {final_model_path}")

    return final_model_path


def cleanup_old_files(directory, pattern):
    """清理目录下的旧文件，只保留最新的KEEP_LATEST_FILES个文件

    Args:
        directory: 要清理的目录
        pattern: 要清理的文件模式
    """
    from config import CLEANUP_OLD_FILES, KEEP_LATEST_FILES

    if not CLEANUP_OLD_FILES:
        return

    files = list(directory.glob(pattern))
    if len(files) <= KEEP_LATEST_FILES:
        return

    # 按修改时间排序，最新的在前
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # 删除旧文件
    for file in files[KEEP_LATEST_FILES:]:
        file.unlink()
        print(f"已清理旧文件: {file}")


def main():
    """主函数入口

    解析命令行参数，调用train_model函数训练条件扩散模型。

    命令行参数：
        python scripts/step2_1_train_model.py [input_preprocess_dir] [output_train_dir] [max_samples]

    参数说明：
        input_preprocess_dir: 预处理数据目录路径，包含preprocess_data.npz文件 (默认: data/results/preprocess)
        output_train_dir: 训练结果的输出目录路径 (默认: data/models/diffusion_model)
        max_samples: 最大使用的样本数量，用于控制内存使用 (默认: 10000)
    """
    import argparse
    from config import PREPROCESS_DIR, DEFAULT_MODEL_DIR

    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="训练条件扩散模型")
    parser.add_argument(
        "input_preprocess",
        nargs="?",
        type=Path,
        default=PREPROCESS_DIR,
        help="预处理数据目录路径，包含preprocess_data.npz文件 (默认: data/results/preprocess)",
    )
    parser.add_argument(
        "output_train",
        nargs="?",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="训练结果的输出目录路径 (默认: data/models/diffusion_model)",
    )
    parser.add_argument(
        "max_samples",
        nargs="?",
        type=int,
        default=None,
        help="最大使用的样本数量，用于控制内存使用 (默认: None，表示使用所有样本)",
    )

    args = parser.parse_args()

    input_preprocess_dir = args.input_preprocess
    output_train_dir = args.output_train
    max_samples = args.max_samples

    # 确保输出目录存在
    output_train_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧文件
    cleanup_old_files(output_train_dir, "diffusion_model_*.pth")
    cleanup_old_files(output_train_dir, "loss_history.npy")

    # 训练模型
    train_model(input_preprocess_dir, output_train_dir, max_samples=max_samples)

    print("步骤2.1：训练条件扩散模型完成！")


if __name__ == "__main__":
    main()
