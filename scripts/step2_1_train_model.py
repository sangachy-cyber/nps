#!/usr/bin/env python3
"""
步骤2.1：训练条件扩散模型
"""

import sys
import os

# 添加当前目录到Python路径，以便找到config模块
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

import torch
import torch.optim as optim
import torch.cuda.amp as amp
import numpy as np
from pathlib import Path

from network_simulation.condition_generation.diffusion_model import (
    ConditionDiffusionModel,
)
from network_simulation.condition_generation.constraint_injector import (
    ConstraintInjector,
)
from network_simulation.utils.logger import get_logger
from config import (
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_SAMPLE_LENGTH
)


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

    # 加载预处理数据
    preprocess_file = input_preprocess_dir / "preprocess_data.npz"
    if not preprocess_file.exists():
        logger.error(f"预处理文件 {preprocess_file} 不存在！")
        sys.exit(1)

    preprocess_data = np.load(preprocess_file, allow_pickle=True)
    features = preprocess_data["features"]
    behavior_ids = preprocess_data["behavior_ids"]
    behavior_mapping = preprocess_data["behavior_mapping"].item()
    valid_loss_values = preprocess_data["valid_loss_values"].tolist()
    num_behaviors = preprocess_data["num_behaviors"].item()

    # 检查归一化方法
    normalization_method = preprocess_data.get("normalization_method", "robust")
    # 将numpy字符串数组转换为Python字符串
    if isinstance(normalization_method, np.ndarray):
        normalization_method = normalization_method.item()
    # 检查是否使用对数变换
    use_log_transform = preprocess_data.get("use_log_transform", False)
    # 将numpy布尔值转换为Python布尔值
    if isinstance(use_log_transform, np.ndarray):
        use_log_transform = use_log_transform.item()
    logger.info(f"是否使用对数变换: {use_log_transform}")

    # 限制样本数量，防止内存不足
    if max_samples is not None and len(features) > max_samples:
        logger.info(f"样本数量过大 ({len(features)}), 将使用前 {max_samples} 个样本进行训练")
        features = features[:max_samples]
        behavior_ids = behavior_ids[:max_samples]

    # 根据归一化方法加载不同的参数
    if normalization_method == "robust":
        # 加载robust归一化参数
        delay_scaler_center_ = preprocess_data["delay_scaler_center_"]
        delay_scaler_scale_ = preprocess_data["delay_scaler_scale_"]
        # 加载robust_scale_min和robust_scale_max
        robust_scale_min = preprocess_data.get("robust_scale_min", 0)
        robust_scale_max = preprocess_data.get("robust_scale_max", 0)
    else:
        # 加载分位数归一化参数
        delay_scaler_center_ = None
        delay_scaler_scale_ = None
        robust_scale_min = None
        robust_scale_max = None
        quantile_sorted_data = preprocess_data["quantile_sorted_data"]
        quantile_quantiles = preprocess_data["quantile_quantiles"]

    logger.info(f"归一化方法: {normalization_method}")

    # 初始化约束注入器
    constraint_injector = ConstraintInjector(valid_loss_values)

    # 设备设置 - 支持MacBook M1/M2/M3 GPU (MPS)和NVIDIA GPU (CUDA)
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # 实现真正的batch training
    # 从配置文件加载训练参数
    lr = DEFAULT_LEARNING_RATE  # 学习率
    num_epochs = DEFAULT_EPOCHS  # 训练轮数
    batch_size = DEFAULT_BATCH_SIZE  # 批次大小
    batch_seq_len = DEFAULT_SAMPLE_LENGTH // 12  # 每个批次的序列长度

    # 打印超参数
    logger.info("\n训练超参数:")
    logger.info(f"学习率: {lr}")
    logger.info(f"训练轮数: {num_epochs}")
    logger.info(f"批次大小: {batch_size}")
    logger.info(f"每个批次序列长度: {batch_seq_len}")

    # 将数据重塑为适合batch training的形状
    # 原始数据形状: (num_samples, 2) -> 转换为: (num_batches, batch_size, batch_seq_len, 2)
    num_samples = len(features)
    total_batch_samples = batch_size * batch_seq_len
    num_batches = num_samples // total_batch_samples
    if num_samples % total_batch_samples != 0:
        num_batches += 1

    # 调整数据长度以适应批次大小
    adjusted_samples = num_batches * total_batch_samples
    if adjusted_samples > num_samples:
        # 补零到最近的批次边界
        features = np.pad(features, ((0, adjusted_samples - num_samples), (0, 0)), mode='constant')
        behavior_ids = np.pad(behavior_ids, (0, adjusted_samples - num_samples), mode='constant')

    # 重塑数据
    features_reshaped = features.reshape(num_batches, batch_size, batch_seq_len, 2)
    behavior_ids_reshaped = behavior_ids.reshape(num_batches, batch_size, batch_seq_len)

    # 转换为张量并移动到设备
    features_tensor = torch.tensor(features_reshaped, dtype=torch.float32).to(device)
    behavior_ids_tensor = torch.tensor(behavior_ids_reshaped, dtype=torch.long).to(device)

    # 初始化条件扩散模型
    model = ConditionDiffusionModel(
        input_dim=2,  # 输入维度：延迟和丢包率
        num_behaviors=num_behaviors,
        behavior_embed_dim=32,  # 行为嵌入维度
        T=1000,  # 扩散步数
    )
    model.to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 创建混合精度训练的GradScaler
    scaler = amp.GradScaler(enabled=True)

    # 创建输出目录
    output_train_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    best_loss = float("inf")
    loss_history = []

    logger.info(f"开始在 {device} 上训练")
    logger.info(f"原始数据形状: ({num_samples}, 2)")
    logger.info(f"批次数据形状: {features_tensor.shape}")
    logger.info(f"批次数量: {num_batches}")
    logger.info(f"批次大小: {batch_size}")
    logger.info(f"每个批次序列长度: {batch_seq_len}")
    logger.info(f"行为数量: {num_behaviors}")
    logger.info("启用混合精度训练")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # 遍历所有批次
        for batch_idx in range(num_batches):
            optimizer.zero_grad()

            # 获取当前批次
            batch_features = features_tensor[batch_idx]
            batch_behavior_ids = behavior_ids_tensor[batch_idx]

            # 使用混合精度上下文管理器
            with amp.autocast(enabled=True):
                # 计算损失
                loss = model.compute_loss(batch_features, batch_behavior_ids)

            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        # 计算平均损失
        avg_epoch_loss = epoch_loss / num_batches
        loss_history.append(avg_epoch_loss)

        logger.info(f"训练轮次 [{epoch + 1}/{num_epochs}], 平均损失值: {avg_epoch_loss:.4f}")

        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_path = output_train_dir / "diffusion_model_best.pth"

            # 构建模型检查点
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "behavior_mapping": behavior_mapping,
                "normalization_method": normalization_method,
                "use_log_transform": use_log_transform,
                "valid_loss_values": valid_loss_values,
            }

            # 根据归一化方法添加不同的参数
            if normalization_method == "robust":
                # 添加robust归一化参数
                checkpoint.update({
                    "delay_scaler_center_": delay_scaler_center_,
                    "delay_scaler_scale_": delay_scaler_scale_,
                    "robust_scale_min": robust_scale_min,
                    "robust_scale_max": robust_scale_max,
                })
            else:
                # 添加分位数归一化参数
                checkpoint.update({
                    "quantile_sorted_data": quantile_sorted_data,
                    "quantile_quantiles": quantile_quantiles,
                })

            # 保存模型
            torch.save(checkpoint, best_model_path)

    # 保存最终模型
    final_model_path = output_train_dir / "diffusion_model_final.pth"

    # 构建最终模型检查点
    final_checkpoint = {
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item(),
        "behavior_mapping": behavior_mapping,
        "normalization_method": normalization_method,
        "use_log_transform": use_log_transform,
        "valid_loss_values": valid_loss_values,
    }

    # 根据归一化方法添加不同的参数
    if normalization_method == "robust":
        # 添加robust归一化参数
        final_checkpoint.update({
            "delay_scaler_center_": delay_scaler_center_,
            "delay_scaler_scale_": delay_scaler_scale_,
            "robust_scale_min": robust_scale_min,
            "robust_scale_max": robust_scale_max,
        })
    else:
        # 添加分位数归一化参数
        final_checkpoint.update({
            "quantile_sorted_data": quantile_sorted_data,
            "quantile_quantiles": quantile_quantiles,
        })

    # 保存最终模型
    torch.save(final_checkpoint, final_model_path)

    # 保存训练历史
    loss_history_path = output_train_dir / "loss_history.npy"
    np.save(loss_history_path, np.array(loss_history))

    print("训练完成！")
    print(f"最佳损失值: {best_loss:.4f}")
    print(f"最终模型已保存到: {final_model_path}")
    print(f"训练历史已保存到: {loss_history_path}")

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
