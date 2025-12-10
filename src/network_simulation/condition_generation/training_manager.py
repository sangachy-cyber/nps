#!/usr/bin/env python3
"""
训练管理器类
封装完整的条件扩散模型训练流程
"""

import sys
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from typing import Optional

from network_simulation.condition_generation.diffusion_model import ConditionDiffusionModel
from network_simulation.condition_generation.normalization import Normalizer
from network_simulation.utils.logger import get_logger
from config import (
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE
)

# 获取日志记录器
logger = get_logger(__name__)


class TrainingManager:
    """训练管理器，封装完整的条件扩散模型训练流程"""

    def __init__(self,
                 input_preprocess_dir: Path,
                 output_train_dir: Path,
                 max_samples: Optional[int] = None):
        """初始化训练管理器

        Args:
            input_preprocess_dir: 预处理数据目录路径，包含preprocess_data.npz文件
            output_train_dir: 训练结果的输出目录路径
            max_samples: 最大使用的样本数量，用于控制内存使用
        """
        self.input_preprocess_dir = input_preprocess_dir
        self.output_train_dir = output_train_dir
        self.max_samples = max_samples

        # 创建输出目录
        self.output_train_dir.mkdir(parents=True, exist_ok=True)

        # 初始化设备
        self.device = self._get_device()
        logger.info(f"使用设备: {self.device}")

        # 加载预处理数据
        self.preprocess_data = None
        self.sequences = None  # 4D序列 [batch, seq_len, 4]
        self.conditions = None  # 条件向量 [batch, cond_dim]
        self.behavior_ids = None
        self.behavior_mapping = None
        self.loss_rate_mapping_up = None
        self.loss_rate_mapping_down = None
        self.valid_loss_values_up = None
        self.valid_loss_values_down = None
        self.num_behaviors = None
        self.use_log_transform = False

        # 归一化参数
        self.delay1_scaler_center_ = None
        self.delay1_scaler_scale_ = None
        self.delay1_robust_scale_min = None
        self.delay1_robust_scale_max = None
        self.delay1_clipped_min = None
        self.delay1_clipped_max = None
        self.delay2_scaler_center_ = None
        self.delay2_scaler_scale_ = None
        self.delay2_robust_scale_min = None
        self.delay2_robust_scale_max = None
        self.delay2_clipped_min = None
        self.delay2_clipped_max = None
        self.normalization_method = "robust"

        # 归一化器
        self.normalizer = None

        # 模型和优化器
        self.model = None
        self.optimizer = None

        # 加载预处理数据
        self._load_preprocess_data()

        # 初始化归一化器
        self._init_normalizer()

    def _get_device(self) -> torch.device:
        """获取可用的计算设备

        Returns:
            可用的计算设备（MPS/CUDA/CPU）
        """
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _load_preprocess_data(self):
        """加载预处理数据"""
        logger.info(f"正在加载预处理数据: {self.input_preprocess_dir}")

        preprocess_file = self.input_preprocess_dir / "preprocess_data.npz"
        if not preprocess_file.exists():
            logger.error(f"预处理文件 {preprocess_file} 不存在！")
            sys.exit(1)

        self.preprocess_data = np.load(preprocess_file, allow_pickle=True)
        self.sequences = self.preprocess_data["sequences"]  # 4D序列 [batch, seq_len, 4]
        self.conditions = self.preprocess_data["conditions"]  # 条件向量 [batch, cond_dim]
        self.behavior_ids = self.preprocess_data["behavior_ids"]
        self.loss_rate_mapping_up = self.preprocess_data["loss_rate_mapping_up"].item()
        self.loss_rate_mapping_down = self.preprocess_data["loss_rate_mapping_down"].item()
        self.valid_loss_values_up = self.preprocess_data["valid_loss_values_up"].tolist()
        self.valid_loss_values_down = self.preprocess_data["valid_loss_values_down"].tolist()
        self.num_behaviors = self.preprocess_data["num_behaviors"].item()

        # 检查是否使用对数变换
        self.use_log_transform = self.preprocess_data.get("delay1_use_log_transform", False)
        if isinstance(self.use_log_transform, np.ndarray):
            self.use_log_transform = self.use_log_transform.item()
        logger.info(f"是否使用对数变换: {self.use_log_transform}")

        # 加载robust归一化参数（上下行独立）
        self.delay1_scaler_center_ = self.preprocess_data["delay1_scaler_center_"]
        self.delay1_scaler_scale_ = self.preprocess_data["delay1_scaler_scale_"]
        self.delay1_robust_scale_min = self.preprocess_data["delay1_robust_scale_min"]
        self.delay1_robust_scale_max = self.preprocess_data["delay1_robust_scale_max"]
        self.delay1_clipped_min = self.preprocess_data["delay1_clipped_min"]
        self.delay1_clipped_max = self.preprocess_data["delay1_clipped_max"]

        self.delay2_scaler_center_ = self.preprocess_data["delay2_scaler_center_"]
        self.delay2_scaler_scale_ = self.preprocess_data["delay2_scaler_scale_"]
        self.delay2_robust_scale_min = self.preprocess_data["delay2_robust_scale_min"]
        self.delay2_robust_scale_max = self.preprocess_data["delay2_robust_scale_max"]
        self.delay2_clipped_min = self.preprocess_data["delay2_clipped_min"]
        self.delay2_clipped_max = self.preprocess_data["delay2_clipped_max"]

        # 限制样本数量，防止内存不足
        if self.max_samples is not None and len(self.sequences) > self.max_samples:
            logger.info(f"样本数量过大 ({len(self.sequences)}), 将使用前 {self.max_samples} 个样本进行训练")
            self.sequences = self.sequences[:self.max_samples]
            self.conditions = self.conditions[:self.max_samples]
            self.behavior_ids = self.behavior_ids[:self.max_samples]

    def _init_normalizer(self):
        """初始化归一化器"""
        logger.info("初始化归一化器")
        self.normalizer = Normalizer(self.valid_loss_values_up, self.valid_loss_values_down)

        # 构建归一化参数字典，支持上下行独立参数
        normalization_params = {
            'delay_up': {
                'normalization_method': 'robust',
                'delay_scaler_center_': self.delay1_scaler_center_,
                'delay_scaler_scale_': self.delay1_scaler_scale_,
                'robust_scale_min': self.delay1_robust_scale_min,
                'robust_scale_max': self.delay1_robust_scale_max,
                'clipped_min': self.delay1_clipped_min,
                'clipped_max': self.delay1_clipped_max,
                'original_min': self.preprocess_data.get('original_min', 0.0),
                'use_log_transform': self.use_log_transform
            },
            'delay_down': {
                'normalization_method': 'robust',
                'delay_scaler_center_': self.delay2_scaler_center_,
                'delay_scaler_scale_': self.delay2_scaler_scale_,
                'robust_scale_min': self.delay2_robust_scale_min,
                'robust_scale_max': self.delay2_robust_scale_max,
                'clipped_min': self.delay2_clipped_min,
                'clipped_max': self.delay2_clipped_max,
                'original_min': self.preprocess_data.get('original_min', 0.0),
                'use_log_transform': self.use_log_transform
            },
            'loss_rate_up': {
                'loss_rate_mapping': self.loss_rate_mapping_up,
                'valid_loss_values': self.valid_loss_values_up
            },
            'loss_rate_down': {
                'loss_rate_mapping': self.loss_rate_mapping_down,
                'valid_loss_values': self.valid_loss_values_down
            }
        }
        self.normalizer.normalization_params = normalization_params

    def _normalize_features(self):
        """归一化特征数据（已废弃，数据在TrainingDataPreprocessor中已归一化）

        Returns:
            归一化后的特征数据
        """
        logger.warning("_normalize_features方法已废弃，数据在TrainingDataPreprocessor中已归一化，直接返回sequences数据")
        return self.sequences

    def _reshape_data(self, _features_norm):
        """重塑数据以适应批量训练

        Args:
            _features_norm: 归一化后的特征数据（已废弃，直接使用self.sequences）

        Returns:
            重塑后的序列张量、条件张量
        """
        # 从配置文件加载训练参数
        lr = 1e-5  # 降低学习率，避免梯度爆炸
        num_epochs = DEFAULT_EPOCHS  # 训练轮数
        batch_size = DEFAULT_BATCH_SIZE  # 批次大小

        # 使用原始数据中的序列长度，不需要重新计算
        batch_seq_len = self.sequences.shape[1]
        input_dim = self.sequences.shape[2]  # 4D输入

        # 打印超参数
        logger.info("\n训练超参数:")
        logger.info(f"学习率: {lr}")
        logger.info(f"训练轮数: {num_epochs}")
        logger.info(f"批次大小: {batch_size}")
        logger.info(f"每个批次序列长度: {batch_seq_len}")
        logger.info(f"输入维度: {input_dim}")

        # 直接使用预处理好的4D序列数据，不需要重塑
        # 原始数据形状: (num_sequences, seq_len, 4)
        num_sequences = len(self.sequences)
        num_batches = num_sequences // batch_size
        if num_sequences % batch_size != 0:
            num_batches += 1

        # 调整数据长度以适应批次大小
        adjusted_sequences = num_batches * batch_size
        if adjusted_sequences > num_sequences:
            # 补零到最近的批次边界
            pad_length = adjusted_sequences - num_sequences
            sequences_padded = np.pad(self.sequences, ((0, pad_length), (0, 0), (0, 0)), mode='constant')
            conditions_padded = np.pad(self.conditions, ((0, pad_length), (0, 0)), mode='constant')
            behavior_ids_padded = np.pad(self.behavior_ids, (0, pad_length), mode='constant')
        else:
            sequences_padded = self.sequences
            conditions_padded = self.conditions
            behavior_ids_padded = self.behavior_ids

        # 重塑为批次形状
        # 形状: (num_batches, batch_size, seq_len, 4)
        sequences_reshaped = sequences_padded.reshape(num_batches, batch_size, batch_seq_len, input_dim)
        # 条件向量形状: (num_batches, batch_size, cond_dim)
        conditions_reshaped = conditions_padded.reshape(num_batches, batch_size, -1)
        # 行为ID形状: (num_batches, batch_size)
        behavior_ids_reshaped = behavior_ids_padded.reshape(num_batches, batch_size)

        # 转换为张量并移动到设备
        sequences_tensor = torch.tensor(sequences_reshaped, dtype=torch.float32).to(self.device)
        conditions_tensor = torch.tensor(conditions_reshaped, dtype=torch.float32).to(self.device)
        behavior_ids_tensor = torch.tensor(behavior_ids_reshaped, dtype=torch.long).to(self.device)

        return sequences_tensor, conditions_tensor, behavior_ids_tensor

    def _init_model(self):
        """初始化模型和优化器"""
        logger.info("初始化条件扩散模型")

        # 构建归一化参数字典，用于模型保存，支持上下行独立参数
        normalization_params = {
            'delay_up': {
                'normalization_method': 'robust',
                'delay_scaler_center_': self.delay1_scaler_center_,
                'delay_scaler_scale_': self.delay1_scaler_scale_,
                'robust_scale_min': self.delay1_robust_scale_min,
                'robust_scale_max': self.delay1_robust_scale_max,
                'clipped_min': self.delay1_clipped_min,
                'clipped_max': self.delay1_clipped_max,
                'original_min': self.preprocess_data.get('original_min', 0.0),
                'use_log_transform': self.use_log_transform
            },
            'delay_down': {
                'normalization_method': 'robust',
                'delay_scaler_center_': self.delay2_scaler_center_,
                'delay_scaler_scale_': self.delay2_scaler_scale_,
                'robust_scale_min': self.delay2_robust_scale_min,
                'robust_scale_max': self.delay2_robust_scale_max,
                'clipped_min': self.delay2_clipped_min,
                'clipped_max': self.delay2_clipped_max,
                'original_min': self.preprocess_data.get('original_min', 0.0),
                'use_log_transform': self.use_log_transform
            },
            'loss_rate_up': {
                'loss_rate_mapping': self.loss_rate_mapping_up,
                'valid_loss_values': self.valid_loss_values_up
            },
            'loss_rate_down': {
                'loss_rate_mapping': self.loss_rate_mapping_down,
                'valid_loss_values': self.valid_loss_values_down
            }
        }

        # 初始化条件扩散模型，支持4D输入
        self.model = ConditionDiffusionModel(
            input_dim=4,  # 输入维度：[delay1, loss1, delay2, loss2]
            behavior_embed_dim=32,  # 行为嵌入维度
            T=1000,  # 扩散步数
            normalization_params=normalization_params
        )
        self.model.to(self.device)

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)

    def train(self):
        """执行模型训练

        Returns:
            最终模型的文件路径
        """
        logger.info("开始模型训练")

        # 归一化特征（已废弃，数据在TrainingDataPreprocessor中已归一化）
        features_norm = self._normalize_features()

        # 重塑数据，获取4D序列张量和条件向量张量
        sequences_tensor, conditions_tensor, behavior_ids_tensor = self._reshape_data(features_norm)

        # 初始化模型
        self._init_model()

        # 训练参数
        num_epochs = DEFAULT_EPOCHS
        gradient_clip_value = 1.0

        # 训练循环
        best_loss = float("inf")
        loss_history = []

        num_batches = sequences_tensor.shape[0]

        logger.info(f"开始在 {self.device} 上训练")
        logger.info(f"原始数据形状: ({sequences_tensor.shape[1] * sequences_tensor.shape[2] * num_batches}, {sequences_tensor.shape[3]})")
        logger.info(f"批次数据形状: {sequences_tensor.shape}")
        logger.info(f"条件向量形状: {conditions_tensor.shape}")
        logger.info(f"批次数量: {num_batches}")
        logger.info(f"批次大小: {sequences_tensor.shape[1]}")
        logger.info(f"每个批次序列长度: {sequences_tensor.shape[2]}")
        logger.info(f"输入维度: {sequences_tensor.shape[3]}")
        logger.info(f"行为数量: {self.num_behaviors}")

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0

            # 遍历所有批次
            for batch_idx in range(num_batches):
                self.optimizer.zero_grad()

                # 获取当前批次
                batch_sequences = sequences_tensor[batch_idx]
                batch_conditions = conditions_tensor[batch_idx]

                # 扩展条件向量到序列长度维度
                # 从 (batch_size, cond_dim) 扩展到 (batch_size, seq_len, cond_dim)
                batch_seq_len = batch_sequences.shape[1]
                batch_conditions_expanded = batch_conditions.unsqueeze(1).expand(-1, batch_seq_len, -1)

                # 计算损失，使用条件向量而不是行为ID
                loss = self.model.compute_loss(batch_sequences, condition_vector=batch_conditions_expanded)

                # 反向传播和优化
                loss.backward()

                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_value)

                self.optimizer.step()

                epoch_loss += loss.item()

            # 调试：打印反归一化的内容，每10个epoch打印一次
            if epoch % 10 == 0:
                # 取一个批次的数据进行反归一化测试
                batch_sequences_np = batch_sequences.cpu().numpy()[0]  # 取第一个样本

                # 反归一化，使用新的4D反归一化方法
                delay1, loss1, delay2, loss2 = self.normalizer.denormalize4d(
                    batch_sequences_np[:, 0], batch_sequences_np[:, 1],
                    batch_sequences_np[:, 2], batch_sequences_np[:, 3]
                )

                # 打印反归一化前后的数据统计
                logger.info(f"\n调试信息 - Epoch {epoch+1}:")
                logger.info(f"归一化延迟1 - Min: {np.min(batch_sequences_np[:, 0]):.4f}, Max: {np.max(batch_sequences_np[:, 0]):.4f}, Mean: {np.mean(batch_sequences_np[:, 0]):.4f}")
                logger.info(f"反归一化延迟1 - Min: {np.min(delay1):.4f}, Max: {np.max(delay1):.4f}, Mean: {np.mean(delay1):.4f}")
                logger.info(f"归一化丢包率1 - Min: {np.min(batch_sequences_np[:, 1]):.4f}, Max: {np.max(batch_sequences_np[:, 1]):.4f}, Mean: {np.mean(batch_sequences_np[:, 1]):.4f}")
                logger.info(f"反归一化丢包率1 - Min: {np.min(loss1):.4f}, Max: {np.max(loss1):.4f}, Mean: {np.mean(loss1):.4f}")
                logger.info(f"归一化延迟2 - Min: {np.min(batch_sequences_np[:, 2]):.4f}, Max: {np.max(batch_sequences_np[:, 2]):.4f}, Mean: {np.mean(batch_sequences_np[:, 2]):.4f}")
                logger.info(f"反归一化延迟2 - Min: {np.min(delay2):.4f}, Max: {np.max(delay2):.4f}, Mean: {np.mean(delay2):.4f}")
                logger.info(f"归一化丢包率2 - Min: {np.min(batch_sequences_np[:, 3]):.4f}, Max: {np.max(batch_sequences_np[:, 3]):.4f}, Mean: {np.mean(batch_sequences_np[:, 3]):.4f}")
                logger.info(f"反归一化丢包率2 - Min: {np.min(loss2):.4f}, Max: {np.max(loss2):.4f}, Mean: {np.mean(loss2):.4f}")

            # 计算平均损失
            avg_epoch_loss = epoch_loss / num_batches
            loss_history.append(avg_epoch_loss)

            logger.info(f"训练轮次 [{epoch + 1}/{num_epochs}], 平均损失值: {avg_epoch_loss:.4f}")

            # 保存最佳模型
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_path = self.output_train_dir / "diffusion_model_best.pth"
                self._save_model(best_model_path, epoch, best_loss)

        # 保存最终模型
        final_model_path = self.output_train_dir / "diffusion_model_final.pth"
        self._save_model(final_model_path, num_epochs, loss.item())

        # 保存训练历史
        loss_history_path = self.output_train_dir / "loss_history.npy"
        np.save(loss_history_path, np.array(loss_history))

        logger.info("训练完成！")
        logger.info(f"最佳损失值: {best_loss:.4f}")
        logger.info(f"最终模型已保存到: {final_model_path}")
        logger.info(f"训练历史已保存到: {loss_history_path}")

        return final_model_path

    def _save_model(self, model_path: Path, epoch: int, loss: float):
        """保存模型检查点

        Args:
            model_path: 模型保存路径
            epoch: 当前训练轮次
            loss: 当前损失值
        """
        logger.info(f"保存模型到: {model_path}")

        # 构建模型检查点，支持上下行独立参数
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "normalization_method": self.normalization_method,
            "use_log_transform": self.use_log_transform,

            # 上下行独立的合法丢包值
            "valid_loss_values_up": self.valid_loss_values_up,
            "valid_loss_values_down": self.valid_loss_values_down,

            # 上下行独立的丢包率映射
            "loss_rate_mapping_up": self.loss_rate_mapping_up,
            "loss_rate_mapping_down": self.loss_rate_mapping_down,

            # 上行归一化参数
            "delay1_scaler_center_": self.delay1_scaler_center_,
            "delay1_scaler_scale_": self.delay1_scaler_scale_,
            "delay1_robust_scale_min": self.delay1_robust_scale_min,
            "delay1_robust_scale_max": self.delay1_robust_scale_max,
            "delay1_clipped_min": self.delay1_clipped_min,
            "delay1_clipped_max": self.delay1_clipped_max,

            # 下行归一化参数
            "delay2_scaler_center_": self.delay2_scaler_center_,
            "delay2_scaler_scale_": self.delay2_scaler_scale_,
            "delay2_robust_scale_min": self.delay2_robust_scale_min,
            "delay2_robust_scale_max": self.delay2_robust_scale_max,
            "delay2_clipped_min": self.delay2_clipped_min,
            "delay2_clipped_max": self.delay2_clipped_max,
        }

        # 保存模型
        torch.save(checkpoint, model_path)
