#!/usr/bin/env python3
"""
归一化处理模块
提供统一的归一化和反归一化功能
"""

import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from network_simulation.utils.logger import get_logger

# 获取日志记录器
logger = get_logger(__name__)


class Normalizer:
    """
    归一化处理类
    提供统一的归一化和反归一化功能
    """

    def __init__(self, valid_loss_up: list = None, valid_loss_down: list = None):
        """
        初始化归一化器

        Args:
            valid_loss_up: 上行合法丢包值列表，默认为[0.0, 1/3, 0.5, 2/3, 1.0]
            valid_loss_down: 下行合法丢包值列表，默认为[0.0, 1/3, 0.5, 2/3, 1.0]
        """
        self.valid_loss_up = valid_loss_up or [0.0, 1/3, 0.5, 2/3, 1.0]
        self.valid_loss_down = valid_loss_down or [0.0, 1/3, 0.5, 2/3, 1.0]

        # 扩展归一化参数结构，支持上下行独立参数
        self.normalization_params = {
            'delay_up': {},
            'delay_down': {},
            'loss_rate_up': {},
            'loss_rate_down': {}
        }
        logger.info(f"初始化归一化器，上行合法丢包值: {self.valid_loss_up}, 下行合法丢包值: {self.valid_loss_down}")

    def normalize4d(self, delay1_values: np.ndarray, loss1_values: np.ndarray, delay2_values: np.ndarray, loss2_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        对上下行延迟和丢包率数据进行归一化处理（双流）

        该方法用于将原始的上下行网络数据（延迟和丢包率）归一化到模型训练所需的范围。
        上行数据和下行数据分别使用独立的归一化参数，确保归一化的准确性。

        Args:
            delay1_values: 上行原始延迟数据，shape (n_samples,)
            loss1_values: 上行原始丢包率数据，shape (n_samples,)
            delay2_values: 下行原始延迟数据，shape (n_samples,)
            loss2_values: 下行原始丢包率数据，shape (n_samples,)

        Returns:
            tuple: 包含四个元素的元组
                - 归一化后的上行延迟数据，shape (n_samples,)
                - 归一化后的上行丢包率数据，shape (n_samples,)
                - 归一化后的下行延迟数据，shape (n_samples,)
                - 归一化后的下行丢包率数据，shape (n_samples,)
        """
        logger.info(f"开始归一化双流数据，延迟1形状: {delay1_values.shape}, 丢包率1形状: {loss1_values.shape}, 延迟2形状: {delay2_values.shape}, 丢包率2形状: {loss2_values.shape}")

        # 归一化上行数据
        delay1_norm, delay1_params = self.normalize_delay(delay1_values, use_log_transform=True, direction='up')
        loss1_norm, loss1_params = self.normalize_loss_rate(loss1_values, self.valid_loss_up, direction='up')
        self.normalization_params['delay_up'] = delay1_params
        self.normalization_params['loss_rate_up'] = loss1_params

        # 归一化下行数据
        delay2_norm, delay2_params = self.normalize_delay(delay2_values, use_log_transform=True, direction='down')
        loss2_norm, loss2_params = self.normalize_loss_rate(loss2_values, self.valid_loss_down, direction='down')
        self.normalization_params['delay_down'] = delay2_params
        self.normalization_params['loss_rate_down'] = loss2_params

        return delay1_norm, loss1_norm, delay2_norm, loss2_norm

    def denormalize4d(self, delay1_norm: np.ndarray, loss1_norm: np.ndarray, delay2_norm: np.ndarray, loss2_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        对归一化后的上下行延迟和丢包率数据进行反归一化处理（双流）

        Args:
            delay1_norm: 归一化后的上行延迟数据，shape (n_samples,)
            loss1_norm: 归一化后的上行丢包率数据，shape (n_samples,)
            delay2_norm: 归一化后的下行延迟数据，shape (n_samples,)
            loss2_norm: 归一化后的下行丢包率数据，shape (n_samples,)

        Returns:
            tuple: 包含四个元素的元组
                - 反归一化后的上行延迟数据，shape (n_samples,)
                - 反归一化后的上行丢包率数据，shape (n_samples,)
                - 反归一化后的下行延迟数据，shape (n_samples,)
                - 反归一化后的下行丢包率数据，shape (n_samples,)
        """
        logger.info(f"开始反归一化双流数据，延迟1形状: {delay1_norm.shape}, 丢包率1形状: {loss1_norm.shape}, 延迟2形状: {delay2_norm.shape}, 丢包率2形状: {loss2_norm.shape}")

        # 反归一化上行数据
        delay1 = self.denormalize_delay(delay1_norm, self.normalization_params['delay_up'])
        loss1 = self.denormalize_loss_rate(loss1_norm, self.normalization_params['loss_rate_up'])

        # 反归一化下行数据
        delay2 = self.denormalize_delay(delay2_norm, self.normalization_params['delay_down'])
        loss2 = self.denormalize_loss_rate(loss2_norm, self.normalization_params['loss_rate_down'])

        return delay1, loss1, delay2, loss2

    def save_params(self, filepath: Path or str):
        """
        保存归一化参数到文件

        Args:
            filepath: 保存路径
        """
        logger.info(f"保存归一化参数到: {filepath}")

        # 转换numpy数组为列表，以便JSON序列化
        params_to_save = {}
        for key, params in self.normalization_params.items():
            params_to_save[key] = {}
            for param_key, param_value in params.items():
                if isinstance(param_value, np.ndarray):
                    params_to_save[key][param_key] = param_value.tolist()
                else:
                    params_to_save[key][param_key] = param_value

        # 保存到JSON文件
        with open(filepath, 'w') as f:
            json.dump(params_to_save, f, indent=2)

        logger.info(f"归一化参数已保存到: {filepath}")

    def load_params(self, filepath: Path or str):
        """
        从文件加载归一化参数

        Args:
            filepath: 加载路径
        """
        logger.info(f"从文件加载归一化参数: {filepath}")

        # 从JSON文件加载
        with open(filepath, 'r') as f:
            params_loaded = json.load(f)

        # 转换列表为numpy数组
        for key, params in params_loaded.items():
            for param_key, param_value in params.items():
                if isinstance(param_value, list):
                    params_loaded[key][param_key] = np.array(param_value)
                else:
                    params_loaded[key][param_key] = param_value

        self.normalization_params = params_loaded
        logger.info(f"归一化参数已加载: {filepath}")

    def from_checkpoint(self, checkpoint: dict):
        """
        从模型检查点加载归一化参数

        Args:
            checkpoint: 模型检查点字典
        """
        logger.info("从模型检查点加载归一化参数")

        # 检查是否有上下行独立的参数
        has_separate_params = 'delay_up_scaler_center_' in checkpoint

        if has_separate_params:
            # 加载上行延迟归一化参数
            delay_up_params = {
                'normalization_method': checkpoint.get('normalization_method', 'robust'),
                'delay_scaler_center_': checkpoint.get('delay_up_scaler_center_', np.array([0.0])),
                'delay_scaler_scale_': checkpoint.get('delay_up_scaler_scale_', np.array([1.0])),
                'robust_scale_min': checkpoint.get('delay_up_robust_scale_min', 0.0),
                'robust_scale_max': checkpoint.get('delay_up_robust_scale_max', 0.0),
                'clipped_min': checkpoint.get('delay_up_clipped_min', -1.0),
                'clipped_max': checkpoint.get('delay_up_clipped_max', 1.0),
                'original_min': checkpoint.get('original_min', 0.0),
                'use_log_transform': checkpoint.get('use_log_transform', False)
            }
            self.normalization_params['delay_up'] = delay_up_params

            # 加载下行延迟归一化参数
            delay_down_params = {
                'normalization_method': checkpoint.get('normalization_method', 'robust'),
                'delay_scaler_center_': checkpoint.get('delay_down_scaler_center_', np.array([0.0])),
                'delay_scaler_scale_': checkpoint.get('delay_down_scaler_scale_', np.array([1.0])),
                'robust_scale_min': checkpoint.get('delay_down_robust_scale_min', 0.0),
                'robust_scale_max': checkpoint.get('delay_down_robust_scale_max', 0.0),
                'clipped_min': checkpoint.get('delay_down_clipped_min', -1.0),
                'clipped_max': checkpoint.get('delay_down_clipped_max', 1.0),
                'original_min': checkpoint.get('original_min', 0.0),
                'use_log_transform': checkpoint.get('use_log_transform', False)
            }
            self.normalization_params['delay_down'] = delay_down_params

            # 加载上行丢包率归一化参数
            valid_loss_up = checkpoint.get('valid_loss_values_up', self.valid_loss_up)
            loss_up_params = {
                'loss_rate_mapping': checkpoint.get('loss_rate_mapping_up', {}),
                'valid_loss_values': valid_loss_up
            }
            self.normalization_params['loss_rate_up'] = loss_up_params

            # 加载下行丢包率归一化参数
            valid_loss_down = checkpoint.get('valid_loss_values_down', self.valid_loss_down)
            loss_down_params = {
                'loss_rate_mapping': checkpoint.get('loss_rate_mapping_down', {}),
                'valid_loss_values': valid_loss_down
            }
            self.normalization_params['loss_rate_down'] = loss_down_params

            # 更新实例的valid_loss值
            self.valid_loss_up = valid_loss_up
            self.valid_loss_down = valid_loss_down
        else:
            # 兼容旧版本，将单流参数复制到上下行
            delay_params = {
                'normalization_method': checkpoint.get('normalization_method', 'robust'),
                'delay_scaler_center_': checkpoint.get('delay_scaler_center_', np.array([0.0])),
                'delay_scaler_scale_': checkpoint.get('delay_scaler_scale_', np.array([1.0])),
                'robust_scale_min': checkpoint.get('robust_scale_min', 0.0),
                'robust_scale_max': checkpoint.get('robust_scale_max', 0.0),
                'clipped_min': checkpoint.get('clipped_min', -1.0),
                'clipped_max': checkpoint.get('clipped_max', 1.0),
                'original_min': checkpoint.get('original_min', 0.0),
                'use_log_transform': checkpoint.get('use_log_transform', False)
            }
            self.normalization_params['delay_up'] = delay_params
            self.normalization_params['delay_down'] = delay_params.copy()

            # 加载丢包率归一化参数
            valid_loss = checkpoint.get('valid_loss_values', self.valid_loss_up)
            loss_params = {
                'loss_rate_mapping': checkpoint.get('loss_rate_mapping', {}),
                'valid_loss_values': valid_loss
            }
            self.normalization_params['loss_rate_up'] = loss_params
            self.normalization_params['loss_rate_down'] = loss_params.copy()

            # 更新实例的valid_loss值
            self.valid_loss_up = valid_loss
            self.valid_loss_down = valid_loss

        logger.info("归一化参数已从检查点加载")

    def to_checkpoint(self) -> dict:
        """
        将归一化参数转换为模型检查点格式

        Returns:
            dict: 归一化参数字典，可直接用于模型检查点
        """
        logger.info("将归一化参数转换为模型检查点格式")

        # 获取上下行延迟参数
        delay_up_params = self.normalization_params.get('delay_up', {})
        delay_down_params = self.normalization_params.get('delay_down', {})
        loss_rate_up_params = self.normalization_params.get('loss_rate_up', {})
        loss_rate_down_params = self.normalization_params.get('loss_rate_down', {})

        checkpoint = {
            # 上行延迟归一化参数
            'delay_up_scaler_center_': delay_up_params.get('delay_scaler_center_', np.array([0.0])),
            'delay_up_scaler_scale_': delay_up_params.get('delay_scaler_scale_', np.array([1.0])),
            'delay_up_robust_scale_min': delay_up_params.get('robust_scale_min', 0.0),
            'delay_up_robust_scale_max': delay_up_params.get('robust_scale_max', 0.0),
            'delay_up_clipped_min': delay_up_params.get('clipped_min', -1.0),
            'delay_up_clipped_max': delay_up_params.get('clipped_max', 1.0),
            # 下行延迟归一化参数
            'delay_down_scaler_center_': delay_down_params.get('delay_scaler_center_', np.array([0.0])),
            'delay_down_scaler_scale_': delay_down_params.get('delay_scaler_scale_', np.array([1.0])),
            'delay_down_robust_scale_min': delay_down_params.get('robust_scale_min', 0.0),
            'delay_down_robust_scale_max': delay_down_params.get('robust_scale_max', 0.0),
            'delay_down_clipped_min': delay_down_params.get('clipped_min', -1.0),
            'delay_down_clipped_max': delay_down_params.get('clipped_max', 1.0),
            # 通用延迟参数
            'normalization_method': delay_up_params.get('normalization_method', 'robust'),
            'original_min': delay_up_params.get('original_min', 0.0),
            'use_log_transform': delay_up_params.get('use_log_transform', False),
            # 上行丢包率归一化参数
            'loss_rate_mapping_up': loss_rate_up_params.get('loss_rate_mapping', {}),
            'valid_loss_values_up': loss_rate_up_params.get('valid_loss_values', self.valid_loss_up),
            # 下行丢包率归一化参数
            'loss_rate_mapping_down': loss_rate_down_params.get('loss_rate_mapping', {}),
            'valid_loss_values_down': loss_rate_down_params.get('valid_loss_values', self.valid_loss_down)
        }

        logger.info("归一化参数已转换为检查点格式")
        return checkpoint

    def normalize_delay(self, delay_values: np.ndarray, use_log_transform: bool = True, direction: str = 'up') -> tuple[np.ndarray, dict]:
        """
        对延迟数据进行归一化处理

        Args:
            delay_values: 原始延迟数据，shape (n_samples,)
            use_log_transform: 是否使用对数变换，默认为True
            direction: 数据方向，'up' 表示上行，'down' 表示下行

        Returns:
            tuple: 包含两个元素的元组
                - 归一化后的延迟数据，shape (n_samples,)
                - 归一化参数字典，包含归一化方法、变换参数等
        """
        logger.info(f"开始归一化{direction}行延迟数据，输入形状: {delay_values.shape}, use_log_transform: {use_log_transform}")
        logger.info(f"原始延迟数据统计 - 最小值: {np.min(delay_values):.4f}, 最大值: {np.max(delay_values):.4f}, 平均值: {np.mean(delay_values):.4f}, 标准差: {np.std(delay_values):.4f}")

        # 保存原始数据的最小值，用于反归一化
        original_min = np.min(delay_values)

        # 复制原始数据，避免修改输入
        delay_values_copy = delay_values.copy()

        # 确保所有延迟值都大于等于0，避免对数变换时出现问题
        delay_values_copy = np.maximum(delay_values_copy, 0)
        logger.info(f"确保延迟值非负后 - 最小值: {np.min(delay_values_copy):.4f}, 最大值: {np.max(delay_values_copy):.4f}, 平均值: {np.mean(delay_values_copy):.4f}, 标准差: {np.std(delay_values_copy):.4f}")

        # 添加对数变换，减少数据偏斜
        if use_log_transform:
            # 加1避免log(0)
            delay_values_copy = np.log(delay_values_copy + 1)
            logger.info(f"对数变换后 - 最小值: {np.min(delay_values_copy):.4f}, 最大值: {np.max(delay_values_copy):.4f}, 平均值: {np.mean(delay_values_copy):.4f}, 标准差: {np.std(delay_values_copy):.4f}")

        # RobustScaler归一化
        # 使用对数变换后的数据进行RobustScaler归一化
        # 使用合理的分位数范围(25, 75)，只考虑中间50%的数据，避免极端值影响
        delay_scaler = RobustScaler(quantile_range=(25, 75))  # 使用标准四分位距范围
        delay_scaled = delay_scaler.fit_transform(delay_values_copy.reshape(-1, 1)).flatten()

        logger.info(f"RobustScaler处理后 - 最小值: {np.min(delay_scaled):.4f}, 最大值: {np.max(delay_scaled):.4f}, 平均值: {np.mean(delay_scaled):.4f}, 标准差: {np.std(delay_scaled):.4f}")
        logger.info(f"RobustScaler参数 - 中心值: {delay_scaler.center_}, 缩放因子: {delay_scaler.scale_}")

        # 计算当前归一化后的最小值和最大值
        current_min = np.min(delay_scaled)
        current_max = np.max(delay_scaled)

        logger.info(f"RobustScaler输出范围 - 最小值: {current_min:.4f}, 最大值: {current_max:.4f}")

        # 使用min-max缩放将RobustScaler输出缩放到[-1, 1]范围
        if current_max > current_min:
            # 将RobustScaler输出缩放到[-1, 1]范围
            delay_norm = 2 * (delay_scaled - current_min) / (current_max - current_min) - 1

            # 确保所有值在[-1, 1]范围内
            delay_norm = np.clip(delay_norm, -1, 1)

            # 更新clipped范围为实际裁剪后的范围
            current_min_clipped = np.min(delay_norm)
            current_max_clipped = np.max(delay_norm)

            logger.info(f"使用min-max缩放归一化 - 原始范围: [{current_min:.4f}, {current_max:.4f}], 缩放到[-1, 1]")
        else:
            delay_norm = np.zeros_like(delay_scaled)
            # 设置默认的clipped值
            current_min_clipped = current_min
            current_max_clipped = current_max

        logger.info(f"最终归一化后 - 最小值: {np.min(delay_norm):.4f}, 最大值: {np.max(delay_norm):.4f}, 平均值: {np.mean(delay_norm):.4f}, 标准差: {np.std(delay_norm):.4f}")

        # 保存归一化参数，包括clipped范围
        normalization_params = {
            'normalization_method': 'robust',
            'delay_scaler_center_': delay_scaler.center_,
            'delay_scaler_scale_': delay_scaler.scale_,
            'robust_scale_min': current_min,
            'robust_scale_max': current_max,
            'clipped_min': current_min_clipped,  # 保存截断后的最小值
            'clipped_max': current_max_clipped,  # 保存截断后的最大值
            'original_min': original_min,
            'use_log_transform': use_log_transform
        }

        # 添加详细日志，用于调试
        logger.info("归一化后数据范围检查:")
        logger.info(f"  截断前RobustScaler输出范围: [{current_min:.6f}, {current_max:.6f}]")
        logger.info(f"  截断后范围: [{current_min_clipped:.6f}, {current_max_clipped:.6f}]")
        logger.info(f"  最终归一化范围: [{np.min(delay_norm):.6f}, {np.max(delay_norm):.6f}]")
        logger.info(f"  归一化后数据平均值: {np.mean(delay_norm):.6f}")
        logger.info(f"  归一化后数据标准差: {np.std(delay_norm):.6f}")

        logger.info(f"归一化参数: {normalization_params}")

        return delay_norm, normalization_params

    def denormalize_delay(self, delay_norm: np.ndarray, normalization_params: dict) -> np.ndarray:
        """
        对延迟数据进行反归一化处理

        Args:
            delay_norm: 归一化后的延迟数据，shape (n_samples,)
            normalization_params: 归一化参数，包含归一化方法、变换参数等

        Returns:
            np.ndarray: 反归一化后的延迟数据，shape (n_samples,)
        """
        logger.info(f"开始反归一化延迟数据，输入形状: {delay_norm.shape}")
        logger.info(f"归一化延迟数据统计 - 最小值: {np.min(delay_norm):.4f}, 最大值: {np.max(delay_norm):.4f}, 平均值: {np.mean(delay_norm):.4f}, 标准差: {np.std(delay_norm):.4f}")
        logger.info(f"反归一化参数: {normalization_params}")

        # 获取原始数据最小值，用于确保延迟值非负
        original_min = normalization_params.get('original_min', 0.0)

        # 检查是否使用对数变换
        use_log_transform = normalization_params.get('use_log_transform', False)

        # RobustScaler反归一化
        logger.info("使用RobustScaler反归一化方法")

        # 从参数中提取值，添加安全检查
        delay_scaler_center_ = normalization_params.get('delay_scaler_center_', np.array([0.0]))
        delay_scaler_scale_ = normalization_params.get('delay_scaler_scale_', np.array([1.0]))
        robust_scale_min = normalization_params.get('robust_scale_min', -1.0)
        robust_scale_max = normalization_params.get('robust_scale_max', 1.0)

        # 1. 先将模型输出的[-1, 1]范围转换回RobustScaler的输出范围
        # 从参数中提取截断范围
        clipped_min = normalization_params.get('clipped_min', robust_scale_min)
        clipped_max = normalization_params.get('clipped_max', robust_scale_max)

        logger.info(f"使用范围进行反变换: [{clipped_min:.4f}, {clipped_max:.4f}]")

        if robust_scale_max > robust_scale_min:
            # 转换回RobustScaler的输出范围
            delay_scaled = ((delay_norm + 1) / 2) * (robust_scale_max - robust_scale_min) + robust_scale_min
        else:
            delay_scaled = delay_norm

        logger.info(f"转换回RobustScaler范围后 - 最小值: {np.min(delay_scaled):.4f}, 最大值: {np.max(delay_scaled):.4f}, 平均值: {np.mean(delay_scaled):.4f}, 标准差: {np.std(delay_scaled):.4f}")

        # 2. 然后应用RobustScaler的反变换
        # RobustScaler的反变换公式：原始值 = (归一化值 * 四分位距) + 中位数
        delay = (delay_scaled * delay_scaler_scale_) + delay_scaler_center_

        logger.info(f"RobustScaler反变换后 - 最小值: {np.min(delay):.4f}, 最大值: {np.max(delay):.4f}, 平均值: {np.mean(delay):.4f}, 标准差: {np.std(delay):.4f}")

        # 应用指数变换，恢复原始值
        if use_log_transform:
            delay = np.exp(delay) - 1  # 指数变换，减1恢复原始值
            logger.info(f"指数变换后 - 最小值: {np.min(delay):.4f}, 最大值: {np.max(delay):.4f}, 平均值: {np.mean(delay):.4f}, 标准差: {np.std(delay):.4f}")

        # 确保延迟值非负，符合物理意义
        # 使用原始数据的最小值作为下限，确保生成的数据与原始数据在同一范围内
        delay = np.clip(delay, a_min=original_min, a_max=None)

        logger.info(f"最终反归一化后 - 最小值: {np.min(delay):.4f}, 最大值: {np.max(delay):.4f}, 平均值: {np.mean(delay):.4f}, 标准差: {np.std(delay):.4f}")

        return delay

    def normalize_loss_rate(self, loss_rate_values: np.ndarray, valid_loss_values: list, direction: str = 'up') -> tuple[np.ndarray, dict]:
        """
        对丢包率数据进行归一化处理

        Args:
            loss_rate_values: 原始丢包率数据，shape (n_samples,)
            valid_loss_values: 合法丢包值列表，包含所有可能的丢包率值
            direction: 数据方向，'up' 表示上行，'down' 表示下行

        Returns:
            tuple: 包含两个元素的元组
                - 归一化后的丢包率数据，shape (n_samples,)
                - 归一化参数字典，包含丢包率映射等信息
        """
        logger.info(f"开始归一化{direction}行丢包率数据，输入形状: {loss_rate_values.shape}")
        logger.info(f"原始丢包率数据统计 - 最小值: {np.min(loss_rate_values):.4f}, 最大值: {np.max(loss_rate_values):.4f}, 平均值: {np.mean(loss_rate_values):.4f}, 标准差: {np.std(loss_rate_values):.4f}")
        logger.info(f"合法丢包值: {valid_loss_values}")

        # 构建loss_rate到index的映射
        loss_rate_mapping = {value: idx for idx, value in enumerate(valid_loss_values)}
        N_vals = len(valid_loss_values)

        logger.info(f"丢包率映射: {loss_rate_mapping}")

        # 处理不在映射中的丢包率值，将它们映射到最接近的合法值
        def get_loss_rate_index(lr):
            if lr in loss_rate_mapping:
                return loss_rate_mapping[lr]
            else:
                closest_value = min(valid_loss_values, key=lambda x: abs(x - lr))
                return loss_rate_mapping[closest_value]

        # 执行序数编码
        loss_rate_indices = np.array([get_loss_rate_index(lr) for lr in loss_rate_values])

        logger.info(f"丢包率序数编码后 - 唯一值: {np.unique(loss_rate_indices)}, 统计: {np.bincount(loss_rate_indices)}")

        # Min-Max缩放到[-1, 1]范围
        if N_vals > 1:
            loss_norm = (loss_rate_indices / (N_vals - 1)) * 2 - 1

            # 检查归一化后的丢包率是否有足够的变化
            unique_norm_loss = np.unique(loss_norm)
            logger.info(f"Min-Max缩放后 - 最小值: {np.min(loss_norm):.4f}, 最大值: {np.max(loss_norm):.4f}, 平均值: {np.mean(loss_norm):.4f}, 标准差: {np.std(loss_norm):.4f}")
            logger.info(f"归一化丢包率唯一值数量: {len(unique_norm_loss)}")

            if len(unique_norm_loss) == 1:
                # 添加均匀分布噪声，范围为[-0.1, 0.1]
                noise = np.random.uniform(-0.1, 0.1, size=loss_norm.shape)
                loss_norm = loss_norm + noise
                # 确保仍在[-1, 1]范围内
                loss_norm = np.clip(loss_norm, -1, 1)
                logger.info(f"添加噪声后 - 最小值: {np.min(loss_norm):.4f}, 最大值: {np.max(loss_norm):.4f}, 平均值: {np.mean(loss_norm):.4f}, 标准差: {np.std(loss_norm):.4f}")
        else:
            # 当只有一个有效丢包值时，直接设置为0
            loss_norm = np.zeros_like(loss_rate_indices)
            logger.info("只有一个有效丢包值，归一化结果为全零")

        # 保存归一化参数
        normalization_params = {
            'loss_rate_mapping': loss_rate_mapping,
            'valid_loss_values': valid_loss_values
        }

        logger.info(f"归一化参数: {normalization_params}")

        return loss_norm, normalization_params

    def denormalize_loss_rate(self, loss_norm: np.ndarray, normalization_params: dict) -> np.ndarray:
        """
        对丢包率数据进行反归一化处理

        Args:
            loss_norm: 归一化后的丢包率数据，shape (n_samples,)
            normalization_params: 归一化参数，包含丢包率映射和合法丢包值列表

        Returns:
            np.ndarray: 反归一化后的丢包率数据，shape (n_samples,)
        """
        logger.info(f"开始反归一化丢包率数据，输入形状: {loss_norm.shape}")
        logger.info(f"归一化丢包率数据统计 - 最小值: {np.min(loss_norm):.4f}, 最大值: {np.max(loss_norm):.4f}, 平均值: {np.mean(loss_norm):.4f}, 标准差: {np.std(loss_norm):.4f}")
        logger.info(f"反归一化参数: {normalization_params}")

        # 从参数中提取值，添加安全检查
        valid_loss_values = normalization_params.get('valid_loss_values', [0.0, 1/3, 0.5, 2/3, 1.0])

        logger.info(f"合法丢包值: {valid_loss_values}")

        # 将[-1, 1]映射到[0, len(valid_loss_values)-1]
        idx_float = (loss_norm + 1) / 2 * (len(valid_loss_values) - 1)
        # 四舍五入并裁剪到有效范围
        idx = np.round(idx_float).astype(int)
        idx = np.clip(idx, a_min=0, a_max=len(valid_loss_values) - 1)

        logger.info(f"映射索引后 - 唯一值: {np.unique(idx)}, 统计: {np.bincount(idx)}")

        # 映射到合法值
        loss_rate = np.array(valid_loss_values)[idx]

        logger.info(f"反归一化后丢包率统计 - 最小值: {np.min(loss_rate):.4f}, 最大值: {np.max(loss_rate):.4f}, 平均值: {np.mean(loss_rate):.4f}, 标准差: {np.std(loss_rate):.4f}")

        return loss_rate


# 保留原有函数作为兼容接口
def normalize_delay(delay_values: np.ndarray, use_log_transform: bool = True) -> tuple[np.ndarray, dict]:
    """
    对延迟数据进行归一化处理（兼容旧接口）

    Args:
        delay_values: 原始延迟数据，shape (n_samples,)
        use_log_transform: 是否使用对数变换，默认为True

    Returns:
        tuple: 包含两个元素的元组
            - 归一化后的延迟数据，shape (n_samples,)
            - 归一化参数字典，包含归一化方法、变换参数等
    """
    normalizer = Normalizer()
    return normalizer.normalize_delay(delay_values, use_log_transform)


def denormalize_delay(delay_norm: np.ndarray, normalization_params: dict) -> np.ndarray:
    """
    对延迟数据进行反归一化处理（兼容旧接口）

    Args:
        delay_norm: 归一化后的延迟数据，shape (n_samples,)
        normalization_params: 归一化参数，包含归一化方法、变换参数等

    Returns:
        np.ndarray: 反归一化后的延迟数据，shape (n_samples,)
    """
    normalizer = Normalizer()
    return normalizer.denormalize_delay(delay_norm, normalization_params)


def normalize_loss_rate(loss_rate_values: np.ndarray, valid_loss_values: list) -> tuple[np.ndarray, dict]:
    """
    对丢包率数据进行归一化处理（兼容旧接口）

    Args:
        loss_rate_values: 原始丢包率数据，shape (n_samples,)
        valid_loss_values: 合法丢包值列表，包含所有可能的丢包率值

    Returns:
        tuple: 包含两个元素的元组
            - 归一化后的丢包率数据，shape (n_samples,)
            - 归一化参数字典，包含丢包率映射等信息
    """
    normalizer = Normalizer(valid_loss_values)
    return normalizer.normalize_loss_rate(loss_rate_values, valid_loss_values)


def denormalize_loss_rate(loss_norm: np.ndarray, normalization_params: dict) -> np.ndarray:
    """
    对丢包率数据进行反归一化处理（兼容旧接口）

    Args:
        loss_norm: 归一化后的丢包率数据，shape (n_samples,)
        normalization_params: 归一化参数，包含丢包率映射和合法丢包值列表

    Returns:
        np.ndarray: 反归一化后的丢包率数据，shape (n_samples,)
    """
    normalizer = Normalizer()
    return normalizer.denormalize_loss_rate(loss_norm, normalization_params)
