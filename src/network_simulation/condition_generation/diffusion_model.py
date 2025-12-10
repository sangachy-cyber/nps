#!/usr/bin/env python3
"""
条件扩散模型实现
基于DDPM框架，支持行为条件的网络时序数据生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network_simulation.utils.logger import get_logger

# 初始化日志记录器
logger = get_logger(__name__)


class Time2Vec(nn.Module):
    """Time2Vec位置编码

    注意：输入时间步应归一化到[0,1]范围，避免cos/sin震荡剧烈影响训练稳定性。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        if d_model < 1:
            raise ValueError("d_model must be >= 1")
        # 第0维用cos，其余d_model-1维用sin
        # 使用合适的初始化值，改善梯度流动
        self.w0 = nn.Parameter(torch.randn(1) * 0.1)
        self.b0 = nn.Parameter(torch.randn(1) * 0.1)
        if d_model > 1:
            self.w = nn.Parameter(torch.randn(d_model - 1) * 0.1)
            self.b = nn.Parameter(torch.randn(d_model - 1) * 0.1)
        else:
            self.w = None
            self.b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入时间步张量，shape (batch_size, seq_len, 1)，应归一化到[0,1]范围

        Returns:
            位置编码后的张量，shape (batch_size, seq_len, d_model)
        """
        # 将输入转换为float类型以匹配参数类型
        x = x.float()

        # 检查输入是否在[0,1]范围，超出则记录警告
        if x.min() < 0 or x.max() > 1:
            logger.warning(
                f"Time2Vec输入不在[0,1]范围内，最小值: {x.min()}, 最大值: {x.max()}，这可能导致训练不稳定。"
            )

        batch_size, seq_len, _ = x.shape

        # 第0维：cos分量
        # x已归一化到[0,1]范围，避免cos震荡剧烈
        v0 = torch.cos(x * self.w0 + self.b0)  # (batch_size, seq_len, 1)

        if self.d_model == 1:
            return v0

        # 剩余维度：sin分量
        x_expanded = x.expand(
            batch_size, seq_len, self.d_model - 1
        )  # (batch_size, seq_len, d_model-1)
        # x已归一化到[0,1]范围，避免sin震荡剧烈
        v1 = torch.sin(x_expanded * self.w + self.b)  # (batch_size, seq_len, d_model-1)

        return torch.cat([v0, v1], dim=-1)

    def __repr__(self) -> str:
        return f"Time2Vec(d_model={self.d_model})"


class CausalDilatedConv(nn.Module):
    """因果膨胀卷积层"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        # 确保padding为非负值，避免后续切片操作出错
        assert self.padding >= 0, f"Padding must be non-negative, got {self.padding}"
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量，shape (batch_size, in_channels, seq_len)

        Returns:
            卷积后的张量，shape (batch_size, out_channels, seq_len)
        """
        out = self.conv(x)
        # 移除填充以保持因果结构，防止 padding=0 时空切片
        if self.padding > 0:
            return out[:, :, : -self.padding]
        else:
            return out

    def __repr__(self) -> str:
        return f"CausalDilatedConv(in_channels={self.conv.in_channels}, out_channels={self.conv.out_channels}, kernel_size={self.conv.kernel_size[0]}, dilation={self.conv.dilation[0]})"


class UNetBlock(nn.Module):
    """UNet块"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ):
        super().__init__()
        self.conv1 = CausalDilatedConv(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalDilatedConv(
            out_channels, out_channels, kernel_size, dilation
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        # Shortcut connection if in_channels != out_channels
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量，shape (batch_size, in_channels, seq_len)

        Returns:
            经过UNet块处理后的张量，shape (batch_size, out_channels, seq_len)
        """
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        return out + residual

    def __repr__(self) -> str:
        return f"UNetBlock(in_channels={self.conv1.conv.in_channels}, out_channels={self.conv1.conv.out_channels}, kernel_size={self.conv1.conv.kernel_size[0]}, dilation={self.conv1.conv.dilation[0]})"


class BehaviorEmbedding(nn.Module):
    """行为嵌入层"""

    def __init__(self, num_behaviors: int, d_embed: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(num_behaviors, d_embed, padding_idx=padding_idx)
        self.dropout = nn.Dropout(0.1)

    def forward(self, behavior_ids: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            behavior_ids: 行为ID序列，shape (batch_size, seq_len)

        Returns:
            行为嵌入向量，shape (batch_size, seq_len, d_embed)
        """
        return self.dropout(self.embedding(behavior_ids))

    def __repr__(self) -> str:
        return f"BehaviorEmbedding(num_behaviors={self.embedding.num_embeddings}, d_embed={self.embedding.embedding_dim}, padding_idx={self.embedding.padding_idx})"


class UNet(nn.Module):
    """简化版U-Net主干网络，使用因果膨胀卷积

    注意：这是一个简化版的U-Net，没有传统的编码器-解码器结构（下采样/上采样），
    而是堆叠的残差块结构，适合因果时序生成任务。
    """

    def __init__(self, input_dim: int, behavior_embed_dim: int, T: int = 1000):
        super().__init__()
        self.input_dim = input_dim
        self.behavior_embed_dim = behavior_embed_dim
        self.T = T

        # Time2Vec位置编码
        self.time2vec = Time2Vec(behavior_embed_dim)

        # 计算总输入维度：原始输入维度 + 行为嵌入维度 + 时间嵌入维度
        total_input_dim = input_dim + behavior_embed_dim + behavior_embed_dim

        # 简化的因果膨胀卷积网络，减少层数和通道数，提高训练稳定性
        # 添加输入投影层，避免信息瓶颈
        proj_out_dim = 64
        self.network = nn.Sequential(
            # 输入投影层，将总输入维度投影到固定维度64，避免信息丢失
            nn.Conv1d(total_input_dim, proj_out_dim, kernel_size=1),
            # 第一层：膨胀率1，提取低层特征
            UNetBlock(proj_out_dim, proj_out_dim, kernel_size=3, dilation=1),
            # 第二层：膨胀率2，扩大感受野
            UNetBlock(proj_out_dim, proj_out_dim, kernel_size=3, dilation=2),
            # 第三层：膨胀率4，进一步扩大感受野
            UNetBlock(proj_out_dim, proj_out_dim, kernel_size=3, dilation=4),
            # 第四层：膨胀率8，扩大感受野
            UNetBlock(proj_out_dim, proj_out_dim, kernel_size=3, dilation=8),
            # 第五层：膨胀率4，缩小感受野
            UNetBlock(proj_out_dim, proj_out_dim, kernel_size=3, dilation=4),
            # 第六层：膨胀率2，缩小感受野
            UNetBlock(proj_out_dim, proj_out_dim, kernel_size=3, dilation=2),
            # 第七层：膨胀率1，恢复细节
            UNetBlock(proj_out_dim, proj_out_dim, kernel_size=3, dilation=1),
            # 输出层：映射回输入维度，使用因果卷积确保因果性
            CausalDilatedConv(proj_out_dim, input_dim, kernel_size=3, dilation=1),
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, behavior_embed: torch.Tensor
    ) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量，shape (batch_size, seq_len, input_dim)
            t: 时间步张量，shape (batch_size, 1, 1) - 每个样本共享一个时间步
            behavior_embed: 行为嵌入向量，shape (batch_size, seq_len, behavior_embed_dim)

        Returns:
            网络输出张量，shape (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, input_dim = x.shape

        # 转换为通道优先格式
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)

        # 将行为嵌入转换为通道优先格式
        behavior_embed = behavior_embed.transpose(
            1, 2
        )  # (batch_size, behavior_embed_dim, seq_len)

        # 处理时间嵌入
        # t 已经是归一化后的值，先计算单个时间步的嵌入，再扩展到序列长度
        # shape (batch_size, 1, behavior_embed_dim)
        time_embed_single = self.time2vec(t)
        # 扩展到序列长度，shape (batch_size, seq_len, behavior_embed_dim)
        time_embed = time_embed_single.expand(batch_size, seq_len, -1)
        # 转换为通道优先格式，shape (batch_size, behavior_embed_dim, seq_len)
        time_embed = time_embed.transpose(1, 2)

        # 合并输入、行为嵌入和时间嵌入
        x = torch.cat(
            [x, behavior_embed, time_embed], dim=1
        )  # (batch_size, total_input_dim, seq_len)

        # 通过网络
        x = self.network(x)

        # 转换回序列优先格式
        x = x.transpose(1, 2)  # (batch_size, seq_len, input_dim)

        return x

    def __repr__(self) -> str:
        return f"UNet(input_dim={self.input_dim}, behavior_embed_dim={self.behavior_embed_dim}, T={self.T})"


class DiffusionSchedule(nn.Module):
    """扩散调度"""

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end

        # 线性噪声调度
        beta = torch.linspace(beta_start, beta_end, T)
        alpha = 1.0 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        # 确保 alpha_cumprod_prev 与 alpha_cumprod 具有相同的 dtype
        one = torch.ones(1, dtype=alpha_cumprod.dtype)
        alpha_cumprod_prev = torch.cat([one, alpha_cumprod[:-1]])

        # 计算扩散所需的其他参数
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
        sqrt_recip_alpha_cumprod = torch.sqrt(1.0 / alpha_cumprod)
        sqrt_recip_m1_alpha_cumprod = torch.sqrt(1.0 / alpha_cumprod - 1)

        # 注册为 buffer，自动管理设备
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        self.register_buffer("sqrt_alpha_cumprod", sqrt_alpha_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod", sqrt_one_minus_alpha_cumprod
        )
        self.register_buffer("sqrt_recip_alpha_cumprod", sqrt_recip_alpha_cumprod)
        self.register_buffer("sqrt_recip_m1_alpha_cumprod", sqrt_recip_m1_alpha_cumprod)

    def add_noise(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ) -> torch.Tensor:
        """添加噪声到干净样本
        Args:
            x_0: 干净样本，shape (batch_size, seq_len, input_dim)
            t: 扩散时间步，shape (batch_size,)
            noise: 噪声，shape (batch_size, seq_len, input_dim)，如果为None则自动生成
        Returns:
            带噪声的样本，shape (batch_size, seq_len, input_dim)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        batch_size, _, _ = x_0.shape

        # 直接用t在相同设备上索引，t和self.sqrt_alpha_cumprod必须同设备
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].view(
            batch_size, 1, 1
        )  # (B, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(
            batch_size, 1, 1
        )  # (B, 1, 1)

        # (B, 1, 1) 会自动广播到 (B, L, D)
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

    def get_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        """获取指定时间步的beta值"""
        return self.beta[t]

    def get_alpha_cumprod_t(self, t: torch.Tensor) -> torch.Tensor:
        """获取指定时间步的alpha_cumprod值"""
        return self.alpha_cumprod[t]

    def __repr__(self) -> str:
        return f"DiffusionSchedule(T={self.T}, beta_start={self.beta_start}, beta_end={self.beta_end})"


class ConditionDiffusionModel(nn.Module):
    """条件扩散模型"""

    # 默认值常量，提高代码可读性和可维护性
    DEFAULT_INPUT_DIM = 4
    DEFAULT_BEHAVIOR_EMBED_DIM = 32
    DEFAULT_T = 1000
    DEFAULT_NUM_BEHAVIORS = 10
    DEFAULT_CONSTRAINT_EVERY = 10

    def __init__(
        self,
        input_dim: int = None,
        behavior_embed_dim: int = None,
        T: int = None,
        constrained_dims: list = None,
        constraint_bounds: dict = None,
        constraint_every: int = 10,
        normalization_params: dict = None,
        cond_dim: int = None,
    ):
        super().__init__()
        # 导入默认配置
        try:
            from ...config import (
                DEFAULT_INPUT_DIM,
                DEFAULT_BEHAVIOR_EMBED_DIM,
                DEFAULT_T,
            )

            # 使用默认配置或传入的参数
            self.input_dim = input_dim or DEFAULT_INPUT_DIM
            self.behavior_embed_dim = behavior_embed_dim or DEFAULT_BEHAVIOR_EMBED_DIM
            self.T = T or DEFAULT_T
        except ImportError:
            # 导入失败时使用类常量作为默认值
            self.input_dim = input_dim or self.DEFAULT_INPUT_DIM
            self.behavior_embed_dim = (
                behavior_embed_dim or self.DEFAULT_BEHAVIOR_EMBED_DIM
            )
            self.T = T or self.DEFAULT_T

        # 设置需要约束的维度，默认为所有四个维度（d1, l1, d2, l2）
        # 0: 上行延迟 (delay1)
        # 1: 上行丢包率 (loss_rate1)
        # 2: 下行延迟 (delay2)
        # 3: 下行丢包率 (loss_rate2)
        self.constrained_dims = constrained_dims or [0, 1, 2, 3]

        # 确保约束维度有效
        for dim in self.constrained_dims:
            assert (
                dim < self.input_dim
            ), f"约束维度 {dim} 超过了 input_dim {self.input_dim}"

        # 设置约束范围，默认为：
        # 0 (d1): 延迟 ≥ 0
        # 1 (l1): 丢包率归一化后 ∈ [-1, 1]
        # 2 (d2): 延迟 ≥ 0
        # 3 (l2): 丢包率归一化后 ∈ [-1, 1]
        self.constraint_bounds = constraint_bounds or {}
        # 确保所有约束维度都有约束范围
        default_bounds = {
            0: (0, None),  # 上行延迟 ≥ 0
            1: (-1.0, 1.0),  # 上行丢包率归一化后 ∈ [-1, 1]
            2: (0, None),  # 下行延迟 ≥ 0
            3: (-1.0, 1.0),  # 下行丢包率归一化后 ∈ [-1, 1]
        }
        for dim in self.constrained_dims:
            if dim not in self.constraint_bounds:
                self.constraint_bounds[dim] = default_bounds.get(dim, (-1.0, 1.0))

        # 设置约束应用频率，使用类常量作为默认值
        self.constraint_every = constraint_every or self.DEFAULT_CONSTRAINT_EVERY

        # 保存归一化参数
        self.normalization_params = normalization_params

        # 创建条件投影层
        self.cond_dim = cond_dim
        self.condition_proj = (
            nn.Linear(cond_dim, self.behavior_embed_dim)
            if cond_dim is not None
            else None
        )

        logger.info(
            f"初始化条件扩散模型，参数: input_dim={self.input_dim}, behavior_embed_dim={self.behavior_embed_dim}, T={self.T}, constrained_dims={self.constrained_dims}, constraint_bounds={self.constraint_bounds}, constraint_every={self.constraint_every}, has_normalization_params={normalization_params is not None}, cond_dim={cond_dim}"
        )

        # U-Net主干，用于预测噪声
        self.unet = UNet(self.input_dim, self.behavior_embed_dim, T=self.T)

        # 扩散调度 - 管理扩散过程的噪声添加和采样参数
        self.schedule = DiffusionSchedule(self.T)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, condition_vector: torch.Tensor
    ) -> torch.Tensor:
        """前向传播（预测噪声）
        Args:
            x: 带噪声的样本，shape (batch_size, seq_len, input_dim)
            t: 扩散时间步，shape (batch_size, 1, 1) - 每个样本一个时间步（非 per-token）
            condition_vector: 连续条件向量，shape (batch_size, seq_len, cond_dim)
        Returns:
            预测的噪声，shape (batch_size, seq_len, input_dim)
        """
        # 如果 condition_proj 未初始化，动态创建
        if self.condition_proj is None:
            self.condition_proj = nn.Linear(
                condition_vector.shape[-1], self.behavior_embed_dim, device=x.device
            )

        # 将条件向量映射到期望的行为嵌入维度
        behavior_embed = self.condition_proj(condition_vector)

        # U-Net预测噪声
        noise_pred = self.unet(x, t, behavior_embed)

        return noise_pred

    def _process_condition_vector(
        self, condition_vector: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, int, int]:
        """处理条件向量
        Args:
            condition_vector: 连续条件向量
            device: 设备
        Returns:
            tuple: (处理后的行为嵌入, batch_size, seq_len)
        """
        # 处理条件输入
        condition_vector = condition_vector.to(device)
        batch_size, seq_len, _ = condition_vector.shape

        # 如果 condition_proj 未初始化，动态创建
        if self.condition_proj is None:
            self.condition_proj = nn.Linear(
                condition_vector.shape[-1], self.behavior_embed_dim, device=device
            )

        # 将条件向量映射到期望的行为嵌入维度
        behavior_embed = self.condition_proj(condition_vector)
        return behavior_embed, batch_size, seq_len

    def _initialize_noise(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """初始化噪声
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            device: 设备
            noise: 初始噪声，如果为None则自动生成
        Returns:
            torch.Tensor: 初始噪声
        """
        # 初始噪声
        if noise is None:
            logger.debug("使用随机初始噪声")
            noise = torch.randn(batch_size, seq_len, self.input_dim, device=device)
        else:
            logger.debug("使用自定义初始噪声")
            noise = noise.to(device)
        return noise

    def _apply_constraints(
        self, x: torch.Tensor, dims: list, bounds: dict
    ) -> torch.Tensor:
        """应用约束
        Args:
            x: 输入张量
            dims: 需要约束的维度列表
            bounds: 约束边界
        Returns:
            torch.Tensor: 应用约束后的张量
        """
        for dim in dims:
            min_val, max_val = bounds[dim]
            x[:, :, dim] = torch.clip(x[:, :, dim], min=min_val, max=max_val)
        return x

    def _reverse_diffusion_step(
        self,
        x: torch.Tensor,
        t: int,
        behavior_embed: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """执行一步反向扩散
        Args:
            x: 当前噪声样本
            t: 当前时间步
            behavior_embed: 行为嵌入
            device: 设备
        Returns:
            torch.Tensor: 反向扩散后的样本
        """
        # 创建时间步张量，shape (batch_size, 1, 1)
        # t 是离散整数，范围 [0, T-1]，与训练时一致
        # 直接传入归一化后的值，避免在 UNet 中重复归一化
        t_norm = float(t) / float(self.T)
        t_tensor = torch.full((x.shape[0], 1, 1), t_norm, device=device)

        # 预测噪声
        noise_pred = self.unet(x, t_tensor, behavior_embed)

        # 计算当前时间步的参数
        # 获取标量参数，shape: (batch_size,)
        t_idx = torch.full((x.shape[0],), t, device=device, dtype=torch.long)

        # 扩展维度为 (B, 1, 1) 以便正确广播
        alpha_t = self.schedule.alpha[t_idx].view(x.shape[0], 1, 1)
        beta_t = self.schedule.beta[t_idx].view(x.shape[0], 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.schedule.sqrt_one_minus_alpha_cumprod[
            t_idx
        ].view(x.shape[0], 1, 1)

        # 计算均值和方差
        if t > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)

        # 标准DDPM反向扩散公式（Eq. 11）
        # 计算均值 mean（标准DDPM公式）
        mean = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * noise_pred
        )

        # 添加噪声
        if t > 0:
            # 标准DDPM的sigma_t计算
            sigma_t = torch.sqrt(beta_t)
            x = mean + sigma_t * z
        else:
            x = mean

        return x

    def sample(
        self,
        condition_vector: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """采样生成样本
        Args:
            condition_vector: 连续条件向量，shape (batch_size, seq_len, cond_dim)
            noise: 初始噪声，shape (batch_size, seq_len, input_dim)，如果为None则自动生成
        Returns:
            生成的样本，shape (batch_size, seq_len, input_dim)
        """
        # 自动获取模型设备，避免设备错配
        device = next(self.parameters()).device

        # 处理条件向量
        behavior_embed, batch_size, seq_len = self._process_condition_vector(
            condition_vector, device
        )
        logger.info(
            f"开始生成样本，batch_size={batch_size}, seq_len={seq_len}, device={device}"
        )

        # 初始化噪声
        x = self._initialize_noise(batch_size, seq_len, device, noise)

        # 反向扩散过程 - 使用完整的T步采样
        steps = list(range(self.T - 1, -1, -1))
        num_steps = len(steps)
        logger.info(f"使用 {num_steps} 步完整采样")

        # 临时切换到 eval 模式，确保 BatchNorm 使用正确的统计量
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                for i, t in enumerate(steps):
                    # 执行一步反向扩散
                    x = self._reverse_diffusion_step(x, t, behavior_embed, device)

                    # 每constraint_every步应用一次物理约束，减少计算量
                    if i % self.constraint_every == 0 or i == num_steps - 1:
                        # 对指定维度施加合理的约束，使用参数化的约束范围
                        x = self._apply_constraints(
                            x, self.constrained_dims, self.constraint_bounds
                        )
                        # 仅在调试级别时记录日志，优化性能
                        if logger.isEnabledFor(10):  # 10 对应 DEBUG 级别
                            logger.debug(
                                f"采样步骤 {i+1}/{num_steps} (t={t}) 应用物理约束"
                            )

                # 最终约束处理
                x = self._apply_constraints(
                    x, self.constrained_dims, self.constraint_bounds
                )
        finally:
            # 恢复原始训练状态
            if was_training:
                self.train()

        logger.info("采样完成，应用最终约束")

        return x

    def compute_loss(
        self, x_0: torch.Tensor, condition_vector: torch.Tensor
    ) -> torch.Tensor:
        """计算损失
        Args:
            x_0: 干净样本，shape (batch_size, seq_len, input_dim)
            condition_vector: 连续条件向量，shape (batch_size, seq_len, cond_dim)
        Returns:
            损失值
        """
        batch_size, seq_len, _ = x_0.shape

        # 随机采样时间步，整个样本共享同一个时间步，形状为(batch_size,)
        t = torch.randint(0, self.T, (batch_size,), device=x_0.device)

        # 生成噪声
        noise = torch.randn_like(x_0)

        # 添加噪声
        x_t = self.schedule.add_noise(x_0, t, noise)

        # 转换t为(batch_size, 1, 1)形状，用于后续的forward调用
        # 归一化t，使其与sample()中的t_tensor保持一致
        t_norm = t.float() / float(self.T)
        t_reshaped = t_norm.view(batch_size, 1, 1)

        # 预测噪声
        noise_pred = self.forward(x_t, t_reshaped, condition_vector)

        # 仅使用MSE损失，简单且稳定
        loss = F.mse_loss(noise_pred, noise)

        return loss

    def denormalize(
        self, x: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """将归一化的生成样本转换回原始物理空间

        Args:
            x: 归一化的生成样本，shape (batch_size, seq_len, input_dim)

        Returns:
            tuple: 反归一化后的延迟1、丢包率1、延迟2、丢包率2，
                  形状分别为 (batch_size, seq_len), (batch_size, seq_len), (batch_size, seq_len), (batch_size, seq_len)
        """
        if self.normalization_params is None:
            raise ValueError("归一化参数未设置，无法进行反归一化！")

        # 将张量转换为numpy数组
        x_np = x.cpu().numpy()
        batch_size, seq_len, _ = x_np.shape

        # 提取延迟和丢包率（仅支持4维输入）
        delay1_norm = x_np[:, :, 0]
        loss1_norm = x_np[:, :, 1]
        delay2_norm = x_np[:, :, 2]
        loss2_norm = x_np[:, :, 3]

        # 导入归一化器
        from .normalization import Normalizer

        # 检查是否有上下行独立的合法丢包值
        if (
            "valid_loss_values_up" in self.normalization_params
            and "valid_loss_values_down" in self.normalization_params
        ):
            # 使用上下行独立的合法丢包值
            normalizer = Normalizer()
            normalizer.normalization_params = self.normalization_params
        else:
            # 使用全局合法丢包值（兼容旧版本）
            valid_loss_values = self.normalization_params.get("loss_rate", {}).get(
                "valid_loss_values", [0.0, 0.5, 1.0]
            )
            normalizer = Normalizer(valid_loss_values)
            normalizer.normalization_params = self.normalization_params

        # 对每个样本进行反归一化
        delay1_list = []
        loss1_list = []
        delay2_list = []
        loss2_list = []

        for i in range(batch_size):
            if self.input_dim == 2:
                # 单流反归一化
                delay1, loss1 = normalizer.denormalize(delay1_norm[i], loss1_norm[i])
                delay1_list.append(delay1)
                loss1_list.append(loss1)
            else:
                # 双流反归一化
                delay1, loss1, delay2, loss2 = normalizer.denormalize4d(
                    delay1_norm[i], loss1_norm[i], delay2_norm[i], loss2_norm[i]
                )
                delay1_list.append(delay1)
                loss1_list.append(loss1)
                delay2_list.append(delay2)
                loss2_list.append(loss2)

        if self.input_dim == 2:
            return np.array(delay1_list), np.array(loss1_list)
        else:
            return (
                np.array(delay1_list),
                np.array(loss1_list),
                np.array(delay2_list),
                np.array(loss2_list),
            )

    def __repr__(self) -> str:
        return f"ConditionDiffusionModel(input_dim={self.input_dim}, behavior_embed_dim={self.behavior_embed_dim}, T={self.T})"
