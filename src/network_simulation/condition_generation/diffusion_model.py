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

    def __init__(self, d_model: int, T: int = 1000):
        """初始化Time2Vec位置编码模块

        Args:
            d_model (int): 输出编码维度，必须 >= 1
            T (int, optional): 扩散步数，用于初始化编码参数，默认值为1000

        Examples:
            >>> from network_simulation.condition_generation.diffusion_model import Time2Vec
            >>> # 初始化一个d_model=16的Time2Vec模块
            >>> time2vec = Time2Vec(d_model=16, T=1000)
        """
        super().__init__()
        self.d_model = d_model
        self.T = T
        if d_model < 1:
            raise ValueError("d_model must be >= 1")
        # 第0维用cos，其余d_model-1维用sin
        # 根据扩散步数T初始化omega_0，让周期覆盖整个[0,1]区间
        # 使用接近2π的随机值初始化，保持良好初始编码能力的同时允许学习
        omega_0 = 2 * np.pi if T > 0 else 1.0
        # 让w0可以学习，初始值接近2π，添加小的随机扰动
        self.w0 = nn.Parameter(torch.tensor([omega_0]) + torch.randn(1) * 0.1)
        self.b0 = nn.Parameter(torch.randn(1) * 0.1)
        if d_model > 1:
            # 使用随机初始化，让不同维度的频率不同，更好地编码时间信息
            # 参考Time2Vec论文，使用对数尺度初始化
            self.w = nn.Parameter(torch.randn(d_model - 1) * 0.1)
            self.b = nn.Parameter(torch.randn(d_model - 1) * 0.1)
        else:
            self.w = None
            self.b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x (torch.Tensor): 输入时间步张量，shape (batch_size, seq_len, 1)，应归一化到[0,1]范围

        Returns:
            torch.Tensor: 位置编码后的张量，shape (batch_size, seq_len, d_model)

        Examples:
            >>> from network_simulation.condition_generation.diffusion_model import Time2Vec
            >>> import torch
            >>> time2vec = Time2Vec(d_model=16, T=1000)
            >>> # 创建一个随机时间步张量，shape (2, 100, 1)，值在[0,1]范围内
            >>> x = torch.rand(2, 100, 1)
            >>> # 生成位置编码
            >>> time_embed = time2vec(x)
            >>> print(f"输入形状: {x.shape}, 输出形状: {time_embed.shape}")
        """
        # 将输入转换为float类型以匹配参数类型
        x = x.float()

        # 检查输入是否在[0,1]范围，超出则记录警告
        if x.min() < -1e-6 or x.max() > 1 + 1e-6:
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
    """因果膨胀卷积层

    该层实现了因果膨胀卷积，确保模型在生成序列时只能看到过去和当前的信息，
    适合用于时序数据生成任务。
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ):
        """初始化因果膨胀卷积层

        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            kernel_size (int): 卷积核大小
            dilation (int): 膨胀率

        Examples:
            >>> from network_simulation.condition_generation.diffusion_model import CausalDilatedConv
            >>> # 初始化一个因果膨胀卷积层
            >>> conv = CausalDilatedConv(
            ...     in_channels=64,
            ...     out_channels=64,
            ...     kernel_size=3,
            ...     dilation=2
            ... )
        """
        super().__init__()
        # 确保kernel_size和dilation合法
        assert kernel_size >= 1, f"Kernel size must be at least 1, got {kernel_size}"
        assert dilation >= 1, f"Dilation must be at least 1, got {dilation}"

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
            x (torch.Tensor): 输入张量，shape (batch_size, in_channels, seq_len)

        Returns:
            torch.Tensor: 卷积后的张量，shape (batch_size, out_channels, seq_len)

        Examples:
            >>> from network_simulation.condition_generation.diffusion_model import CausalDilatedConv
            >>> import torch
            >>> conv = CausalDilatedConv(in_channels=64, out_channels=64, kernel_size=3, dilation=2)
            >>> # 创建一个随机输入张量
            >>> x = torch.randn(2, 64, 100)
            >>> # 执行卷积操作
            >>> out = conv(x)
            >>> print(f"输入形状: {x.shape}, 输出形状: {out.shape}")
        """
        out = self.conv(x)
        # 移除填充以保持因果结构，防止 padding=0 时空切片
        return out[:, :, : -self.padding] if self.padding > 0 else out

    def __repr__(self) -> str:
        return f"CausalDilatedConv(in_channels={self.conv.in_channels}, out_channels={self.conv.out_channels}, kernel_size={self.conv.kernel_size[0]}, dilation={self.conv.dilation[0]})"


class UNetBlock(nn.Module):
    """UNet块

    该块实现了基于因果膨胀卷积的残差连接结构，用于构建深层因果网络，
    适合用于时序数据生成任务。
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ):
        """初始化UNet块

        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            kernel_size (int): 卷积核大小
            dilation (int): 膨胀率

        Examples:
            >>> from network_simulation.condition_generation.diffusion_model import UNetBlock
            >>> # 初始化一个UNet块
            >>> unet_block = UNetBlock(
            ...     in_channels=64,
            ...     out_channels=64,
            ...     kernel_size=3,
            ...     dilation=1
            ... )
        """
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
            x (torch.Tensor): 输入张量，shape (batch_size, in_channels, seq_len)

        Returns:
            torch.Tensor: 经过UNet块处理后的张量，shape (batch_size, out_channels, seq_len)

        Examples:
            >>> from network_simulation.condition_generation.diffusion_model import UNetBlock
            >>> import torch
            >>> unet_block = UNetBlock(in_channels=64, out_channels=64, kernel_size=3, dilation=1)
            >>> # 创建一个随机输入张量
            >>> x = torch.randn(2, 64, 100)
            >>> # 执行UNet块操作
            >>> out = unet_block(x)
            >>> print(f"输入形状: {x.shape}, 输出形状: {out.shape}")
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


class CausalDilatedNet(nn.Module):
    """因果膨胀卷积网络，使用因果膨胀卷积

    这是一个基于因果膨胀卷积的网络，用于条件扩散模型中的噪声预测。
    它采用堆叠的残差块结构，适合因果时序生成任务。
    """

    def __init__(self, input_dim: int, behavior_embed_dim: int, T: int = 1000):
        """初始化因果膨胀卷积网络

        Args:
            input_dim (int): 输入维度，即网络处理的特征维度
            behavior_embed_dim (int): 行为嵌入维度
            T (int, optional): 扩散步数，用于初始化Time2Vec编码，默认值为1000

        Examples:
            >>> from network_simulation.condition_generation.diffusion_model import CausalDilatedNet
            >>> # 初始化一个因果膨胀卷积网络
            >>> net = CausalDilatedNet(
            ...     input_dim=4,
            ...     behavior_embed_dim=32,
            ...     T=1000
            ... )
        """
        super().__init__()
        self.input_dim = input_dim
        self.behavior_embed_dim = behavior_embed_dim
        self.T = T

        # Time2Vec位置编码，传递扩散步数T
        self.time2vec = Time2Vec(behavior_embed_dim, self.T)

        # 计算总输入维度：原始输入维度 + 行为嵌入维度 + 时间嵌入维度
        total_input_dim = input_dim + behavior_embed_dim + behavior_embed_dim

        # 简化的因果膨胀卷积网络，减少层数和通道数，提高训练稳定性
        # 添加输入投影层，避免信息瓶颈
        proj_out_dim = 64
        self.network = nn.Sequential(
            # 输入投影层，将总输入维度投影到固定维度64，避免信息丢失
            nn.Conv1d(total_input_dim, proj_out_dim, kernel_size=1),
            # 因果膨胀卷积层，使用递增的膨胀率扩大感受野
            UNetBlock(proj_out_dim, proj_out_dim, kernel_size=3, dilation=1),
            UNetBlock(proj_out_dim, proj_out_dim, kernel_size=3, dilation=2),
            UNetBlock(proj_out_dim, proj_out_dim, kernel_size=3, dilation=4),
            UNetBlock(proj_out_dim, proj_out_dim, kernel_size=3, dilation=8),
            # 输出层：映射回输入维度，使用因果卷积确保因果性
            CausalDilatedConv(proj_out_dim, input_dim, kernel_size=3, dilation=1),
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, behavior_embed: torch.Tensor
    ) -> torch.Tensor:
        """前向传播

        Args:
            x (torch.Tensor): 输入张量，shape (batch_size, seq_len, input_dim)
            t (torch.Tensor): 时间步张量，shape (batch_size, 1, 1) - 每个样本共享一个时间步
            behavior_embed (torch.Tensor): 行为嵌入向量，shape (batch_size, seq_len, behavior_embed_dim)

        Returns:
            torch.Tensor: 网络输出张量，shape (batch_size, seq_len, input_dim)

        Examples:
            >>> from network_simulation.condition_generation.diffusion_model import CausalDilatedNet
            >>> import torch
            >>> net = CausalDilatedNet(input_dim=4, behavior_embed_dim=32, T=1000)
            >>> # 创建随机输入张量
            >>> batch_size, seq_len = 2, 100
            >>> x = torch.randn(batch_size, seq_len, 4)
            >>> t = torch.randn(batch_size, 1, 1)
            >>> behavior_embed = torch.randn(batch_size, seq_len, 32)
            >>> # 执行前向传播
            >>> out = net(x, t, behavior_embed)
            >>> print(f"输入形状: {x.shape}, 输出形状: {out.shape}")
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
        return f"CausalDilatedNet(input_dim={self.input_dim}, behavior_embed_dim={self.behavior_embed_dim}, T={self.T})"


class DiffusionSchedule(nn.Module):
    """扩散调度

    该类实现了DDPM（Denoising Diffusion Probabilistic Models）的扩散过程调度，
    负责管理噪声添加和采样过程中的参数计算。
    """

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        """初始化扩散调度

        Args:
            T (int, optional): 扩散步数，默认值为1000
            beta_start (float, optional): 初始噪声强度，默认值为1e-4
            beta_end (float, optional): 最终噪声强度，默认值为0.02

        Examples:
            >>> from network_simulation.condition_generation.diffusion_model import DiffusionSchedule
            >>> # 初始化一个扩散调度
            >>> schedule = DiffusionSchedule(
            ...     T=1000,
            ...     beta_start=1e-4,
            ...     beta_end=0.02
            ... )
        """
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

        根据扩散时间步t，将噪声添加到干净样本x_0上，生成带噪声的样本x_t。

        Args:
            x_0 (torch.Tensor): 干净样本，shape (batch_size, seq_len, input_dim)
            t (torch.Tensor): 扩散时间步，shape (batch_size,)
            noise (torch.Tensor, optional): 噪声，shape (batch_size, seq_len, input_dim)，如果为None则自动生成

        Returns:
            torch.Tensor: 带噪声的样本，shape (batch_size, seq_len, input_dim)

        Examples:
            >>> from network_simulation.condition_generation.diffusion_model import DiffusionSchedule
            >>> import torch
            >>> schedule = DiffusionSchedule(T=1000)
            >>> # 创建干净样本
            >>> batch_size, seq_len, input_dim = 2, 100, 4
            >>> x_0 = torch.randn(batch_size, seq_len, input_dim)
            >>> # 随机选择时间步
            >>> t = torch.randint(0, 1000, (batch_size,))
            >>> # 添加噪声
            >>> x_t = schedule.add_noise(x_0, t)
            >>> print(f"干净样本形状: {x_0.shape}, 噪声样本形状: {x_t.shape}")
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

        # 因果膨胀卷积网络主干，用于预测噪声
        self.model = CausalDilatedNet(self.input_dim, self.behavior_embed_dim, T=self.T)

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
        # 统一设备
        device = next(self.parameters()).device
        x = x.to(device)
        t = t.to(device)
        condition_vector = condition_vector.to(device)

        # 确保condition_proj已初始化
        if self.condition_proj is None:
            raise ValueError(
                "condition_proj未初始化，请确保在模型初始化时提供cond_dim参数"
            )

        # 将条件向量映射到期望的行为嵌入维度
        behavior_embed = self.condition_proj(condition_vector)

        # 因果膨胀卷积网络预测噪声
        noise_pred = self.model(x, t, behavior_embed)

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

        # 确保condition_proj已初始化
        if self.condition_proj is None:
            raise ValueError(
                "condition_proj未初始化，请确保在模型初始化时提供cond_dim参数"
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
        noise_pred = self.model(x, t_tensor, behavior_embed)

        # 计算当前时间步的参数
        # 获取标量参数，shape: (batch_size,)
        t_idx = torch.full((x.shape[0],), t, device=device, dtype=torch.long)

        # 扩展维度为 (B, 1, 1) 以便正确广播
        alpha_t = self.schedule.alpha[t_idx].view(x.shape[0], 1, 1)
        beta_t = self.schedule.beta[t_idx].view(x.shape[0], 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.schedule.sqrt_one_minus_alpha_cumprod[
            t_idx
        ].view(x.shape[0], 1, 1)
        sqrt_recip_alpha_cumprod_t = self.schedule.sqrt_recip_alpha_cumprod[t_idx].view(
            x.shape[0], 1, 1
        )

        # 计算均值和方差
        if t > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)

        # 标准DDPM反向扩散公式（Eq. 11）
        # 计算均值 mean（标准DDPM公式）
        # 注意：由于 alpha_t = 1 - beta_t，所以 (1 - alpha_t) = beta_t
        # 因此，以下两种表示等价：
        # coeff = (1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t
        # coeff = beta_t / sqrt_one_minus_alpha_cumprod_t
        # 当前代码使用 (1 - alpha_t)，这是DDPM论文中使用的形式，更清晰
        coeff = (1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t
        mean = sqrt_recip_alpha_cumprod_t * (x - coeff * noise_pred)

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

    def __repr__(self) -> str:
        return f"ConditionDiffusionModel(input_dim={self.input_dim}, behavior_embed_dim={self.behavior_embed_dim}, T={self.T})"
