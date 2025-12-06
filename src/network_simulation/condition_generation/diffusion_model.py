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
    """Time2Vec位置编码"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(1, d_model)
        self.w0 = nn.Parameter(torch.randn(1, 1))
        self.b0 = nn.Parameter(torch.randn(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入时间步张量，shape (batch_size, seq_len, 1)

        Returns:
            位置编码后的张量，shape (batch_size, seq_len, d_model)
        """
        # 将输入转换为float类型以匹配参数类型
        x = x.float()
        v0 = torch.cos(torch.matmul(x, self.w0) + self.b0)
        v1 = self.linear(x)
        v1 = torch.sin(v1)
        return torch.cat([v0, v1[:, :, 1:]], dim=-1)


class CausalDilatedConv(nn.Module):
    """因果膨胀卷积层"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
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
        return out[:, :, : -self.padding]  # 移除填充以保持因果结构


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


class CrossAttention(nn.Module):
    """交叉注意力层"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """前向传播

        Args:
            query: 查询张量，shape (batch_size, seq_len, d_model)
            key: 键张量，shape (batch_size, seq_len, d_model)
            value: 值张量，shape (batch_size, seq_len, d_model)

        Returns:
            注意力机制处理后的张量，shape (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)
        seq_len = query.size(1)

        # 线性投影
        Q = (
            self.q_linear(query)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_linear(key)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_linear(value)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)

        # 输出
        out = (
            torch.matmul(attention, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, -1)
        )
        out = self.out_linear(out)
        return out


class BehaviorEmbedding(nn.Module):
    """行为嵌入层"""

    def __init__(self, num_behaviors: int, d_embed: int):
        super().__init__()
        self.embedding = nn.Embedding(num_behaviors, d_embed)
        self.dropout = nn.Dropout(0.1)

    def forward(self, behavior_ids: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            behavior_ids: 行为ID张量，shape (batch_size, seq_len)

        Returns:
            行为嵌入向量，shape (batch_size, seq_len, d_embed)
        """
        return self.dropout(self.embedding(behavior_ids))


class UNet(nn.Module):
    """简化版U-Net主干网络，使用因果膨胀卷积"""

    def __init__(self, input_dim: int, behavior_embed_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.behavior_embed_dim = behavior_embed_dim

        # Time2Vec位置编码
        self.time2vec = Time2Vec(behavior_embed_dim)

        # 计算总输入维度：原始输入维度 + 行为嵌入维度
        total_input_dim = input_dim + behavior_embed_dim

        # 深层因果膨胀卷积网络，提高表达能力
        # 使用更长的膨胀率序列，扩大感受野，捕捉更复杂的时序关系
        self.network = nn.Sequential(
            # 第一层：膨胀率1，提取低层特征
            UNetBlock(total_input_dim, 64, kernel_size=3, dilation=1),
            # 第二层：膨胀率2，扩大感受野
            UNetBlock(64, 128, kernel_size=3, dilation=2),
            # 第三层：膨胀率4，进一步扩大感受野
            UNetBlock(128, 256, kernel_size=3, dilation=4),
            # 第四层：膨胀率8，扩大感受野
            UNetBlock(256, 512, kernel_size=3, dilation=8),
            # 第五层：膨胀率16，最大感受野
            UNetBlock(512, 512, kernel_size=3, dilation=16),
            # 第六层：膨胀率32，最大感受野
            UNetBlock(512, 512, kernel_size=3, dilation=32),
            # 第七层：膨胀率64，最大感受野
            UNetBlock(512, 512, kernel_size=3, dilation=64),
            # 第八层：膨胀率32，缩小感受野
            UNetBlock(512, 512, kernel_size=3, dilation=32),
            # 第九层：膨胀率16，缩小感受野
            UNetBlock(512, 512, kernel_size=3, dilation=16),
            # 第十层：膨胀率8，缩小感受野
            UNetBlock(512, 256, kernel_size=3, dilation=8),
            # 第十一层：膨胀率4，缩小感受野
            UNetBlock(256, 128, kernel_size=3, dilation=4),
            # 第十二层：膨胀率2，缩小感受野
            UNetBlock(128, 64, kernel_size=3, dilation=2),
            # 第十三层：膨胀率1，恢复细节
            UNetBlock(64, 64, kernel_size=3, dilation=1),
            # 输出层：映射回输入维度
            nn.Conv1d(64, input_dim, kernel_size=3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, behavior_embed: torch.Tensor
    ) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量，shape (batch_size, seq_len, input_dim)
            t: 时间步张量，shape (batch_size, seq_len, 1)
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

        # 合并输入和行为嵌入
        x = torch.cat(
            [x, behavior_embed], dim=1
        )  # (batch_size, total_input_dim, seq_len)

        # 通过网络
        x = self.network(x)

        # 转换回序列优先格式
        x = x.transpose(1, 2)  # (batch_size, seq_len, input_dim)

        return x


class DiffusionSchedule:
    """扩散调度"""

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end

        # 线性噪声调度
        self.beta = torch.linspace(beta_start, beta_end, T)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.alpha_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alpha_cumprod[:-1]]
        )

        # 计算扩散所需的其他参数
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alpha_cumprod = torch.sqrt(1.0 / self.alpha_cumprod)
        self.sqrt_recip_m1_alpha_cumprod = torch.sqrt(1.0 / self.alpha_cumprod - 1)

    def add_noise(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ) -> torch.Tensor:
        """添加噪声到干净样本
        Args:
            x_0: 干净样本，shape (batch_size, seq_len, input_dim)
            t: 扩散时间步，shape (batch_size, seq_len)
            noise: 噪声，shape (batch_size, seq_len, input_dim)，如果为None则自动生成
        Returns:
            带噪声的样本，shape (batch_size, seq_len, input_dim)
        """
        # 将调度参数张量移动到与输入x_0相同的设备
        device = x_0.device
        self.sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(device)
        self.sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod.to(device)

        if noise is None:
            noise = torch.randn_like(x_0)

        # 将t移动到与调度参数相同的设备（CPU）进行索引操作
        t_cpu = t.cpu()

        # 获取时间步对应的系数
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[
            t_cpu
        ]  # shape (batch_size, seq_len)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[
            t_cpu
        ]  # shape (batch_size, seq_len)

        # 将结果移回原始设备
        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.to(device)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.to(device)

        # 扩展维度以匹配x_0和noise的形状
        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.unsqueeze(
            -1
        )  # shape (batch_size, seq_len, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(
            -1
        )  # shape (batch_size, seq_len, 1)

        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

    def get_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        """获取指定时间步的beta值"""
        return self.beta[t]

    def get_alpha_cumprod_t(self, t: torch.Tensor) -> torch.Tensor:
        """获取指定时间步的alpha_cumprod值"""
        return self.alpha_cumprod[t]


class ConditionDiffusionModel(nn.Module):
    """条件扩散模型"""

    def __init__(
        self,
        input_dim: int = None,
        num_behaviors: int = None,
        behavior_embed_dim: int = None,
        T: int = None,
    ):
        super().__init__()
        # 导入默认配置
        try:
            from ...config import (
                DEFAULT_INPUT_DIM,
                DEFAULT_BEHAVIOR_EMBED_DIM,
                DEFAULT_T
            )
            # 使用默认配置或传入的参数
            self.input_dim = input_dim or DEFAULT_INPUT_DIM
            self.behavior_embed_dim = behavior_embed_dim or DEFAULT_BEHAVIOR_EMBED_DIM
            self.T = T or DEFAULT_T
        except ImportError:
            # 导入失败时使用默认值
            self.input_dim = input_dim or 2
            self.behavior_embed_dim = behavior_embed_dim or 32
            self.T = T or 1000

        # 设置行为数量，默认为10
        self.num_behaviors = num_behaviors or 10

        logger.info(f"初始化条件扩散模型，参数: input_dim={self.input_dim}, num_behaviors={self.num_behaviors}, behavior_embed_dim={self.behavior_embed_dim}, T={self.T}")

        # 行为嵌入层
        self.behavior_embedding = BehaviorEmbedding(self.num_behaviors, self.behavior_embed_dim)

        # U-Net主干
        self.unet = UNet(self.input_dim, self.behavior_embed_dim)

        # 扩散调度
        self.schedule = DiffusionSchedule(self.T)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, behavior_ids: torch.Tensor
    ) -> torch.Tensor:
        """前向传播（预测噪声）
        Args:
            x: 带噪声的样本，shape (batch_size, seq_len, input_dim)
            t: 扩散时间步，shape (batch_size, seq_len, 1)
            behavior_ids: 行为ID，shape (batch_size, seq_len)
        Returns:
            预测的噪声，shape (batch_size, seq_len, input_dim)
        """
        # 获取行为嵌入
        behavior_embed = self.behavior_embedding(behavior_ids)

        # U-Net预测噪声
        noise_pred = self.unet(x, t, behavior_embed)

        return noise_pred

    def sample(
        self,
        behavior_ids: torch.Tensor,
        device: torch.device,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """采样生成样本
        Args:
            behavior_ids: 行为ID序列，shape (batch_size, seq_len)
            device: 设备
            noise: 初始噪声，shape (batch_size, seq_len, input_dim)，如果为None则自动生成
        Returns:
            生成的样本，shape (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len = behavior_ids.shape
        logger.info(f"开始生成样本，batch_size={batch_size}, seq_len={seq_len}, device={device}")

        # 初始噪声
        if noise is None:
            logger.debug("使用随机初始噪声")
            noise = torch.randn(batch_size, seq_len, self.input_dim, device=device)
        else:
            logger.debug("使用自定义初始噪声")

        x = noise

        # 提前计算所有行为嵌入，避免重复计算
        behavior_embed = self.behavior_embedding(behavior_ids)

        # 反向扩散过程 - 减少采样步数，加速生成过程
        # 只使用25%的步数，仍然可以生成高质量的样本
        step_skip = 4
        steps = list(range(self.T - 1, -1, -step_skip))
        num_steps = len(steps)
        logger.info(f"使用 {num_steps} 步采样（原 {self.T} 步的 25%）")

        for i, t in enumerate(steps):
            # 创建时间步张量
            t_tensor = torch.full(
                (batch_size, seq_len, 1), t, device=device, dtype=torch.long
            )

            # 预测噪声
            with torch.no_grad():
                noise_pred = self.unet(x, t_tensor, behavior_embed)

            # 计算当前时间步的参数
            beta_t = self.schedule.get_beta_t(t)
            alpha_t = self.schedule.alpha[t]
            alpha_cumprod_t = self.schedule.get_alpha_cumprod_t(t)

            # 计算均值和方差
            if t > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)

            # 反向扩散公式
            x = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * noise_pred
            )
            if t > 0:
                # 计算sigma_t，考虑跳过的步数
                sigma_t = torch.sqrt(beta_t * step_skip)
                x += sigma_t * z

            # 每10步应用一次物理约束，减少计算量
            if i % 10 == 0 or i == num_steps - 1:
                # 对延迟施加合理的约束，确保生成的归一化延迟值在[-1, 1]范围内
                # 这个范围与训练数据的归一化范围一致
                x[:, :, 0] = torch.clip(x[:, :, 0], min=-1.0, max=1.0)
                # 对丢包率施加约束，因为丢包率的归一化范围固定为[-1, 1]
                x[:, :, 1] = torch.clip(x[:, :, 1], min=-1.0, max=1.0)
                logger.debug(f"采样步骤 {i+1}/{num_steps} (t={t}) 应用物理约束")

        # 最终约束处理
        x[:, :, 0] = torch.clip(x[:, :, 0], min=-1.0, max=1.0)
        x[:, :, 1] = torch.clip(x[:, :, 1], min=-1.0, max=1.0)
        logger.info("采样完成，应用最终约束")

        return x

    def compute_loss(
        self,
        x_0: torch.Tensor,
        behavior_ids: torch.Tensor
    ) -> torch.Tensor:
        """计算损失
        Args:
            x_0: 干净样本，shape (batch_size, seq_len, input_dim)
            behavior_ids: 行为ID，shape (batch_size, seq_len)
        Returns:
            损失值
        """
        batch_size, seq_len, _ = x_0.shape

        # 随机采样时间步，形状为(batch_size, seq_len)
        t = torch.randint(0, self.T, (batch_size, seq_len), device=x_0.device)

        # 生成噪声
        noise = torch.randn_like(x_0)

        # 添加噪声
        x_t = self.schedule.add_noise(x_0, t, noise)

        # 转换t为(batch_size, seq_len, 1)形状，用于后续的forward调用
        t_reshaped = t.unsqueeze(-1)

        # 预测噪声
        noise_pred = self.forward(x_t, t_reshaped, behavior_ids)

        # 1. MSE损失（基础损失）
        mse_loss = F.mse_loss(noise_pred, noise)

        # 2. 统计特性损失
        # 计算原始数据的统计特性
        x_0_mean = torch.mean(x_0, dim=1, keepdim=True)  # (batch_size, 1, input_dim)
        x_0_var = torch.var(x_0, dim=1, keepdim=True)    # (batch_size, 1, input_dim)

        # 计算去噪后的数据（使用噪声预测）
        # 简化的去噪过程，用于损失计算
        # 确保调度参数与输入在同一设备上
        device = x_0.device
        alpha = self.schedule.alpha.to(device)  # 移动到正确设备
        alpha_cumprod = self.schedule.alpha_cumprod.to(device)  # 移动到正确设备

        # 获取当前设备上的时间步值
        t_device = t.to(device)

        # 使用设备匹配的张量进行计算
        alpha_t = alpha[t_device].unsqueeze(-1)  # (batch_size, seq_len, 1)
        alpha_cumprod_t = alpha_cumprod[t_device].unsqueeze(-1)  # (batch_size, seq_len, 1)

        # 简化的去噪计算
        x_pred = (x_t - noise_pred * torch.sqrt(1 - alpha_cumprod_t)) / torch.sqrt(alpha_t)

        # 计算去噪后数据的统计特性
        x_pred_mean = torch.mean(x_pred, dim=1, keepdim=True)  # (batch_size, 1, input_dim)
        x_pred_var = torch.var(x_pred, dim=1, keepdim=True)    # (batch_size, 1, input_dim)

        # 统计特性损失（均值和方差匹配）
        mean_loss = F.mse_loss(x_pred_mean, x_0_mean)
        var_loss = F.mse_loss(x_pred_var, x_0_var)
        stat_loss = mean_loss + var_loss

        # 3. 时序特性损失（自相关系数）
        def autocorrelation(x, lag=1):
            """计算自相关系数"""
            # x shape: (batch_size, seq_len, input_dim)
            batch_size, seq_len, input_dim = x.shape
            x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)

            # 计算均值
            mean = torch.mean(x, dim=2, keepdim=True)  # (batch_size, input_dim, 1)

            # 标准化
            x_std = x - mean  # (batch_size, input_dim, seq_len)
            var = torch.sum(x_std ** 2, dim=2, keepdim=True)  # (batch_size, input_dim, 1)

            # 计算自协方差
            cov = torch.sum(x_std[:, :, :-lag] * x_std[:, :, lag:], dim=2, keepdim=True)

            # 计算自相关系数
            corr = cov / (var + 1e-8)
            return corr.squeeze(-1)  # (batch_size, input_dim)

        # 计算自相关系数
        x_0_acf = autocorrelation(x_0)
        x_pred_acf = autocorrelation(x_pred)

        # 自相关系数损失
        acf_loss = F.mse_loss(x_pred_acf, x_0_acf)

        # 4. Wasserstein距离损失（地球移动距离）
        def wasserstein_distance(x, y):
            """计算Wasserstein距离（基于排序）"""
            # x, y shape: (batch_size, seq_len, input_dim)
            batch_size, seq_len, input_dim = x.shape

            # 对每个维度进行排序
            x_sorted, _ = torch.sort(x, dim=1)
            y_sorted, _ = torch.sort(y, dim=1)

            # 计算累积距离
            dist = torch.mean(torch.abs(x_sorted - y_sorted), dim=(1, 2))  # (batch_size,)
            return dist

        # 计算Wasserstein距离损失
        wasserstein_loss = wasserstein_distance(x_0, x_pred)
        wasserstein_loss = torch.mean(wasserstein_loss)  # 计算批次平均

        # 简化损失函数，只保留高效的损失函数
        # 移除基于KDE的损失函数，减少内存开销
        # MSE损失（基础损失） + 统计特性损失 + 时序特性损失 + Wasserstein距离损失
        total_loss = 0.3 * mse_loss + 0.2 * stat_loss + 0.2 * acf_loss + 0.3 * wasserstein_loss

        return total_loss
