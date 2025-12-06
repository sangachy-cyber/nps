#!/usr/bin/env python3
import torch
from network_simulation.utils.logger import get_logger

# 初始化日志记录器
logger = get_logger(__name__)

# 加载checkpoint
checkpoint = torch.load(
    "/Users/xiaotuanzi/PycharmProjects/NPS/output/train_results/diffusion_model_final.pth",
    map_location="cpu",
    weights_only=False,
)

# 打印checkpoint的键
logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

# 打印模型状态字典的键和形状
state_dict = checkpoint["model_state_dict"]
logger.info("\nModel state dict keys:")
for key in state_dict.keys():
    logger.info(f"  {key}: {state_dict[key].shape}")
