#!/usr/bin/env python3
import torch

# 加载checkpoint
checkpoint = torch.load(
    "/Users/xiaotuanzi/PycharmProjects/NPS/output/train_results/diffusion_model_final.pth",
    map_location="cpu",
    weights_only=False,
)

# 打印checkpoint的键
print("Checkpoint keys:", list(checkpoint.keys()))

# 打印模型状态字典的键和形状
state_dict = checkpoint["model_state_dict"]
print("\nModel state dict keys:")
for key in state_dict.keys():
    print(f"  {key}: {state_dict[key].shape}")
