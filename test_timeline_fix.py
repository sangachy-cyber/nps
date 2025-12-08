#!/usr/bin/env python3
"""
测试脚本：验证时延曲线修复
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(os.path.abspath('src'))

# 从实际生成的处理后数据中读取数据
data_path = Path("data/results/e2e_pipeline/processed/20251203_230356_b6x-playback_processed.csv")

if not data_path.exists():
    print(f"数据文件不存在: {data_path}")
    sys.exit(1)

# 读取数据
raw_data = pd.read_csv(data_path, parse_dates=["timestamp"])
print(f"数据加载完成，共 {len(raw_data)} 行")
print(f"时间范围: {raw_data['timestamp'].min()} 到 {raw_data['timestamp'].max()}")
print(f"时延范围: {raw_data['delay'].min()} 到 {raw_data['delay'].max()} ms")
print(f"丢包率范围: {raw_data['loss_rate'].min()} 到 {raw_data['loss_rate'].max()}")

# 计算相对时间
raw_data['relative_time'] = (raw_data['timestamp'] - raw_data['timestamp'].min()).dt.total_seconds()

# 重采样数据，减少数据点数量
if len(raw_data) > 1000:
    step = len(raw_data) // 1000
    sampled_data = raw_data.iloc[::step]
else:
    sampled_data = raw_data

# 确保数据按照relative_time排序
sampled_data = sampled_data.sort_values('relative_time')

print(f"\n采样后数据点数量: {len(sampled_data)}")
print(f"relative_time范围: {sampled_data['relative_time'].min()} 到 {sampled_data['relative_time'].max()} 秒")
print(f"delay范围: {sampled_data['delay'].min()} 到 {sampled_data['delay'].max()} ms")

# 创建测试图表，模拟修复后的代码
plt.figure(figsize=(15, 8))
ax1 = plt.subplot(111)

# 绘制时延曲线
delay_line, = ax1.plot(sampled_data['relative_time'], sampled_data['delay'], 'b-', linewidth=3, alpha=0.9, label='时延 (ms)')

# 使用自适应Y轴范围
delay_min = sampled_data['delay'].min()
delay_max = sampled_data['delay'].max()
y_min = max(0, delay_min - 50)
y_max = min(2000, delay_max + 50)
ax1.set_ylim(y_min, y_max)

# 设置Y轴刻度
y_ticks = np.linspace(y_min, y_max, 5)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels([f'{tick:.0f}' for tick in y_ticks], fontweight='bold', fontsize=12, color='b')

# 设置其他样式
ax1.set_xlabel('相对时间 (秒)', fontsize=14)
ax1.set_ylabel('时延 (ms)', color='b', fontsize=14, fontweight='bold')
ax1.tick_params('y', colors='b', labelsize=12, width=3, length=15, direction='out')
ax1.spines['left'].set_color('b')
ax1.spines['left'].set_linewidth(3)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper right', fontsize=10)
ax1.set_title('测试：修复后的时延曲线', fontsize=16)

# 保存测试图表
test_output = Path("test_timeline_fix.png")
plt.savefig(test_output, dpi=150, bbox_inches="tight")
print(f"\n测试图表已保存到: {test_output}")

# 检查数据是否正确排序
if sampled_data['relative_time'].is_monotonic_increasing:
    print("✓ 数据已正确按照relative_time排序")
else:
    print("✗ 数据未按照relative_time排序")

# 检查X轴和Y轴数据
print(f"\n前5个数据点:")
print(sampled_data[['relative_time', 'delay']].head())

print(f"\n后5个数据点:")
print(sampled_data[['relative_time', 'delay']].tail())

print("\n测试完成！")
