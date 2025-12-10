# 网络仿真参数生成方案 v1.4

[English Version](README.md)

## 项目概述

本项目提供了一个基于真实网络数据的高保真网络仿真参数生成方案。它可以生成10分钟的网络时序数据，精确模拟任意混合网络场景，如稳定→突发丢包→高抖动→恢复。

## 功能特性

- ✅ 生成单一网络行为（如高抖动、突发丢包、持续拥塞）
- ✅ 通过灵活的调度表生成任意混合行为序列
- ✅ 确保统计特性、时序动态和协议表现的全面保真
- ✅ 支持8类行为标签的规则事件检测
- ✅ 基于条件扩散模型的高保真数据生成
- ✅ 生成数据的三层评估体系
- ✅ 完整的可视化体系（行为样本、时间轴、特征空间）
- ✅ 支持上下行数据独立处理和生成

## 项目结构

```
NPS/
├── data/             # 数据目录
│   ├── raw/          # 原始数据
│   └── reports/      # 报告和可视化结果
├── docs/             # 文档目录
├── scripts/          # 脚本文件
├── src/              # 源代码
│   └── network_simulation/  # 主包
│       ├── condition_generation/  # 条件生成模型
│       │   ├── condition_generator.py
│       │   ├── constraint_injector.py
│       │   ├── diffusion_model.py
│       │   ├── normalization.py
│       │   ├── sample_generator.py
│       │   ├── training_data_preprocessor.py
│       │   └── training_manager.py
│       ├── data_processing/  # 数据处理
│       │   ├── data_loader.py
│       │   └── data_processor.py
│       ├── evaluation/        # 评估模块
│       │   └── evaluator.py
│       ├── feature_extraction/  # 特征提取
│       │   └── feature_extractor.py
│       ├── pattern_discovery/  # 模式发现
│       │   └── pattern_identifier.py
│       ├── utils/             # 工具函数
│       │   └── logger.py
│       └── visualization/     # 可视化模块
│           ├── base_visualizer.py
│           ├── behavior_visualizer.py
│           ├── feature_space_visualizer.py
│           ├── report_generator.py
│           ├── results_visualizer.py
│           └── visualizer.py
├── tests/            # 测试代码
└── README_zh.md      # 中文说明文档
```

## 安装说明

```bash
# 安装依赖
uv install

# 运行端到端测试
uv run python scripts/e2e_pipeline.py data/raw/20251207_223333_vXS-playback.txt
```

## 使用方法

### 快速开始

```bash
# 运行完整的端到端流程
uv run python scripts/e2e_pipeline.py data/raw/20251207_223333_vXS-playback.txt
```

### 行为标签定义

| 标签 | 名称 | 影响程度 |
|------|------|----------|
| 0 | STABLE | 最轻 |
| 1 | WEAK_BURST | 较轻 |
| 4 | HIGH_DELAY_NO_LOSS | 中等 |
| 6 | FREQUENT_FLUCTUATION | 中等 |
| 2 | STRONG_BURST | 较大 |
| 5 | HIGH_LOSS_STEADY | 较大 |
| 7 | LOW_DELAY_HIGH_LOSS | 严重 |
| 3 | INSTANT_SPIKE | 最严重 |
| -1 | INVALID | 无效 |

## 架构设计

1. **行为建模模块**：从网络数据中提取特征，使用规则事件检测引擎生成8类行为标签
2. **智能调度接口**：允许用户定义行为调度表，进行合理性验证和优化
3. **条件生成模型**：基于条件扩散模型生成高保真网络数据
4. **智能调度系统**：自动插入过渡段，确保行为对齐
5. **三层评估体系**：从统计保真度、不可区分性和动态合理性三个维度评估生成数据

## 可视化体系

1. **行为样本图**：每类行为1~3个窗口，直观展示行为特征
2. **标签时间轴图**：全局行为分布色块图，快速定位异常行为链
3. **特征空间降维图**：t-SNE/UMAP/PCA降维，展示行为在特征空间的分布

## 技术栈

- **核心框架**：PyTorch（支持 GPU/CPU/MPS）
- **数据处理**：pandas, numpy, scipy
- **机器学习**：scikit-learn
- **可视化**：matplotlib, seaborn
- **项目管理**：uv

## 许可证

MIT