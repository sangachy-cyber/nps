# 网络仿真参数生成方案 v1.0

[English Version](README.md)

## 项目概述

本项目提供了一个基于真实网络数据的高保真网络仿真参数生成方案。它可以生成10分钟的网络时序数据，精确模拟任意混合网络场景，如稳定→突发丢包→高抖动→恢复。

## 功能特性

- ✅ 生成单一网络行为（如高抖动、突发丢包）
- ✅ 通过灵活的调度表生成任意混合行为序列
- ✅ 确保统计特性、时序动态和协议表现的全面保真
- ✅ 支持多种聚类方法（GMM、K-means、HDBSCAN）
- ✅ 智能调度表验证和优化
- ✅ 生成数据的三层评估体系

## 项目结构

```
src/network_simulation/
├── cli.py                    # 命令行界面
├── data_processing/          # 原始数据处理
│   └── data_loader.py
├── feature_extraction/       # 网络数据特征提取
│   └── feature_extractor.py
├── pattern_discovery/        # 网络行为模式发现
│   └── pattern_identifier.py
├── condition_generation/     # 条件网络数据生成
│   └── condition_generator.py
├── smart_scheduling/         # 智能行为调度
│   └── scheduler.py
├── evaluation/               # 数据质量评估
│   └── evaluator.py
└── utils/                    # 工具函数
```

## 安装说明

```bash
# 安装依赖
poetry install

# 激活虚拟环境
poetry shell
```

## 使用方法

### 命令行参数

```bash
# 处理原始数据
python3 -m src.network_simulation.cli process-data --input <输入文件> --output <输出目录>

# 提取特征
python3 -m src.network_simulation.cli extract-features --input <输入文件1> <输入文件2> --output <输出目录>

# 发现行为模式（默认使用 HDBSCAN 方法）
python3 -m src.network_simulation.cli discover-patterns --input <特征文件> --raw-data <原始数据文件> --output <输出目录> --method <gmm|kmeans|hdbscan>

# 生成网络仿真数据
python3 -m src.network_simulation.cli generate-simulation --schedule <调度文件> --patterns <模式目录> --output <输出目录> --duration <秒数>

# 评估生成的仿真数据
python3 -m src.network_simulation.cli evaluate-simulation --input <仿真数据> --output <输出目录>
```

### 参数说明

| 命令 | 参数 | 说明 | 默认值 |
|------|------|------|--------|
| discover-patterns | --input, -i | 输入特征文件 | 必需 |
| discover-patterns | --raw-data, -r | 可选的原始数据文件，用于保存聚类后的原始数据段 | 无 |
| discover-patterns | --output, -o | 输出目录 | data/results/patterns |
| discover-patterns | --method, -m | 聚类方法（gmm/kmeans/hdbscan） | hdbscan |
| extract-features | --input, -i | 输入处理后的数据文件（支持多个） | 必需 |
| extract-features | --output, -o | 输出目录 | data/results/features |
| process-data | --input, -i | 输入原始数据文件 | 必需 |
| process-data | --output, -o | 输出目录 | data/processed |

### 示例调度表

```json
{
  "segments": [
    {
      "start_time": 0,
      "end_time": 120,
      "behavior_type": "stable"
    },
    {
      "start_time": 120,
      "end_time": 180,
      "behavior_type": "burst_loss"
    },
    {
      "start_time": 180,
      "end_time": 300,
      "behavior_type": "high_jitter"
    },
    {
      "start_time": 300,
      "end_time": 600,
      "behavior_type": "recovery"
    }
  ]
}
```

## 架构设计

1. **行为模式发现**：从网络数据中提取特征，使用聚类算法发现行为模式
2. **智能调度接口**：允许用户定义行为调度表，进行合理性验证和优化
3. **条件生成模型**：基于条件扩散模型、自回归生成和物理约束注入生成数据
4. **智能调度系统**：自动插入过渡段，确保行为对齐
5. **三层评估体系**：从统计保真度、不可区分性和动态合理性三个维度评估生成数据

## 技术栈

- **核心框架**：PyTorch（支持 GPU/CPU/NPU）
- **数据处理**：pandas, numpy, scipy
- **机器学习**：scikit-learn, hdbscan
- **可视化**：matplotlib, seaborn
- **项目管理**：Poetry

## 许可证

MIT
