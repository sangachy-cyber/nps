# NPS (Network Performance Simulation)

## 项目概述

NPS是一个用于网络性能模拟的开源框架，基于扩散模型生成网络状态数据，用于测试和验证网络协议、算法和系统的性能。

## 核心功能

- 网络状态数据预处理
- 行为模式发现与聚类
- 条件扩散模型训练
- 网络状态样本生成
- 生成样本质量评估
- 可视化对比分析

## 技术栈

- Python 3.8+
- PyTorch
- HDBSCAN
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn

## 目录结构

```
项目根目录/
├── config.py              # 统一配置文件
├── README.md              # 项目说明文档
├── src/                   # 源代码目录
│   └── network_simulation/ # 网络模拟核心代码
│       ├── condition_generation/ # 条件生成模块
│       ├── pattern_discovery/    # 模式发现模块
│       └── feature_extraction/   # 特征提取模块
├── scripts/               # 执行脚本
├── tests/                 # 测试用例
├── data/                  # 数据目录
│   ├── processed/         # 原始处理后的数据
│   ├── results/           # 中间结果
│   │   ├── features/      # 提取的特征数据
│   │   ├── patterns/      # 行为模式发现结果
│   │   └── preprocess/    # 预处理数据
│   ├── models/            # 训练好的模型
│   └── generated/         # 生成的样本数据
└── tmp/                   # 临时文件目录（自动清理）
```

## 安装方法

### 1. 克隆仓库

```bash
git clone git@github.com:sangachy-cyber/nps.git
cd nps
```

### 2. 安装依赖

使用uv作为包管理器：

```bash
# 安装uv（如果尚未安装）
pip install uv

# 创建虚拟环境
uv venv

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
uv install -r requirements.txt
```

## 快速开始

### 端到端运行

使用端到端脚本运行完整流程：

```bash
python scripts/e2e_pipeline.py <input_raw_file_or_dir>
```

### 分步运行

1. **步骤1：处理原始数据**
   ```bash
   python scripts/step1_1_process_raw.py data/raw
   ```

2. **步骤2：提取特征**
   ```bash
   python scripts/step1_2_extract_features.py data/processed
   ```

3. **步骤3：发现行为模式**
   ```bash
   python scripts/step1_3_discover_patterns.py data/results/features
   ```

4. **步骤4：预处理训练数据**
   ```bash
   python scripts/step2_0_preprocess_data.py data/results/patterns
   ```

5. **步骤5：训练条件扩散模型**
   ```bash
   python scripts/step2_1_train_model.py data/results/preprocess
   ```

6. **步骤6：生成样本**
   ```bash
   python scripts/step2_2_generate_samples.py data/processed
   ```

7. **步骤7：可视化结果**
   ```bash
   python scripts/step2_3_visualize_results.py data/results/generation
   ```

## 配置说明

所有配置参数都集中在`config.py`文件中，主要包括：

- 目录结构配置
- 默认文件名配置
- 模型参数配置
- 训练参数配置
- HDBSCAN参数配置
- 清理配置
- 生成样本配置
- 窗口配置

## 项目规则

请参考`project_rules.md`文件了解详细的项目原则与规范，包括：

- 文件目录结构规范
- 文件命名规范
- 输入输出原则
- 脚本执行规范
- 临时文件处理规范
- 测试规范
- 代码质量规范
- 中文谷歌风格注释规范
- 版本控制规范
- 资源管理规范
- 安全规范
- 部署规范
- 维护规范
- 协作规范
- 性能规范
- 可扩展性规范
- 可视化规范
- 日志规范

## 贡献指南

1. Fork仓库
2. 创建功能分支
3. 提交代码
4. 创建Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过GitHub Issues提交。
