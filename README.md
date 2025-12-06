# NPS (Network Performance Simulation)

## 项目概述

NPS是一个用于网络性能模拟的开源框架，基于条件扩散模型生成高保真网络状态数据，用于测试和验证网络协议、算法和系统的性能。该框架能够模拟各种网络条件下的延迟、丢包等网络状态，帮助开发者在不同网络环境下评估系统性能。

## 核心功能

- **网络状态数据预处理**：将原始网络数据转换为适合模型训练的格式
- **行为模式发现与聚类**：自动发现网络行为模式并进行聚类分析
- **条件扩散模型训练**：基于真实数据训练条件扩散模型
- **网络状态样本生成**：根据行为模式和调度生成高质量网络状态样本
- **生成样本质量评估**：评估生成样本的统计保真度和区分度
- **可视化对比分析**：直观展示生成样本与真实样本的对比结果

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
├── project_rules.md       # 项目原则与规范
├── README.md              # 项目说明文档
├── src/                   # 源代码目录
│   └── network_simulation/ # 网络模拟核心代码
│       ├── condition_generation/ # 条件生成模块
│       ├── pattern_discovery/    # 模式发现模块
│       ├── feature_extraction/   # 特征提取模块
│       ├── evaluation/           # 评估模块
│       ├── visualization/        # 可视化模块
│       └── utils/                # 工具模块
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
uv install
```

## 快速开始

### 端到端运行

使用端到端脚本运行完整流程：

```bash
# 使用默认配置运行完整流程
uv run python scripts/e2e_pipeline.py data/raw

# 使用优化版本的端到端脚本
uv run python scripts/optimized_e2e_pipeline.py data/raw
```

### 分步运行

1. **步骤1：处理原始数据**
   ```bash
   uv run python scripts/step1_1_process_raw.py data/raw
   ```

2. **步骤2：提取特征**
   ```bash
   uv run python scripts/step1_2_extract_features.py data/processed
   ```

3. **步骤3：发现行为模式**
   ```bash
   uv run python scripts/step1_3_discover_patterns.py data/results/features
   ```

4. **步骤4：预处理训练数据**
   ```bash
   uv run python scripts/step2_0_preprocess_data.py data/results/patterns
   ```

5. **步骤5：训练条件扩散模型**
   ```bash
   uv run python scripts/step2_1_train_model.py data/results/preprocess
   ```

6. **步骤6：生成样本**
   ```bash
   uv run python scripts/step2_2_generate_samples.py data/processed
   ```

7. **步骤7：可视化结果**
   ```bash
   uv run python scripts/step2_3_visualize_results.py data/results/generation
   ```

## 配置说明

所有配置参数都集中在`config.py`文件中，主要包括：

- **目录结构配置**：定义项目数据、模型、结果等目录路径
- **默认文件名配置**：定义各步骤输出文件的默认名称
- **模型参数配置**：定义扩散模型的结构和训练参数
- **HDBSCAN参数配置**：定义行为模式发现的聚类参数
- **窗口配置**：定义特征提取的窗口大小和滑动步长
- **生成样本配置**：定义生成样本的数量和质量控制参数

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
- 日志规范
- 可视化规范
- 资源管理规范
- 安全规范
- 版本控制规范

## 日志使用规范

- 所有代码必须使用`src/network_simulation/utils/logger.py`中定义的`get_logger`函数获取日志记录器
- 避免直接使用`print()`语句，应使用`logger.info()`、`logger.debug()`等日志方法
- 日志内容应清晰、准确，包含足够的上下文信息
- 日志级别使用指南：
  - `DEBUG`：用于调试信息，如变量值、函数调用细节等
  - `INFO`：用于正常运行信息，如功能执行开始/完成、重要操作结果等
  - `WARNING`：用于警告信息，如非致命错误、参数调整等
  - `ERROR`：用于错误信息，如致命错误、配置错误等

## 可视化规范

- 所有可视化代码必须添加跨平台中文支持配置
- 可视化输出使用高分辨率（DPI ≥ 300）
- 丢包率相关的可视化必须将丢包率范围限制在0-1之间
- 可视化文件命名遵循`{功能}_{描述}_{日期时间戳}.png`格式

## 测试说明

### 运行测试用例

```bash
# 运行所有测试用例
uv run pytest tests/

# 运行特定模块的测试用例
uv run pytest tests/test_diffusion_model.py

# 运行测试并生成覆盖率报告
uv run pytest tests/ --cov=src
```

### 测试要求

- 所有核心功能必须有测试用例覆盖
- 测试用例应覆盖正常情况、边界情况和异常情况
- 测试用例应独立运行，不依赖外部资源
- 测试用例应具有清晰的断言和测试目的

## 贡献指南

1. **Fork仓库**：在GitHub上Fork本仓库到您自己的账户
2. **创建功能分支**：基于main分支创建功能分支，命名格式为`feature/xxx`或`fix/xxx`
3. **提交代码**：
   - 确保代码符合项目规范
   - 编写相应的测试用例
   - 运行测试确保所有用例通过
   - 编写清晰的提交信息
4. **创建Pull Request**：向本仓库的main分支提交Pull Request，描述您的修改内容和理由

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过以下方式联系：

- **GitHub Issues**：通过GitHub Issues提交问题或建议
- **Email**：如有紧急问题，可通过项目仓库中的联系方式发送邮件

## 版本历史

### v1.0.0
- 初始版本发布
- 支持基本的网络状态数据生成和评估
- 实现了条件扩散模型训练和样本生成

## 常见问题解答

### Q: 如何调整生成样本的数量和质量？
A: 可以通过修改`config.py`文件中的`GENERATION_CONFIG`参数来调整生成样本的数量和质量。

### Q: 如何添加新的网络行为模式？
A: 可以通过扩展`pattern_discovery`模块来添加新的网络行为模式发现算法。

### Q: 如何调整模型训练参数？
A: 可以通过修改`config.py`文件中的`MODEL_CONFIG`参数来调整模型训练参数。

### Q: 如何评估生成样本的质量？
A: 可以使用`scripts/step3_1_evaluate_samples.py`脚本评估生成样本的质量，该脚本会生成详细的评估报告。
