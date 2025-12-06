## 文件输入输出整改计划

### 1. 统一目录结构
```
data/
├── processed/          # 原始处理后的数据
├── results/            # 中间结果
│   ├── features/       # 提取的特征数据
│   ├── patterns/       # 行为模式发现结果
│   └── preprocess/     # 预处理数据
├── models/             # 训练好的模型
└── generated/          # 生成的样本数据
```

### 2. 默认参数设置
- 为每个脚本设置合理的默认参数
- 确保脚本可以直接执行
- 使用`argparse`库处理命令行参数

### 3. 自动清理功能
- 每个脚本执行前清理旧文件
- 只保留最新的一份文件

### 4. 脚本整改具体步骤

#### 4.1 创建统一配置文件
创建`config.py`文件，包含所有默认参数和路径配置。

#### 4.2 修改所有脚本
- 添加默认参数，确保可以直接执行
- 实现自动清理功能
- 统一文件命名和路径

#### 4.3 测试执行
- 依次执行所有脚本，验证执行流程
- 检查生成的文件是否符合要求

### 5. 脚本修改示例

以`step2_2_generate_samples.py`为例：
```python
import argparse
from pathlib import Path

# 添加命令行参数解析
parser = argparse.ArgumentParser(description="生成网络状态样本数据")
parser.add_argument("--input_processed", type=Path, default=Path("data/processed/processed_data.csv"), help="处理后的数据文件路径")
parser.add_argument("--input_patterns", type=Path, default=Path("data/results/patterns"), help="行为模式目录路径")
parser.add_argument("--input_model", type=Path, default=Path("data/models/diffusion_model"), help="训练好的模型目录路径")
parser.add_argument("--output_generation", type=Path, default=Path("data/generated"), help="生成样本的输出目录路径")
args = parser.parse_args()

# 自动清理旧文件
def cleanup_old_files(directory, pattern):
    """清理目录下的旧文件，只保留最新的一份"""
    files = sorted(directory.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    for file in files[1:]:
        file.unlink()

# 清理旧的生成文件
cleanup_old_files(args.output_generation, "original_sample_*.csv")
cleanup_old_files(args.output_generation, "generated_sample_*.csv")
```

### 6. 预期效果
- 所有脚本可以直接执行，无需手动输入所有参数
- 每个目录下只保留最新的一份文件
- 目录结构清晰，文件命名统一
- 执行流程流畅，没有路径错误
- 历史文件自动清理，避免目录混乱