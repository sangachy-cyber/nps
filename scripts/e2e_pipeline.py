#!/usr/bin/env python3
"""
端到端网络行为模式发现与生成管道
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path


def clean_directory(directory: Path):
    """清理目录中的所有文件和子目录

    该函数清理指定目录中的所有内容，如果目录不存在则创建它。

    Args:
        directory: 要清理的目录路径
    """
    if directory.exists():
        print(f"正在清理目录: {directory}")
        for item in directory.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        print(f"目录不存在，正在创建: {directory}")
        directory.mkdir(parents=True, exist_ok=True)


def run_command(cmd: list, description: str):
    """运行命令并处理结果

    该函数运行指定的命令，并处理命令的执行结果，包括输出和错误信息。

    Args:
        cmd: 要执行的命令列表
        description: 命令的描述信息

    Returns:
        subprocess.CompletedProcess: 命令执行结果对象

    Raises:
        SystemExit: 如果命令执行失败，退出程序
    """
    print(f"\n{description}")
    print(f"命令: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"错误: {description} 失败")
        print(f"标准输出: {result.stdout}")
        print(f"标准错误: {result.stderr}")
        sys.exit(1)
    else:
        print(f"成功: {description} 完成")
        print(result.stdout)

    return result


def main():
    """主函数入口

    解析命令行参数，执行完整的端到端网络行为模式发现与生成管道。

    命令行参数：
        python scripts/e2e_pipeline.py <input_raw_file_or_dir>

    参数说明：
        input_raw_file_or_dir: 原始网络数据文件或目录路径
    """
    if len(sys.argv) < 2:
        print("用法: python scripts/e2e_pipeline.py <input_raw_file_or_dir>")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    # 检查输入路径是否存在
    if not input_path.exists():
        print(f"错误: 输入路径 {input_path} 不存在")
        sys.exit(1)

    # 固定输出目录结构
    base_output_dir = Path("data/results/e2e_pipeline")

    # 子目录结构
    processed_dir = base_output_dir / "processed"  # 处理后的数据
    features_dir = base_output_dir / "features"  # 提取的特征
    patterns_dir = base_output_dir / "patterns"  # 发现的行为模式
    preprocess_dir = base_output_dir / "preprocess"  # 预处理的训练数据
    train_dir = base_output_dir / "train_results"  # 训练结果
    generation_dir = base_output_dir / "generation"  # 生成的样本
    visualization_dir = base_output_dir / "visualization"  # 可视化结果

    # 清理所有输出目录
    clean_directory(base_output_dir)

    # 重新创建目录结构
    processed_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    patterns_dir.mkdir(parents=True, exist_ok=True)
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    generation_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("端到端网络行为模式发现与生成管道")
    print(f"输入路径: {input_path}")
    print(f"输出目录: {base_output_dir}")
    print("=" * 70)

    # 步骤1.1: 处理原始数据
    cmd = [
        sys.executable,
        "scripts/step1_1_process_raw.py",
        str(input_path),
        str(processed_dir),
    ]
    run_command(cmd, "步骤1.1: 处理原始数据")

    # 步骤1.2: 提取特征
    cmd = [
        sys.executable,
        "scripts/step1_2_extract_features.py",
        str(processed_dir),
        str(features_dir),
    ]
    run_command(cmd, "步骤1.2: 提取特征")

    # 步骤1.3: 发现行为模式
    cmd = [
        sys.executable,
        "scripts/step1_3_discover_patterns.py",
        str(features_dir),
        str(processed_dir),
        str(patterns_dir),
    ]
    run_command(cmd, "步骤1.3: 发现行为模式")

    # 步骤2.0: 预处理数据
    cmd = [
        sys.executable,
        "scripts/step2_0_preprocess_data.py",
        str(patterns_dir),
        str(processed_dir),
        str(preprocess_dir),
    ]
    run_command(cmd, "步骤2.0: 预处理训练数据")

    # 步骤2.1: 训练条件扩散模型
    cmd = [
        sys.executable,
        "scripts/step2_1_train_model.py",
        str(preprocess_dir),
        str(train_dir)
        # 不设置max_samples参数，使用所有样本进行训练
    ]
    run_command(cmd, "步骤2.1: 训练条件扩散模型")

    # 步骤2.2: 生成样本
    cmd = [
        sys.executable,
        "scripts/step2_2_generate_samples.py",
        str(processed_dir),
        str(patterns_dir),
        str(train_dir),
        str(generation_dir),
    ]
    run_command(cmd, "步骤2.2: 生成网络状态样本")

    # 步骤2.3: 可视化结果对比
    cmd = [
        sys.executable,
        "scripts/step2_3_visualize_results.py",
        str(generation_dir),
        str(visualization_dir),
    ]
    run_command(cmd, "步骤2.3: 可视化结果对比")

    # 步骤2.4: 评估生成样本质量
    cmd = [
        sys.executable,
        "scripts/step2_4_evaluate_generation.py",
        str(generation_dir),
        str(base_output_dir / "evaluation"),
    ]
    run_command(cmd, "步骤2.4: 评估生成样本质量")

    print("\n" + "=" * 70)
    print("端到端管道执行完成!")
    print(f"所有结果保存在: {base_output_dir}")
    print("=" * 70)


def process_single_file(
    raw_file,
    processed_dir,
    features_dir,
    patterns_dir,
    preprocess_dir,
    train_dir,
    generation_dir,
    visualization_dir,
):
    """处理单个文件的流水线

    该函数对单个原始网络数据文件执行完整的处理流水线，包括数据处理、特征提取、
    行为模式发现、模型训练、样本生成和结果可视化。

    Args:
        raw_file: 原始网络数据文件路径
        processed_dir: 处理后数据的输出目录路径
        features_dir: 特征数据的输出目录路径
        patterns_dir: 行为模式的输出目录路径
        preprocess_dir: 预处理数据的输出目录路径
        train_dir: 模型训练结果的输出目录路径
        generation_dir: 生成样本的输出目录路径
        visualization_dir: 可视化结果的输出目录路径
    """
    # 步骤1.1: 处理原始数据
    cmd = [
        sys.executable,
        "scripts/step1_1_process_raw.py",
        str(raw_file),
        str(processed_dir),
    ]
    run_command(cmd, "步骤1.1: 处理原始数据")

    # 获取处理后的文件路径
    processed_file = processed_dir / f"{raw_file.stem}_processed.csv"

    # 步骤1.2: 提取特征
    cmd = [
        sys.executable,
        "scripts/step1_2_extract_features.py",
        str(processed_file),
        str(features_dir),
    ]
    run_command(cmd, "步骤1.2: 提取特征")

    # 获取特征文件路径
    features_file = features_dir / f"{raw_file.stem}_processed_features.csv"

    # 步骤1.3: 发现行为模式
    # 使用同一个patterns_dir，覆盖之前的结果
    cmd = [
        sys.executable,
        "scripts/step1_3_discover_patterns.py",
        str(features_file),
        str(processed_file),
        str(patterns_dir),
    ]
    run_command(cmd, "步骤1.3: 发现行为模式")

    # 步骤2.0: 预处理数据
    cmd = [
        sys.executable,
        "scripts/step2_0_preprocess_data.py",
        str(patterns_dir),
        str(processed_file),
        str(preprocess_dir),
    ]
    run_command(cmd, "步骤2.0: 预处理训练数据")

    # 步骤2.1: 训练条件扩散模型
    cmd = [
        sys.executable,
        "scripts/step2_1_train_model.py",
        str(preprocess_dir),
        str(train_dir)
        # 不设置max_samples参数，使用所有样本进行训练
    ]
    run_command(cmd, "步骤2.1: 训练条件扩散模型")

    # 获取模型文件路径
    model_file = train_dir / "diffusion_model_final.pth"

    # 步骤2.2: 生成样本数据
    # 使用同一个generation_dir，覆盖之前的结果
    cmd = [
        sys.executable,
        "scripts/step2_2_generate_samples.py",
        str(processed_file),
        str(patterns_dir),
        str(model_file),
        str(generation_dir),
    ]
    run_command(cmd, "步骤2.2: 生成网络状态样本")

    # 获取生成的文件路径
    original_sample_file = generation_dir / "original_sample_6000.csv"
    generated_sample_file = generation_dir / "generated_sample_6000.csv"

    # 步骤2.3: 可视化结果
    # 使用同一个visualization_dir，覆盖之前的结果
    cmd = [
        sys.executable,
        "scripts/step2_3_visualize_results.py",
        str(generation_dir),  # 修正：传递目录而非文件
        str(visualization_dir),
    ]
    run_command(cmd, "步骤2.3: 可视化结果")

    print(f"\n文件 {raw_file} 的处理结果:")
    print(f"- 处理后的数据: {processed_file}")
    print(f"- 特征数据: {features_file}")
    print(f"- 行为模式: {patterns_dir}")
    print(f"- 训练模型: {model_file}")
    print(f"- 原始样本: {original_sample_file}")
    print(f"- 生成样本: {generated_sample_file}")
    print(f"- 可视化结果: {visualization_dir}")


if __name__ == "__main__":
    main()
