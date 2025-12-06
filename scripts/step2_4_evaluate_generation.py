#!/usr/bin/env python3
"""
评估生成数据质量的脚本
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import sys
from pathlib import Path
import glob


# 计算统计指标
def calculate_stats(df, prefix):
    stats_dict = {
        f"{prefix}_mean": df["delay"].mean(),
        f"{prefix}_std": df["delay"].std(),
        f"{prefix}_min": df["delay"].min(),
        f"{prefix}_max": df["delay"].max(),
        f"{prefix}_median": df["delay"].median(),
        f"{prefix}_skew": stats.skew(df["delay"]),
        f"{prefix}_kurtosis": stats.kurtosis(df["delay"]),
    }

    # 丢包率统计
    stats_dict[f"{prefix}_loss_mean"] = df["loss_rate"].mean()
    stats_dict[f"{prefix}_loss_std"] = df["loss_rate"].std()
    stats_dict[f"{prefix}_loss_unique"] = len(df["loss_rate"].unique())

    return stats_dict


# 评估单组样本
def evaluate_single_sample(original_file, generated_file, output_dir):
    """评估单组生成样本的质量"""
    # 读取数据
    original_df = pd.read_csv(original_file)
    generated_df = pd.read_csv(generated_file)

    # 计算原始数据和生成数据的统计指标
    original_stats = calculate_stats(original_df, "original")
    generated_stats = calculate_stats(generated_df, "generated")

    # 合并统计结果
    stats_df = pd.DataFrame([original_stats, generated_stats])

    # 计算差异
    stats_df["delay_mean_diff"] = abs(
        stats_df["original_mean"] - stats_df["generated_mean"]
    )
    stats_df["delay_std_diff"] = abs(
        stats_df["original_std"] - stats_df["generated_std"]
    )
    stats_df["loss_mean_diff"] = abs(
        stats_df["original_loss_mean"] - stats_df["generated_loss_mean"]
    )

    # 生成评估报告
    report = []
    report.append("=" * 60)
    report.append(f"生成数据质量评估报告 - {os.path.basename(generated_file)}")
    report.append("=" * 60)
    report.append("")

    report.append("1. 时延统计对比:")
    report.append(
        f"   原始数据 - 平均值: {original_stats['original_mean']:.2f}, 标准差: {original_stats['original_std']:.2f}"
    )
    report.append(
        f"   生成数据 - 平均值: {generated_stats['generated_mean']:.2f}, 标准差: {generated_stats['generated_std']:.2f}"
    )
    report.append(
        f"   平均值差异: {abs(original_stats['original_mean'] - generated_stats['generated_mean']):.2f} ({abs(original_stats['original_mean'] - generated_stats['generated_mean']) / original_stats['original_mean'] * 100:.2f}%)"
    )
    report.append(
        f"   标准差差异: {abs(original_stats['original_std'] - generated_stats['generated_std']):.2f} ({abs(original_stats['original_std'] - generated_stats['generated_std']) / original_stats['original_std'] * 100:.2f}%)"
    )

    report.append("")
    report.append("2. 丢包率统计对比:")
    report.append(
        f"   原始数据 - 平均值: {original_stats['original_loss_mean']:.4f}, 标准差: {original_stats['original_loss_std']:.4f}"
    )
    report.append(
        f"   生成数据 - 平均值: {generated_stats['generated_loss_mean']:.4f}, 标准差: {generated_stats['generated_loss_std']:.4f}"
    )
    report.append(f"   原始数据丢包率唯一值: {original_stats['original_loss_unique']}")
    report.append(
        f"   生成数据丢包率唯一值: {generated_stats['generated_loss_unique']}"
    )

    report.append("")
    report.append("3. 分布相似性评估:")
    # KS检验
    delay_ks_stat, delay_ks_pvalue = stats.ks_2samp(
        original_df["delay"], generated_df["delay"]
    )
    report.append(
        f"   时延KS检验 - 统计量: {delay_ks_stat:.4f}, p值: {delay_ks_pvalue:.4f}"
    )
    if delay_ks_pvalue > 0.05:
        report.append("   结论: 时延分布相似 (p > 0.05)")
    else:
        report.append("   结论: 时延分布存在显著差异 (p ≤ 0.05)")

    # 生成数据有效性检查
    report.append("")
    report.append("4. 生成数据有效性检查:")
    invalid_delay = (generated_df["delay"] < 0).sum()
    invalid_loss = (
        (generated_df["loss_rate"] < 0) | (generated_df["loss_rate"] > 1)
    ).sum()
    report.append(f"   无效时延值数量: {invalid_delay}")
    report.append(f"   无效丢包率值数量: {invalid_loss}")
    report.append(
        f"   数据有效性: {'100%' if invalid_delay == 0 and invalid_loss == 0 else f'{(1 - (invalid_delay + invalid_loss) / len(generated_df)) * 100:.2f}%'}"
    )

    report.append("")
    report.append("5. 生成数据多样性评估:")
    # 计算生成数据的多样性指标
    delay_range = generated_df["delay"].max() - generated_df["delay"].min()
    loss_rate_values = sorted(generated_df["loss_rate"].unique())
    report.append(f"   时延范围: {delay_range:.2f}")
    report.append(f"   丢包率覆盖值: {loss_rate_values}")
    report.append(f"   丢包率覆盖类别数: {len(loss_rate_values)}")

    report.append("")
    report.append("=" * 60)
    report.append("评估完成!")
    report.append("=" * 60)

    # 打印评估报告
    report_text = "\n".join(report)
    print(report_text)

    # 保存评估报告
    group_name = (
        os.path.basename(generated_file).split(".")[0].split("_")[-3:]
    )  # 获取group_1_behavior_1
    group_dir_name = "_" + "_".join(group_name)
    group_output_dir = output_dir / f"group{group_dir_name}"
    group_output_dir.mkdir(parents=True, exist_ok=True)

    report_file = group_output_dir / "evaluation_report.txt"
    with open(report_file, "w") as f:
        f.write(report_text)

    return {
        "group": group_name,
        "original_stats": original_stats,
        "generated_stats": generated_stats,
        "delay_ks_stat": delay_ks_stat,
        "delay_ks_pvalue": delay_ks_pvalue,
        "invalid_delay": invalid_delay,
        "invalid_loss": invalid_loss,
        "generated_file": generated_file,
    }


def main():
    """主函数入口"""
    if len(sys.argv) < 2:
        print(
            "用法: python scripts/step2_4_evaluate_generation.py <input_generation_dir> [output_evaluation_dir]"
        )
        sys.exit(1)

    # 获取命令行参数
    input_generation_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_evaluation_dir = Path(sys.argv[2])
    else:
        # 默认输出目录
        output_evaluation_dir = input_generation_dir.parent / "evaluation"

    # 确保输出目录存在
    output_evaluation_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有原始样本和生成样本文件
    original_files = sorted(
        glob.glob(str(input_generation_dir / "original_sample_6000*.csv"))
    )
    generated_files = sorted(
        glob.glob(str(input_generation_dir / "generated_sample_6000*.csv"))
    )

    if not original_files:
        print(f"错误: 在 {input_generation_dir} 中未找到原始样本文件")
        sys.exit(1)

    if not generated_files:
        print(f"错误: 在 {input_generation_dir} 中未找到生成样本文件")
        sys.exit(1)

    if len(original_files) != len(generated_files):
        print(
            f"警告: 原始样本文件数 ({len(original_files)}) 与生成样本文件数 ({len(generated_files)}) 不匹配"
        )

    # 评估所有样本对
    all_results = []
    for original_file, generated_file in zip(original_files, generated_files):
        result = evaluate_single_sample(
            original_file, generated_file, output_evaluation_dir
        )
        all_results.append(result)

    # 生成汇总报告
    generate_summary_report(all_results, output_evaluation_dir)

    print(
        f"\n所有样本评估完成！汇总报告已保存到: {output_evaluation_dir / 'summary_report.txt'}"
    )


def generate_summary_report(results, output_dir):
    """生成所有样本的汇总评估报告"""
    report = []
    report.append("=" * 60)
    report.append("生成数据质量评估汇总报告")
    report.append("=" * 60)
    report.append("")

    # 计算平均指标
    if results:
        avg_delay_mean_diff = np.mean(
            [
                abs(
                    r["generated_stats"]["generated_mean"]
                    - r["original_stats"]["original_mean"]
                )
                for r in results
            ]
        )
        avg_delay_std_diff = np.mean(
            [
                abs(
                    r["generated_stats"]["generated_std"]
                    - r["original_stats"]["original_std"]
                )
                for r in results
            ]
        )
        avg_loss_mean_diff = np.mean(
            [
                abs(
                    r["generated_stats"]["generated_loss_mean"]
                    - r["original_stats"]["original_loss_mean"]
                )
                for r in results
            ]
        )
        avg_ks_stat = np.mean([r["delay_ks_stat"] for r in results])
        avg_ks_pvalue = np.mean([r["delay_ks_pvalue"] for r in results])

        report.append("1. 平均统计差异:")
        report.append(f"   平均时延平均值差异: {avg_delay_mean_diff:.2f}")
        report.append(f"   平均时延标准差差异: {avg_delay_std_diff:.2f}")
        report.append(f"   平均丢包率平均值差异: {avg_loss_mean_diff:.4f}")
        report.append(f"   平均KS检验统计量: {avg_ks_stat:.4f}")
        report.append(f"   平均KS检验p值: {avg_ks_pvalue:.4f}")
        report.append("")

    report.append("2. 样本评估详情:")
    for i, result in enumerate(results):
        report.append(f"   样本组 {i + 1}:")
        report.append(f"     文件名: {os.path.basename(result['generated_file'])}")
        report.append(f"     主要行为类别: {result['group'][-1]}")
        report.append(
            f"     时延平均值差异: {abs(result['original_stats']['original_mean'] - result['generated_stats']['generated_mean']):.2f}"
        )
        report.append(
            f"     丢包率平均值差异: {abs(result['original_stats']['original_loss_mean'] - result['generated_stats']['generated_loss_mean']):.4f}"
        )
        report.append(f"     KS检验p值: {result['delay_ks_pvalue']:.4f}")
        # 计算数据有效性
        generated_df = pd.read_csv(result["generated_file"])
        total_samples = len(generated_df)
        data_validity = (
            1 - (result["invalid_delay"] + result["invalid_loss"]) / total_samples
        ) * 100
        report.append(f"     数据有效性: {data_validity:.2f}%")

    report.append("")
    report.append("=" * 60)
    report.append("汇总评估完成!")
    report.append("=" * 60)

    # 保存汇总报告
    summary_report_file = output_dir / "summary_report.txt"
    with open(summary_report_file, "w") as f:
        f.write("\n".join(report))


if __name__ == "__main__":
    main()
