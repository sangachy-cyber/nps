#!/usr/bin/env python3
"""
网络模拟参数生成方案 CLI
"""

import argparse
import sys
from pathlib import Path

from network_simulation.data_processing.data_loader import DataLoader
from network_simulation.feature_extraction.feature_extractor import FeatureExtractor
from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
from network_simulation.condition_generation.condition_generator import (
    ConditionGenerator,
)
from network_simulation.smart_scheduling.scheduler import SmartScheduler
from network_simulation.evaluation.evaluator import Evaluator
from network_simulation.visualization.visualizer import Visualizer


def main():
    """CLI的主入口点

    该函数解析命令行参数，并根据不同的命令调用相应的功能模块，
    包括数据处理、特征提取、行为模式发现、模拟数据生成和评估等功能。

    可用命令：
        process-data: 处理原始网络数据文件
        extract-features: 从处理后的数据中提取网络特征
        discover-patterns: 发现网络行为模式
        generate-simulation: 生成网络模拟数据
        evaluate-simulation: 评估生成的模拟数据

    示例用法：
        python -m network_simulation.cli process-data -i data/raw/20251203_230356_b6x-playback.txt -o data/processed
        python -m network_simulation.cli extract-features -i data/processed/processed_data.csv -o data/results/features
        python -m network_simulation.cli discover-patterns -i data/results/features/features.csv -o data/results/patterns

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="网络模拟参数生成方案 v1.0")

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 命令: process-data
    process_data_parser = subparsers.add_parser(
        "process-data", help="处理原始网络数据文件"
    )
    process_data_parser.add_argument(
        "--input", "-i", required=True, type=Path, help="输入文件路径"
    )
    process_data_parser.add_argument(
        "--output",
        "-o",
        default=Path("data/processed"),
        type=Path,
        help="输出目录",
    )

    # 命令: extract-features
    extract_features_parser = subparsers.add_parser(
        "extract-features", help="从处理后的数据中提取网络特征"
    )
    extract_features_parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        nargs="+",
        help="输入处理后的数据文件（允许多个）",
    )
    extract_features_parser.add_argument(
        "--output",
        "-o",
        default=Path("data/results/features"),
        type=Path,
        help="输出目录",
    )

    # 命令: discover-patterns
    discover_patterns_parser = subparsers.add_parser(
        "discover-patterns", help="发现网络行为模式"
    )
    discover_patterns_parser.add_argument(
        "--input", "-i", required=True, type=Path, help="输入特征文件"
    )
    discover_patterns_parser.add_argument(
        "--raw-data",
        "-r",
        type=Path,
        help="可选的原始数据文件，用于保存聚类后的分段",
    )
    discover_patterns_parser.add_argument(
        "--output",
        "-o",
        default=Path("data/results/patterns"),
        type=Path,
        help="输出目录",
    )
    discover_patterns_parser.add_argument(
        "--method",
        "-m",
        choices=["gmm", "kmeans", "hdbscan"],
        default="hdbscan",
        help="使用的聚类方法",
    )

    # 命令: generate-simulation
    generate_simulation_parser = subparsers.add_parser(
        "generate-simulation", help="生成网络模拟数据"
    )
    generate_simulation_parser.add_argument(
        "--schedule", "-s", required=True, type=Path, help="行为调度文件"
    )
    generate_simulation_parser.add_argument(
        "--patterns",
        "-p",
        required=True,
        type=Path,
        help="发现的模式目录",
    )
    generate_simulation_parser.add_argument(
        "--output",
        "-o",
        default=Path("data/results/simulations"),
        type=Path,
        help="输出目录",
    )
    generate_simulation_parser.add_argument(
        "--duration", "-d", type=int, default=600, help="模拟持续时间（秒）"
    )

    # 命令: evaluate-simulation
    evaluate_simulation_parser = subparsers.add_parser(
        "evaluate-simulation", help="评估生成的模拟数据"
    )
    evaluate_simulation_parser.add_argument(
        "--input", "-i", required=True, type=Path, help="输入模拟数据文件"
    )
    evaluate_simulation_parser.add_argument(
        "--output",
        "-o",
        default=Path("data/results/evaluations"),
        type=Path,
        help="输出目录",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "process-data":
            data_loader = DataLoader()
            data = data_loader.load(args.input)
            processed_data = data_loader.preprocess(data)
            data_loader.save(processed_data, args.output / "processed_data.csv")
            print(f"处理后的数据已保存到 {args.output / 'processed_data.csv'}")

        elif args.command == "extract-features":
            feature_extractor = FeatureExtractor()
            all_features = []

            # 处理所有输入文件
            for input_file in args.input:
                processed_data = feature_extractor.load_data(input_file)
                features = feature_extractor.extract(processed_data)
                all_features.append(features)

            # 合并所有文件的特征
            if all_features:
                # 使用 pandas 合并特征
                import pandas as pd

                combined_features = pd.concat(all_features, ignore_index=True)
                feature_extractor.save(combined_features, args.output / "features.csv")
                print(
                    f"从 {len(args.input)} 个文件中提取特征，并保存到 {args.output / 'features.csv'}"
                )
            else:
                print("未从输入文件中提取到特征。")

        elif args.command == "discover-patterns":
            pattern_identifier = PatternIdentifier(method=args.method)
            features = pattern_identifier.load_features(args.input)

            # 如果提供了原始数据，加载它
            raw_data = None
            if args.raw_data:
                import pandas as pd

                raw_data = pd.read_csv(args.raw_data, parse_dates=["timestamp"])

            patterns = pattern_identifier.identify(features, raw_data)
            pattern_identifier.save(patterns, args.output, features)

            # 使用 Visualizer 生成 HTML 报告
            print("正在生成 HTML 报告...")
            visualizer = Visualizer(args.output)
            # 提取用于可视化的特征列
            X = features[pattern_identifier.feature_columns].values
            visualizer.generate_html_report(
                patterns, X, pattern_identifier.feature_columns, raw_data, features
            )

            print(f"行为模式已发现并保存到 {args.output}")

        elif args.command == "generate-simulation":
            scheduler = SmartScheduler()
            schedule = scheduler.load_schedule(args.schedule)
            patterns = scheduler.load_patterns(args.patterns)

            generator = ConditionGenerator()
            simulation_data = generator.generate(
                schedule, patterns, duration=args.duration
            )

            generator.save(simulation_data, args.output / "simulation_data.csv")
            print(f"模拟数据已生成并保存到 {args.output / 'simulation_data.csv'}")

        elif args.command == "evaluate-simulation":
            evaluator = Evaluator()
            simulation_data = evaluator.load_data(args.input)
            evaluation_results = evaluator.evaluate(simulation_data)
            evaluator.save(evaluation_results, args.output)
            print(f"评估已完成，结果保存到 {args.output}")

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
