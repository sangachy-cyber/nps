#!/usr/bin/env python3
"""
行为发现流程管理模块
管理step1的行为发现流程
"""

import pandas as pd
from pathlib import Path
from network_simulation.utils.logger import get_logger

logger = get_logger(__name__)


class BehaviorDiscoveryPipeline:
    """行为发现流程管理器

    负责管理网络行为发现的完整流程，包括特征提取、模式识别和结果评估三个主要步骤。
    该类采用懒加载方式初始化各个组件，提高系统启动效率。

    输出文件名常量：
    - MERGED_PROCESSED_FILENAME: 合并后的处理数据文件名
    - MERGED_FEATURES_FILENAME: 合并后的特征数据文件名
    - LABELED_WINDOWS_FILENAME: 标记窗口数据文件名
    - EVALUATION_REPORT_FILENAME: 评估报告文件名
    """

    # 输出文件名常量
    MERGED_PROCESSED_FILENAME = "merged_processed_data.csv"
    MERGED_FEATURES_FILENAME = "merged_features.csv"
    LABELED_WINDOWS_FILENAME = "labeled_windows.csv"
    EVALUATION_REPORT_FILENAME = "evaluation_report.json"

    def __init__(self):
        """初始化行为发现流程管理器

        初始化流程管理器的各个组件为None，采用懒加载方式在需要时初始化。

        Examples:
            >>> from network_simulation.pipeline.behavior_discovery_pipeline import BehaviorDiscoveryPipeline
            >>> pipeline = BehaviorDiscoveryPipeline()
        """
        self.feature_extractor = None
        self.pattern_identifier = None
        self.evaluator = None

    def _init_feature_extractor(self):
        """初始化特征提取器

        懒加载方式初始化特征提取器，如果尚未初始化则创建一个新实例。

        Examples:
            >>> pipeline = BehaviorDiscoveryPipeline()
            >>> pipeline._init_feature_extractor()
            >>> print(pipeline.feature_extractor is not None)  # 输出: True
        """
        if self.feature_extractor is None:
            from network_simulation.pattern_discovery.feature_extractor import (
                FeatureExtractor,
            )

            self.feature_extractor = FeatureExtractor()

    def _init_pattern_identifier(self):
        """初始化模式识别器

        懒加载方式初始化模式识别器，如果尚未初始化则创建一个新实例，使用规则方法和配置文件中的配置。

        Examples:
            >>> pipeline = BehaviorDiscoveryPipeline()
            >>> pipeline._init_pattern_identifier()
            >>> print(pipeline.pattern_identifier is not None)  # 输出: True
        """
        if self.pattern_identifier is None:
            from network_simulation.pattern_discovery.pattern_identifier import (
                PatternIdentifier,
            )

            # 从配置文件加载 PatternIdentifier 的配置
            from config import PATTERN_IDENTIFIER_CONFIG

            self.pattern_identifier = PatternIdentifier(
                method="rule", config=PATTERN_IDENTIFIER_CONFIG
            )

    def _init_evaluator(self):
        """初始化评估器

        懒加载方式初始化评估器，如果尚未初始化则创建一个新实例。

        Examples:
            >>> pipeline = BehaviorDiscoveryPipeline()
            >>> pipeline._init_evaluator()
            >>> print(pipeline.evaluator is not None)  # 输出: True
        """
        if self.evaluator is None:
            from network_simulation.evaluation.evaluator import Evaluator

            self.evaluator = Evaluator()

    def run_step1_2(self, input_processed_path: Path, output_features_dir: Path):
        """运行step1.2：提取特征

        从处理后的数据中提取网络行为特征，支持处理单个文件或目录下的多个文件。
        对于目录情况，会直接处理所有文件，合并特征后保存。

        Args:
            input_processed_path (Path): 处理后的数据文件或目录路径
            output_features_dir (Path): 特征数据的输出目录路径

        Examples:
            >>> from network_simulation.pipeline.behavior_discovery_pipeline import BehaviorDiscoveryPipeline
            >>> from pathlib import Path
            >>> pipeline = BehaviorDiscoveryPipeline()
            >>> input_path = Path("data/processed")
            >>> output_dir = Path("output/features")
            >>> pipeline.run_step1_2(input_path, output_dir)
        """
        self._init_feature_extractor()

        logger.info(
            f"开始提取特征，输入: {input_processed_path}, 输出: {output_features_dir}"
        )

        # 确保输出目录存在
        output_features_dir.mkdir(parents=True, exist_ok=True)

        if input_processed_path.is_file():
            # 处理单个文件
            df = pd.read_csv(input_processed_path, parse_dates=["timestamp"])
            features_df = self.feature_extractor.extract(df)
            # 保存特征数据
            features_output_file = (
                output_features_dir / f"{input_processed_path.stem}_features.csv"
            )
            self.feature_extractor.save(features_df, features_output_file)

            # 保存合并后的特征文件（单个文件时也生成合并特征）
            merged_features_file = output_features_dir / self.MERGED_FEATURES_FILENAME
            self.feature_extractor.save(features_df, merged_features_file)
            logger.info(f"已将特征保存到 {merged_features_file}")
        elif input_processed_path.is_dir():
            # 处理目录，为每个文件提取特征并合并
            processed_files = sorted(
                list(input_processed_path.glob("*.csv")), key=lambda x: x.name
            )
            if not processed_files:
                logger.warning(f"在 {input_processed_path} 中未找到 .csv 文件")
                return

            # 为每个文件提取特征，不进行相关性移除
            all_features = []
            for processed_file in processed_files:
                logger.info(f"正在处理文件: {processed_file.name}")
                df = pd.read_csv(processed_file, parse_dates=["timestamp"])
                # 提取原始特征，不进行相关性移除，不进行归一化
                features_df = self.feature_extractor.extract(
                    df, remove_correlated=False, normalize=False
                )  # 不进行相关性移除，不进行归一化，后续统一归一化
                all_features.append(features_df)
                logger.info(
                    f"文件 {processed_file.stem} 特征提取完成，特征数: {len(features_df)}"
                )

            # 合并所有特征
            if not all_features:
                logger.error("No valid features found after processing all files.")
                return

            merged_features_df = pd.concat(all_features, ignore_index=True)
            logger.info(
                f"合并了 {len(processed_files)} 个文件的特征，总特征数: {len(merged_features_df)}"
            )

            # 对合并后的特征统一移除高度相关的特征
            merged_features_df = (
                self.feature_extractor.remove_highly_correlated_features(
                    merged_features_df, correlation_threshold=0.8
                )
            )
            logger.info(
                f"统一移除高度相关特征后，剩余特征数: {len([col for col in merged_features_df.columns if col.startswith('feat_')])}"
            )

            # 保存合并后的特征文件
            merged_features_file = output_features_dir / self.MERGED_FEATURES_FILENAME
            self.feature_extractor.save(merged_features_df, merged_features_file)
            logger.info(f"已将合并特征保存到 {merged_features_file}")

            # 对每个文件的特征统一进行相关性移除并保存
            for i, (processed_file, raw_features_df) in enumerate(
                zip(processed_files, all_features)
            ):
                # 只保留与合并特征相同的列
                aligned_features_df = raw_features_df[
                    raw_features_df.columns.intersection(merged_features_df.columns)
                ]

                # 保存单个文件的特征
                single_features_file = (
                    output_features_dir / f"{processed_file.stem}_features.csv"
                )
                self.feature_extractor.save(aligned_features_df, single_features_file)
                logger.info(f"已保存对齐后的特征到 {single_features_file}")

    def run_step1_3(
        self,
        input_features_path: Path,
        input_processed_path: Path,
        output_patterns_dir: Path,
    ):
        """运行step1.3：发现行为模式

        基于提取的特征数据发现网络行为模式，支持处理单个文件对或目录情况。
        对于目录情况，会使用合并后的特征文件进行分析。

        Args:
            input_features_path (Path): 特征数据文件或目录路径
            input_processed_path (Path): 处理后的数据文件或目录路径
            output_patterns_dir (Path): 行为模式结果的输出目录路径

        Examples:
            >>> from network_simulation.pipeline.behavior_discovery_pipeline import BehaviorDiscoveryPipeline
            >>> from pathlib import Path
            >>> pipeline = BehaviorDiscoveryPipeline()
            >>> features_path = Path("output/features")
            >>> processed_path = Path("data/processed")
            >>> output_dir = Path("output/patterns")
            >>> pipeline.run_step1_3(features_path, processed_path, output_dir)
        """
        self._init_pattern_identifier()

        logger.info(
            f"开始发现行为模式，输入特征: {input_features_path}, 输入处理数据: {input_processed_path}, 输出: {output_patterns_dir}"
        )

        # 确保输出目录存在
        output_patterns_dir.mkdir(parents=True, exist_ok=True)

        # 处理目录情况
        if input_features_path.is_dir() and input_processed_path.is_dir():
            # 获取所有特征文件和处理后的数据文件
            feature_files = sorted(
                list(input_features_path.glob("*.csv")), key=lambda x: x.name
            )
            processed_files = sorted(
                list(input_processed_path.glob("*.csv")), key=lambda x: x.name
            )

            # 排除merged_features.csv，处理每个原始文件
            feature_files = [
                f for f in feature_files if f.name != self.MERGED_FEATURES_FILENAME
            ]

            if not feature_files or not processed_files:
                logger.error("在输入目录中未找到足够的文件")
                return

            # 处理每个文件对
            for feature_file, processed_file in zip(feature_files, processed_files):
                # 确保文件名匹配
                feature_name = feature_file.stem.replace("_features", "")
                processed_name = processed_file.stem

                if feature_name == processed_name:
                    logger.info(f"处理文件对: {feature_name}")
                    self.pattern_identifier.identify_and_save(
                        feature_file, processed_file, output_patterns_dir
                    )
                else:
                    logger.warning(
                        f"文件名不匹配: {feature_file.name} 和 {processed_file.name}，跳过此文件对"
                    )
        elif input_features_path.is_file() and input_processed_path.is_file():
            # 处理单个文件对
            self.pattern_identifier.identify_and_save(
                input_features_path, input_processed_path, output_patterns_dir
            )
        else:
            logger.error(
                f"输入类型不匹配 - 特征输入: {input_features_path.is_file() and '文件' or '目录'}, 处理后数据输入: {input_processed_path.is_file() and '文件' or '目录'}"
            )
            logger.error("请确保两个输入都是文件或都是目录")

    def run_step1_4(
        self, input_features_path: Path, input_patterns_dir: Path, output_eval_dir: Path
    ):
        """运行step1.4：评估行为发现结果

        评估行为发现结果的质量，包括模式分离度、稳定性等多个指标，支持处理单个文件或目录情况。
        对于目录情况，会使用合并后的特征文件进行评估。

        Args:
            input_features_path (Path): 特征数据文件或目录路径
            input_patterns_dir (Path): 行为模式结果目录路径
            output_eval_dir (Path): 评估结果的输出目录路径

        Examples:
            >>> from network_simulation.pipeline.behavior_discovery_pipeline import BehaviorDiscoveryPipeline
            >>> from pathlib import Path
            >>> pipeline = BehaviorDiscoveryPipeline()
            >>> features_path = Path("output/features")
            >>> patterns_dir = Path("output/patterns")
            >>> output_dir = Path("output/evaluation")
            >>> pipeline.run_step1_4(features_path, patterns_dir, output_dir)
        """
        self._init_evaluator()

        logger.info(
            f"开始评估行为发现结果，输入特征: {input_features_path}, 输入模式: {input_patterns_dir}, 输出: {output_eval_dir}"
        )

        # 确保输出目录存在
        output_eval_dir.mkdir(parents=True, exist_ok=True)

        if input_features_path.is_dir():
            # 使用合并后的特征文件
            merged_features_file = input_features_path / self.MERGED_FEATURES_FILENAME
            if not merged_features_file.exists():
                logger.error(
                    f"在 {input_features_path} 中未找到 {self.MERGED_FEATURES_FILENAME}"
                )
                logger.error("请先运行特征提取脚本生成合并特征文件")
                return

            self.evaluator.evaluate_patterns(
                merged_features_file, input_patterns_dir, output_eval_dir
            )
        elif input_features_path.is_file():
            # 处理单个文件
            self.evaluator.evaluate_patterns(
                input_features_path, input_patterns_dir, output_eval_dir
            )

    def run(
        self,
        input_processed_path: Path,
        output_features_dir: Path,
        output_patterns_dir: Path,
        output_eval_dir: Path,
    ):
        """运行完整的step1行为发现流程

        运行完整的网络行为发现流程，包括三个主要步骤：
        1. 特征提取 (step1.2)
        2. 行为模式发现 (step1.3)
        3. 结果评估 (step1.4)

        Args:
            input_processed_path (Path): 处理后的数据文件或目录路径
            output_features_dir (Path): 特征数据的输出目录路径
            output_patterns_dir (Path): 行为模式结果的输出目录路径
            output_eval_dir (Path): 评估结果的输出目录路径

        Examples:
            >>> from network_simulation.pipeline.behavior_discovery_pipeline import BehaviorDiscoveryPipeline
            >>> from pathlib import Path
            >>> pipeline = BehaviorDiscoveryPipeline()
            >>> input_path = Path("data/processed")
            >>> features_dir = Path("output/features")
            >>> patterns_dir = Path("output/patterns")
            >>> eval_dir = Path("output/evaluation")
            >>> pipeline.run(input_path, features_dir, patterns_dir, eval_dir)
        """
        logger.info("开始运行完整的行为发现流程")

        # 运行step1.2：提取特征
        self.run_step1_2(input_processed_path, output_features_dir)

        # 运行step1.3：发现行为模式
        self.run_step1_3(output_features_dir, input_processed_path, output_patterns_dir)

        # 运行step1.4：评估行为发现结果
        self.run_step1_4(output_features_dir, output_patterns_dir, output_eval_dir)

        logger.info("行为发现流程运行完成")
