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
        对于目录情况，会先合并所有处理后的文件，然后提取特征并保存。

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
            # 添加文件标识
            features_df["file_id"] = input_processed_path.stem
            # 保存特征数据
            features_output_file = (
                output_features_dir / f"{input_processed_path.stem}_features.csv"
            )
            self.feature_extractor.save(features_df, features_output_file)
        elif input_processed_path.is_dir():
            # 处理目录，合并所有处理后的文件
            processed_files = sorted(
                list(input_processed_path.glob("*.csv")), key=lambda x: x.name
            )
            if not processed_files:
                logger.warning(f"在 {input_processed_path} 中未找到 .csv 文件")
                return

            # 合并所有处理后的文件
            all_processed_df = []
            for processed_file in processed_files:
                df = pd.read_csv(processed_file, parse_dates=["timestamp"])
                all_processed_df.append(df)

            # 合并为一个DataFrame
            merged_processed_df = pd.concat(all_processed_df, ignore_index=True)
            logger.info(
                f"合并了 {len(processed_files)} 个处理后的文件，总样本数: {len(merged_processed_df)}"
            )

            # 检查合并后的数据是否为空
            if merged_processed_df.empty:
                logger.error("No valid data found after merging input files.")
                return

            # 保存合并后的文件，供后续使用
            merged_processed_file = (
                input_processed_path / self.MERGED_PROCESSED_FILENAME
            )
            merged_processed_df.to_csv(merged_processed_file, index=False)
            logger.info(f"合并后的处理数据已保存到: {merged_processed_file}")

            # 提取特征
            features_df = self.feature_extractor.extract(merged_processed_df)
            # 添加文件标识
            features_df["file_id"] = "merged"
            # 保存特征数据
            features_file = output_features_dir / "merged_processed_data_features.csv"
            self.feature_extractor.save(features_df, features_file)

            # 保存合并后的特征文件
            merged_features_file = output_features_dir / self.MERGED_FEATURES_FILENAME
            import shutil

            shutil.copy2(features_file, merged_features_file)
            logger.info(f"已将合并数据的特征保存到 {merged_features_file}")

    def run_step1_3(
        self,
        input_features_path: Path,
        input_processed_path: Path,
        output_patterns_dir: Path,
    ):
        """运行step1.3：发现行为模式

        基于提取的特征数据发现网络行为模式，支持处理单个文件对或目录情况。
        对于目录情况，会使用合并后的特征文件和处理数据文件。

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
            # 使用合并后的特征文件
            merged_features_file = input_features_path / self.MERGED_FEATURES_FILENAME
            if not merged_features_file.exists():
                logger.error(
                    f"在 {input_features_path} 中未找到 {self.MERGED_FEATURES_FILENAME}"
                )
                logger.error("请先运行特征提取脚本生成合并特征文件")
                return

            # 检查是否存在合并后的处理数据
            merged_processed_file = (
                input_processed_path / self.MERGED_PROCESSED_FILENAME
            )
            if not merged_processed_file.exists():
                logger.error(
                    f"在 {input_processed_path} 中未找到 {self.MERGED_PROCESSED_FILENAME}"
                )
                logger.error("请先运行数据处理脚本生成合并处理数据")
                return

            # 使用合并后的特征和处理数据
            self.pattern_identifier.identify_and_save(
                merged_features_file, merged_processed_file, output_patterns_dir
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
