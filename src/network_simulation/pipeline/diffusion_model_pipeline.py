#!/usr/bin/env python3
"""
扩散模型流程管理模块
管理step2的扩散模型流程
"""

import pandas as pd
from pathlib import Path
from network_simulation.utils.logger import get_logger

logger = get_logger(__name__)


class DiffusionModelPipeline:
    """扩散模型流程管理器

    负责管理扩散模型的完整流程，包括数据预处理、模型训练、样本生成、结果可视化和评估五个主要步骤。
    该类采用懒加载方式初始化各个组件，提高系统启动效率。

    主要组件：
    - training_data_preprocessor: 训练数据预处理器
    - training_manager: 模型训练管理器
    - generation_manager: 样本生成管理器
    - visualizer: 结果可视化器
    - evaluator: 结果评估器
    """

    def __init__(self):
        """初始化扩散模型流程管理器

        初始化流程管理器的各个组件为None，采用懒加载方式在需要时初始化。

        Examples:
            >>> from network_simulation.pipeline.diffusion_model_pipeline import DiffusionModelPipeline
            >>> pipeline = DiffusionModelPipeline()
        """
        self.training_data_preprocessor = None
        self.training_manager = None
        self.generation_manager = None
        self.visualizer = None
        self.evaluator = None

    def _init_training_data_preprocessor(self):
        """初始化训练数据预处理器

        懒加载方式初始化训练数据预处理器，如果尚未初始化则创建一个新实例。

        Examples:
            >>> pipeline = DiffusionModelPipeline()
            >>> pipeline._init_training_data_preprocessor()
            >>> print(pipeline.training_data_preprocessor is not None)  # 输出: True
        """
        if self.training_data_preprocessor is None:
            from network_simulation.condition_generation.training_data_preprocessor import (
                TrainingDataPreprocessor,
            )

            self.training_data_preprocessor = TrainingDataPreprocessor()

    def _init_training_manager(self):
        """初始化训练管理器

        懒加载方式初始化训练管理器，如果尚未初始化则引用TrainingManager类。

        Examples:
            >>> pipeline = DiffusionModelPipeline()
            >>> pipeline._init_training_manager()
            >>> print(pipeline.training_manager is not None)  # 输出: True
        """
        if self.training_manager is None:
            from network_simulation.condition_generation.training_manager import (
                TrainingManager,
            )

            self.training_manager = TrainingManager

    def _init_generation_manager(self):
        """初始化生成管理器

        懒加载方式初始化生成管理器，如果尚未初始化则引用SampleGenerator类。

        Examples:
            >>> pipeline = DiffusionModelPipeline()
            >>> pipeline._init_generation_manager()
            >>> print(pipeline.generation_manager is not None)  # 输出: True
        """
        if self.generation_manager is None:
            from network_simulation.condition_generation.sample_generator import (
                SampleGenerator,
            )

            self.generation_manager = SampleGenerator

    def _init_visualizer(self):
        """初始化可视化器

        懒加载方式初始化可视化器，如果尚未初始化则引用Visualizer类。

        Examples:
            >>> pipeline = DiffusionModelPipeline()
            >>> pipeline._init_visualizer()
            >>> print(pipeline.visualizer is not None)  # 输出: True
        """
        if self.visualizer is None:
            from network_simulation.visualization.visualizer import Visualizer

            self.visualizer = Visualizer

    def _init_evaluator(self):
        """初始化评估器

        懒加载方式初始化评估器，如果尚未初始化则创建一个新实例。

        Examples:
            >>> pipeline = DiffusionModelPipeline()
            >>> pipeline._init_evaluator()
            >>> print(pipeline.evaluator is not None)  # 输出: True
        """
        if self.evaluator is None:
            from network_simulation.evaluation.evaluator import Evaluator

            self.evaluator = Evaluator()

    def run_step2_0(
        self,
        input_patterns_dir: Path,
        input_processed_path: Path,
        output_preprocess_dir: Path,
    ):
        """运行step2.0：预处理训练数据

        预处理训练数据，包括合并处理后的数据文件、提取行为模式信息等，为后续模型训练做准备。
        支持处理单个文件或目录下的多个文件。

        Args:
            input_patterns_dir (Path): 行为模式目录路径
            input_processed_path (Path): 处理后的数据文件或目录路径
            output_preprocess_dir (Path): 预处理结果的输出目录路径

        Examples:
            >>> from network_simulation.pipeline.diffusion_model_pipeline import DiffusionModelPipeline
            >>> from pathlib import Path
            >>> pipeline = DiffusionModelPipeline()
            >>> patterns_dir = Path("output/patterns")
            >>> processed_path = Path("data/processed")
            >>> output_dir = Path("output/preprocess")
            >>> pipeline.run_step2_0(patterns_dir, processed_path, output_dir)
        """
        self._init_training_data_preprocessor()

        logger.info(
            f"开始预处理训练数据，输入模式: {input_patterns_dir}, 输入处理数据: {input_processed_path}, 输出: {output_preprocess_dir}"
        )

        # 确保输出目录存在
        output_preprocess_dir.mkdir(parents=True, exist_ok=True)

        if input_processed_path.is_file():
            # 处理单个文件
            self.training_data_preprocessor.preprocess_data(
                input_patterns_dir, input_processed_path, output_preprocess_dir
            )
        elif input_processed_path.is_dir():
            # 处理目录，合并所有处理后的文件
            processed_files = list(input_processed_path.glob("*.csv"))
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

            # 保存合并后的文件
            merged_file_path = output_preprocess_dir / "merged_processed_data.csv"
            merged_processed_df.to_csv(merged_file_path, index=False)

            logger.info("\n正在处理行为模式和合并后的数据...")
            # 预处理合并后的数据
            self.training_data_preprocessor.preprocess_data(
                input_patterns_dir, merged_file_path, output_preprocess_dir
            )

    def run_step2_1(
        self,
        input_preprocess_dir: Path,
        output_train_dir: Path,
        max_samples: int = None,
    ):
        """运行step2.1：训练条件扩散模型

        训练条件扩散模型，使用预处理后的数据进行模型训练

        Args:
            input_preprocess_dir (Path): 预处理数据目录路径
            output_train_dir (Path): 训练结果的输出目录路径
            max_samples (int, optional): 最大使用的样本数量，用于控制内存使用，默认为None

        Examples:
            >>> pipeline = DiffusionModelPipeline()
            >>> pipeline.run_step2_1(
            ...     input_preprocess_dir=Path("output/preprocess"),
            ...     output_train_dir=Path("output/train"),
            ...     max_samples=10000
            ... )
        """
        self._init_training_manager()

        logger.info(
            f"开始训练条件扩散模型，输入: {input_preprocess_dir}, 输出: {output_train_dir}, 最大样本数: {max_samples}"
        )

        # 确保输出目录存在
        output_train_dir.mkdir(parents=True, exist_ok=True)

        # 初始化训练管理器
        training_manager = self.training_manager(
            input_preprocess_dir, output_train_dir, max_samples
        )
        # 执行训练
        training_manager.train()

    def run_step2_2(
        self,
        input_patterns_dir: Path,
        input_processed_path: Path,
        input_model_dir: Path,
        output_generate_dir: Path,
        input_preprocess_dir: Path = None,
    ):
        """运行step2.2：生成样本

        使用训练好的条件扩散模型生成网络行为样本

        Args:
            input_patterns_dir (Path): 行为模式目录路径
            input_processed_path (Path): 处理后的数据文件或目录路径
            input_model_dir (Path): 训练模型目录路径
            output_generate_dir (Path): 生成结果的输出目录路径
            input_preprocess_dir (Path, optional): 预处理数据目录路径，默认为None

        Examples:
            >>> pipeline = DiffusionModelPipeline()
            >>> pipeline.run_step2_2(
            ...     input_patterns_dir=Path("output/patterns"),
            ...     input_processed_path=Path("data/processed"),
            ...     input_model_dir=Path("output/train"),
            ...     output_generate_dir=Path("output/generate"),
            ...     input_preprocess_dir=Path("output/preprocess")
            ... )
        """
        self._init_generation_manager()

        logger.info(
            f"开始生成样本，输入模式: {input_patterns_dir}, 输入处理数据: {input_processed_path}, 输入模型: {input_model_dir}, 输出: {output_generate_dir}"
        )

        # 确保输出目录存在
        output_generate_dir.mkdir(parents=True, exist_ok=True)

        # 获取模型文件路径
        model_files = list(input_model_dir.glob("diffusion_model_*.pth"))
        if not model_files:
            logger.error(f"在 {input_model_dir} 中未找到扩散模型文件")
            return

        # 使用最新的模型文件
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        input_model_path = model_files[0]
        logger.info(f"使用模型文件: {input_model_path}")

        # 获取处理后的数据文件
        if input_processed_path.is_dir():
            processed_files = list(input_processed_path.glob("*.csv"))
            if not processed_files:
                logger.error(f"在 {input_processed_path} 中未找到处理后的数据文件")
                return
            input_processed_file = processed_files[0]
        else:
            input_processed_file = input_processed_path

        # 选择参考样本：使用处理后的数据文件作为参考样本
        reference_sample_path = input_processed_file
        logger.info(f"使用参考样本: {reference_sample_path}")

        # 初始化生成管理器
        generation_manager = self.generation_manager()
        # 执行生成，使用参考样本生成模式
        generation_manager.generate_samples(
            input_processed_file,
            input_patterns_dir,
            input_model_path,
            output_generate_dir,
            input_preprocess_dir,
            generation_mode="reference_based",
            reference_sample_path=reference_sample_path,
        )

    def run_step2_3(self, input_generate_dir: Path, output_visualize_dir: Path):
        """运行step2.3：可视化结果

        对生成的样本进行可视化，生成相关的图表和报告

        Args:
            input_generate_dir (Path): 生成结果的目录路径
            output_visualize_dir (Path): 可视化结果的输出目录路径

        Examples:
            >>> pipeline = DiffusionModelPipeline()
            >>> pipeline.run_step2_3(
            ...     input_generate_dir=Path("output/generate"),
            ...     output_visualize_dir=Path("output/visualize")
            ... )
        """
        self._init_visualizer()

        logger.info(
            f"开始可视化结果，输入: {input_generate_dir}, 输出: {output_visualize_dir}"
        )

        # 确保输出目录存在
        output_visualize_dir.mkdir(parents=True, exist_ok=True)

        # 初始化可视化器
        visualizer = self.visualizer(output_visualize_dir)
        # 执行可视化
        visualizer.visualize_batch_results(input_generate_dir)

    def run_step2_4(
        self,
        input_generate_dir: Path,
        input_model_dir: Path,
        output_evaluation_dir: Path,
    ):
        """运行step2.4：评估生成结果

        评估生成样本的质量，包括统计特性、行为模式和视觉质量等方面

        Args:
            input_generate_dir (Path): 生成结果的目录路径
            input_model_dir (Path): 训练模型目录路径
            output_evaluation_dir (Path): 评估结果的输出目录路径

        Examples:
            >>> pipeline = DiffusionModelPipeline()
            >>> pipeline.run_step2_4(
            ...     input_generate_dir=Path("output/generate"),
            ...     input_model_dir=Path("output/train"),
            ...     output_evaluation_dir=Path("output/evaluation")
            ... )
        """
        self._init_evaluator()

        logger.info(
            f"开始评估生成结果，输入生成: {input_generate_dir}, 输入模型: {input_model_dir}, 输出: {output_evaluation_dir}"
        )

        # 确保输出目录存在
        output_evaluation_dir.mkdir(parents=True, exist_ok=True)

        # 执行评估
        self.evaluator.evaluate_batch_samples(input_generate_dir, output_evaluation_dir)

    def run(
        self,
        input_patterns_dir: Path,
        input_processed_path: Path,
        output_preprocess_dir: Path,
        output_train_dir: Path,
        output_generate_dir: Path,
        output_visualize_dir: Path,
        output_evaluation_dir: Path,
        max_samples: int = None,
    ):
        """运行完整的step2扩散模型流程

        执行完整的step2扩散模型流程，包括数据预处理、模型训练、样本生成、结果可视化和评估

        Args:
            input_patterns_dir (Path): 行为模式目录路径
            input_processed_path (Path): 处理后的数据文件或目录路径
            output_preprocess_dir (Path): 预处理结果的输出目录路径
            output_train_dir (Path): 训练结果的输出目录路径
            output_generate_dir (Path): 生成结果的输出目录路径
            output_visualize_dir (Path): 可视化结果的输出目录路径
            output_evaluation_dir (Path): 评估结果的输出目录路径
            max_samples (int, optional): 训练时使用的最大样本数量，默认为None

        Examples:
            >>> pipeline = DiffusionModelPipeline()
            >>> pipeline.run(
            ...     input_patterns_dir=Path("output/patterns"),
            ...     input_processed_path=Path("data/processed"),
            ...     output_preprocess_dir=Path("output/preprocess"),
            ...     output_train_dir=Path("output/train"),
            ...     output_generate_dir=Path("output/generate"),
            ...     output_visualize_dir=Path("output/visualize"),
            ...     output_evaluation_dir=Path("output/evaluation"),
            ...     max_samples=10000
            ... )
        """
        logger.info("开始运行完整的扩散模型流程")

        # 运行step2.0：预处理训练数据
        self.run_step2_0(
            input_patterns_dir, input_processed_path, output_preprocess_dir
        )

        # 运行step2.1：训练条件扩散模型
        self.run_step2_1(
            output_preprocess_dir, output_train_dir, max_samples=max_samples
        )

        # 运行step2.2：生成样本
        self.run_step2_2(
            input_patterns_dir,
            input_processed_path,
            output_train_dir,
            output_generate_dir,
            output_preprocess_dir,
        )

        # 运行step2.3：可视化结果
        self.run_step2_3(output_generate_dir, output_visualize_dir)

        # 运行step2.4：评估生成结果
        self.run_step2_4(output_generate_dir, output_train_dir, output_evaluation_dir)

        logger.info("扩散模型流程运行完成")
