#!/usr/bin/env python3
"""
流程管理器模块
管理完整的端到端流程
"""

from pathlib import Path
from network_simulation.utils.logger import get_logger
from network_simulation.pipeline.behavior_discovery_pipeline import (
    BehaviorDiscoveryPipeline,
)
from network_simulation.pipeline.diffusion_model_pipeline import DiffusionModelPipeline

logger = get_logger(__name__)


class PipelineManager:
    """整体流程管理器
    负责协调和管理完整的端到端流程，包括行为发现和扩散模型流程
    """

    def __init__(self):
        """初始化整体流程管理器

        初始化行为发现流程和扩散模型流程的实例
        """
        self.behavior_discovery_pipeline = BehaviorDiscoveryPipeline()
        self.diffusion_model_pipeline = DiffusionModelPipeline()

    def run_behavior_discovery(
        self,
        input_processed_path: Path,
        output_features_dir: Path,
        output_patterns_dir: Path,
        output_eval_dir: Path,
    ):
        """运行行为发现流程（step1）

        执行完整的行为发现流程，包括特征提取、行为模式发现和结果评估

        Args:
            input_processed_path (Path): 处理后的数据文件或目录路径
            output_features_dir (Path): 特征数据的输出目录路径
            output_patterns_dir (Path): 行为模式结果的输出目录路径
            output_eval_dir (Path): 评估结果的输出目录路径

        Examples:
            >>> from pathlib import Path
            >>> from network_simulation.pipeline.pipeline_manager import PipelineManager
            >>> manager = PipelineManager()
            >>> manager.run_behavior_discovery(
            ...     input_processed_path=Path("data/processed"),
            ...     output_features_dir=Path("output/features"),
            ...     output_patterns_dir=Path("output/patterns"),
            ...     output_eval_dir=Path("output/eval")
            ... )
        """
        logger.info("开始运行行为发现流程")

        self.behavior_discovery_pipeline.run(
            input_processed_path=input_processed_path,
            output_features_dir=output_features_dir,
            output_patterns_dir=output_patterns_dir,
            output_eval_dir=output_eval_dir,
        )

    def run_diffusion_model(
        self,
        input_patterns_dir: Path,
        input_processed_path: Path,
        output_preprocess_dir: Path,
        output_train_dir: Path,
        output_generate_dir: Path,
        output_visualize_dir: Path,
        output_evaluation_dir: Path,
        max_samples: int = None,
        num_samples: int = 1000,
    ):
        """运行扩散模型流程（step2）

        执行完整的扩散模型流程，包括数据预处理、模型训练、样本生成、可视化和评估

        Args:
            input_patterns_dir (Path): 行为模式目录路径
            input_processed_path (Path): 处理后的数据文件或目录路径
            output_preprocess_dir (Path): 预处理结果的输出目录路径
            output_train_dir (Path): 训练结果的输出目录路径
            output_generate_dir (Path): 生成结果的输出目录路径
            output_visualize_dir (Path): 可视化结果的输出目录路径
            output_evaluation_dir (Path): 评估结果的输出目录路径
            max_samples (int, optional): 训练时使用的最大样本数量，默认为None
            num_samples (int, optional): 生成的样本数量，默认为1000

        Examples:
            >>> from pathlib import Path
            >>> from network_simulation.pipeline.pipeline_manager import PipelineManager
            >>> manager = PipelineManager()
            >>> manager.run_diffusion_model(
            ...     input_patterns_dir=Path("output/patterns"),
            ...     input_processed_path=Path("data/processed"),
            ...     output_preprocess_dir=Path("output/preprocess"),
            ...     output_train_dir=Path("output/train"),
            ...     output_generate_dir=Path("output/generate"),
            ...     output_visualize_dir=Path("output/visualize"),
            ...     output_evaluation_dir=Path("output/evaluation"),
            ...     max_samples=10000,
            ...     num_samples=1000
            ... )
        """
        logger.info("开始运行扩散模型流程")

        self.diffusion_model_pipeline.run(
            input_patterns_dir=input_patterns_dir,
            input_processed_path=input_processed_path,
            output_preprocess_dir=output_preprocess_dir,
            output_train_dir=output_train_dir,
            output_generate_dir=output_generate_dir,
            output_visualize_dir=output_visualize_dir,
            output_evaluation_dir=output_evaluation_dir,
            max_samples=max_samples,
            num_samples=num_samples,
        )

    def run_e2e(
        self,
        input_processed_path: Path,
        output_features_dir: Path,
        output_patterns_dir: Path,
        output_eval_dir: Path,
        output_preprocess_dir: Path,
        output_train_dir: Path,
        output_generate_dir: Path,
        output_visualize_dir: Path,
        output_evaluation_dir: Path,
        max_samples: int = None,
        num_samples: int = 1000,
    ):
        """运行完整的端到端流程

        执行完整的端到端流程，包括行为发现流程和扩散模型流程

        Args:
            input_processed_path (Path): 处理后的数据文件或目录路径
            output_features_dir (Path): 特征数据的输出目录路径
            output_patterns_dir (Path): 行为模式结果的输出目录路径
            output_eval_dir (Path): 行为发现评估结果的输出目录路径
            output_preprocess_dir (Path): 预处理结果的输出目录路径
            output_train_dir (Path): 训练结果的输出目录路径
            output_generate_dir (Path): 生成结果的输出目录路径
            output_visualize_dir (Path): 可视化结果的输出目录路径
            output_evaluation_dir (Path): 生成评估结果的输出目录路径
            max_samples (int, optional): 训练时使用的最大样本数量，默认为None
            num_samples (int, optional): 生成的样本数量，默认为1000

        Examples:
            >>> from pathlib import Path
            >>> from network_simulation.pipeline.pipeline_manager import PipelineManager
            >>> manager = PipelineManager()
            >>> manager.run_e2e(
            ...     input_processed_path=Path("data/processed"),
            ...     output_features_dir=Path("output/features"),
            ...     output_patterns_dir=Path("output/patterns"),
            ...     output_eval_dir=Path("output/eval"),
            ...     output_preprocess_dir=Path("output/preprocess"),
            ...     output_train_dir=Path("output/train"),
            ...     output_generate_dir=Path("output/generate"),
            ...     output_visualize_dir=Path("output/visualize"),
            ...     output_evaluation_dir=Path("output/evaluation"),
            ...     max_samples=10000,
            ...     num_samples=1000
            ... )
        """
        logger.info("开始运行完整的端到端流程")

        # 运行行为发现流程
        self.run_behavior_discovery(
            input_processed_path=input_processed_path,
            output_features_dir=output_features_dir,
            output_patterns_dir=output_patterns_dir,
            output_eval_dir=output_eval_dir,
        )

        # 运行扩散模型流程
        self.run_diffusion_model(
            input_patterns_dir=output_patterns_dir,
            input_processed_path=input_processed_path,
            output_preprocess_dir=output_preprocess_dir,
            output_train_dir=output_train_dir,
            output_generate_dir=output_generate_dir,
            output_visualize_dir=output_visualize_dir,
            output_evaluation_dir=output_evaluation_dir,
            max_samples=max_samples,
            num_samples=num_samples,
        )

        logger.info("完整的端到端流程运行完成")
