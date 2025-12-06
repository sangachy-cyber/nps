#!/usr/bin/env python3
"""
统一配置文件
包含所有脚本的默认参数和路径配置
"""

from pathlib import Path

# 目录结构配置
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
FEATURES_DIR = RESULTS_DIR / "features"
PATTERNS_DIR = RESULTS_DIR / "patterns"
PREPROCESS_DIR = RESULTS_DIR / "preprocess"
EVALUATION_DIR = RESULTS_DIR / "evaluation"
MODELS_DIR = DATA_DIR / "models"
GENERATED_DIR = DATA_DIR / "generated"

# 默认文件名配置
DEFAULT_PROCESSED_FILE = PROCESSED_DIR / "processed_data.csv"
DEFAULT_FEATURES_FILE = FEATURES_DIR / "merged_features.csv"
DEFAULT_PATTERNS_DIR = PATTERNS_DIR
DEFAULT_PREPROCESS_DIR = PREPROCESS_DIR
DEFAULT_MODEL_DIR = MODELS_DIR / "diffusion_model"
DEFAULT_GENERATED_DIR = GENERATED_DIR

# 模型参数配置
DEFAULT_INPUT_DIM = 2
DEFAULT_BEHAVIOR_EMBED_DIM = 32
DEFAULT_T = 1000

# 训练参数配置
DEFAULT_EPOCHS = 100  # 训练轮数：从50增加到100，以提高模型收敛效果
DEFAULT_LEARNING_RATE = 2e-4  # 学习率：微调为2e-4，平衡训练速度和收敛质量
DEFAULT_BATCH_SIZE = 32  # 批次大小：使用32，在训练效率和内存占用间取得平衡

# HDBSCAN参数配置
DEFAULT_MIN_CLUSTER_SIZE = 15  # 最小聚类大小：增加到15，减少生成的类别数量，提高聚类纯度
DEFAULT_MIN_SAMPLES = 5        # 最小样本数：增加到5，提高聚类质量，减少噪声聚类
DEFAULT_CLUSTER_SELECTION_EPSILON = 0.3  # 聚类选择阈值：增加到0.3，放宽聚类条件，允许更多相似簇合并

# 清理配置
CLEANUP_OLD_FILES = True
KEEP_LATEST_FILES = 1

# 生成样本配置
DEFAULT_SAMPLE_LENGTH = 6000
DEFAULT_NUM_GROUPS = 2

# 窗口配置
DEFAULT_WINDOW_SIZE = 100
DEFAULT_STRIDE = 50
