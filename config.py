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
REPORTS_DIR = DATA_DIR / "reports"

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

# 清理配置
CLEANUP_OLD_FILES = True
KEEP_LATEST_FILES = 1

# 生成样本配置
DEFAULT_SAMPLE_LENGTH = 6000
DEFAULT_NUM_GROUPS = 2

# 预处理配置
DEFAULT_WINDOW_SIZE = 100  # 窗口大小，用于特征提取和行为标签对齐
DEFAULT_STRIDE = 50  # 滑动步长，用于特征提取和行为标签对齐

# 拥堵检测配置
CONGESTION_DELAY_THRESHOLD = 100  # 拥堵检测的延迟阈值（ms）
CONGESTION_LOSS_THRESHOLD = 0.3  # 拥堵检测的丢包率阈值

# 可视化相关配置
VISUALIZATION_DIR = DATA_DIR / "visualization"
PLOTS_DIR = REPORTS_DIR / "plots"
BEHAVIOR_SAMPLES_DIR = PLOTS_DIR / "behavior_samples"
TIMELINES_DIR = PLOTS_DIR / "timelines"
FEATURE_SPACE_DIR = PLOTS_DIR / "feature_space"

# 报告相关配置
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 可视化文件名配置
DEFAULT_INTERACTIVE_TRANSITION_GRAPH = "interactive_transition_graph.html"
DEFAULT_TRANSITION_MATRIX_HEATMAP = "transition_matrix_heatmap.png"
DEFAULT_TRANSITION_METRICS = "transition_metrics.png"
DEFAULT_TYPICAL_SAMPLES = "typical_samples.png"
DEFAULT_BEHAVIOR_SAMPLE_PREFIX = "behavior_"
DEFAULT_FEATURE_DISTRIBUTIONS = "feature_distributions.png"
DEFAULT_FEATURE_CORRELATION = "feature_correlation.png"

# 综合报告图片文件名配置
DEFAULT_SAMPLE_COMPARISON = "sample_comparison.png"
DEFAULT_GENERATED_DISTRIBUTION = "generated_distribution.png"
DEFAULT_ORIGINAL_DISTRIBUTION = "original_distribution.png"
DEFAULT_GENERATED_TIMELINE = "generated_timeline.png"
DEFAULT_ORIGINAL_TIMELINE = "original_timeline.png"

# 报告文件名配置
DEFAULT_HTML_REPORT_NAME = "behavior_pattern_report.html"
DEFAULT_MD_REPORT_NAME = "behavior_pattern_report.md"
DEFAULT_EVALUATION_HTML_REPORT_NAME = "comprehensive_evaluation_report.html"
DEFAULT_EVALUATION_MD_REPORT_NAME = "comprehensive_evaluation_report.md"

# HTML报告配置
DEFAULT_HTML_TITLE = "网络模拟参数生成评估报告"

# 报告中图片相对路径配置
DEFAULT_REPORT_IMAGE_PATH = "plots/"

# 行为类别映射配置
BEHAVIOR_CATEGORY_MAP = {
    0: "稳定",
    1: "弱突发",
    2: "强突发",
    3: "瞬时峰值",
    4: "高延迟无丢包",
    5: "高丢包低延迟",
    6: "强突发高延迟",
    7: "复杂网络行为"
}

# 行为类别中文标签列表
BEHAVIOR_CATEGORY_LABELS = [
    "稳定",
    "弱突发",
    "强突发",
    "瞬时峰值",
    "高延迟无丢包",
    "高丢包低延迟",
    "强突发高延迟",
    "复杂网络行为"
]

# 行为类别统一颜色配置
BEHAVIOR_CATEGORY_COLORS = {
    0: "#4CAF50",  # 稳定 - 深绿色
    1: "#FF9800",  # 弱突发 - 橙色
    2: "#F44336",  # 强突发 - 红色
    3: "#FFEB3B",  # 瞬时峰值 - 黄色
    4: "#2196F3",  # 高延迟无丢包 - 蓝色
    5: "#9C27B0",  # 高丢包低延迟 - 紫色
    6: "#FF5722",  # 强突发高延迟 - 深橙色
    7: "#00BCD4"   # 复杂网络行为 - 青色
}

# 行为类别颜色列表，与BEHAVIOR_CATEGORY_LABELS对应
BEHAVIOR_CATEGORY_COLOR_LIST = [
    "#4CAF50",  # 稳定 - 深绿色
    "#FF9800",  # 弱突发 - 橙色
    "#F44336",  # 强突发 - 红色
    "#FFEB3B",  # 瞬时峰值 - 黄色
    "#2196F3",  # 高延迟无丢包 - 蓝色
    "#9C27B0",  # 高丢包低延迟 - 紫色
    "#FF5722",  # 强突发高延迟 - 深橙色
    "#00BCD4"   # 复杂网络行为 - 青色
]

# 可视化算法参数配置
# UMAP参数
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# t-SNE参数
TSNE_PERPLEXITY = 30
TSNE_MAX_ITER = 300

# 图表参数
DEFAULT_DPI = 150
DEFAULT_FIGSIZE = (10, 8)
SCATTER_ALPHA = 0.6
HEATMAP_FIGSIZE = (12, 10)

# 时间轴参数
DEFAULT_TIMELINE_FIGSIZE = (15, 8)
DEFAULT_TIMELINE_XTICK_INTERVAL = 100  # 秒

# 行为样本参数
DEFAULT_BEHAVIOR_SAMPLES_PER_CLASS = 2
