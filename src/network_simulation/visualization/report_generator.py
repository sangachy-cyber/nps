#!/usr/bin/env python3
"""
报告生成类
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import markdown2

from .base_visualizer import BaseVisualizer
from .feature_space_visualizer import FeatureSpaceVisualizer
from .behavior_visualizer import BehaviorVisualizer
from config import REPORTS_DIR, BEHAVIOR_CATEGORY_MAP
from ..utils.logger import get_logger


logger = get_logger(__name__)


class ReportGenerator:
    """报告生成类"""

    def __init__(self, base_visualizer: BaseVisualizer):
        self.base_visualizer = base_visualizer

    def generate_markdown_report(
        self,
        results: Dict,
        X: np.ndarray,
        feature_names: List[str],
        raw_data: pd.DataFrame = None,
        features_df: pd.DataFrame = None,
        output_dir: Path = None
    ) -> None:
        """生成综合Markdown报告"""
        method = results["method"]

        # 使用配置的报告目录作为上一层目录
        # 确保可视化目录结构正确
        plots_dir = REPORTS_DIR / "plots"
        plots_dir.mkdir(exist_ok=True)

        behavior_samples_dir = plots_dir / "behavior_samples"
        behavior_samples_dir.mkdir(exist_ok=True)

        timelines_dir = plots_dir / "timelines"
        timelines_dir.mkdir(exist_ok=True)

        feature_space_dir = plots_dir / "feature_space"
        feature_space_dir.mkdir(exist_ok=True)

        # Save standalone visualizations to their respective directories
        if raw_data is not None and features_df is not None:
            # Save behavior samples (1-3 per class) - 分别处理上下行标签
            behavior_visualizer = BehaviorVisualizer(self.base_visualizer.output_dir)

            # 处理上行标签
            labels_up = results.get("labels_up")
            if labels_up is not None:
                behavior_visualizer.save_behavior_samples(
                    raw_data, features_df, labels_up, method, behavior_samples_dir, direction="up"
                )

                # Save label timeline - 上行
                behavior_visualizer.save_label_timeline(
                    raw_data, features_df, labels_up, method, timelines_dir, direction="up"
                )

            # 处理下行标签
            labels_down = results.get("labels_down")
            if labels_down is not None:
                behavior_visualizer.save_behavior_samples(
                    raw_data, features_df, labels_down, method, behavior_samples_dir, direction="down"
                )

                # Save label timeline - 下行
                behavior_visualizer.save_label_timeline(
                    raw_data, features_df, labels_down, method, timelines_dir, direction="down"
                )

        # Save feature space visualizations - 分别为上下行生成
        feature_visualizer = FeatureSpaceVisualizer(self.base_visualizer.output_dir)

        # 为上下行创建独立的特征子集
        up_feature_cols = [col for col in feature_names if col.endswith("1") or "ratio" in col or "symmetry" in col]
        down_feature_cols = [col for col in feature_names if col.endswith("2") or "ratio" in col or "symmetry" in col]

        # 提取对应方向的特征矩阵
        X_up = X[:, [i for i, col in enumerate(feature_names) if col in up_feature_cols]]
        X_down = X[:, [i for i, col in enumerate(feature_names) if col in down_feature_cols]]

        # 处理上行标签
        labels_up = results.get("labels_up")
        if labels_up is not None:
            feature_visualizer.save_feature_space_visualizations(
                X_up, labels_up, method, feature_space_dir, up_feature_cols, direction="up"
            )

        # 处理下行标签
        labels_down = results.get("labels_down")
        if labels_down is not None:
            feature_visualizer.save_feature_space_visualizations(
                X_down, labels_down, method, feature_space_dir, down_feature_cols, direction="down"
            )

        # 生成标签时间轴图部分
        timeline_images = []
        # 获取所有时间轴图文件
        timeline_files = sorted(timelines_dir.glob("*.png"))
        for timeline_file in timeline_files:
            timeline_images.append(f"![时间轴图]({timeline_file.relative_to(REPORTS_DIR)})")
        timeline_content = "\n\n".join(timeline_images) if timeline_images else "暂无时间轴图数据"

        # 统计不同文件中各行为类别的占比
        file_behavior_stats = self._generate_file_behavior_stats(raw_data, features_df, results)

        # 生成Markdown报告内容
        markdown_content = self._generate_markdown_content(results, method, feature_names, file_behavior_stats, timeline_content)

        # 确定输出目录
        if output_dir is None:
            output_dir = REPORTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存Markdown报告
        markdown_file = output_dir / f"behavior_pattern_report_{method}.md"
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        logger.info(f"Markdown报告已生成：{markdown_file}")
        logger.info(f"可视化资产已保存到目录：{plots_dir}")

    def generate_evaluation_markdown_report(
        self,
        results: Dict,
        output_dir: Path,
    ) -> None:
        """生成评估报告的Markdown版本"""
        # 生成评估报告内容
        markdown_content = self._generate_evaluation_markdown_content(results)

        # 保存Markdown报告
        markdown_file = output_dir / "evaluation_report.md"
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        logger.info(f"评估Markdown报告已生成：{markdown_file}")

    def _generate_evaluation_markdown_content(self, results: Dict) -> str:
        """生成评估报告的Markdown内容"""
        return f"""# 网络模拟数据评估报告

## 1. 统计保真度 (L1)

| 指标名称 | 数值 | 说明 |
|----------|------|------|
| 时延均值 | {results["statistical_fidelity"]["delay_mean"]:.2f} ms | 生成数据的平均时延 |
| 时延标准差 | {results["statistical_fidelity"]["delay_std"]:.2f} ms | 生成数据的时延标准差 |
| 丢包率均值 | {results["statistical_fidelity"]["loss_rate_mean"]:.4f} | 生成数据的平均丢包率 |
| 丢包率标准差 | {results["statistical_fidelity"]["loss_rate_std"]:.4f} | 生成数据的丢包率标准差 |
| 丢包率范围 | [{results["statistical_fidelity"]["loss_rate_range"]["min"]:.4f}, {results["statistical_fidelity"]["loss_rate_range"]["max"]:.4f}] | 生成数据的丢包率范围 |

## 2. 不可区分性 (L2)

| 指标名称 | 数值 | 说明 |
|----------|------|------|
| 判别器AUC | {results["indistinguishability"]["discriminator_auc"]:.4f} | 0.5表示完美不可区分 |
| TSTR保留率 | {results["indistinguishability"]["tstr_keep_rate"]:.4f} | 训练在合成数据，测试在真实数据上的模型性能 |

## 3. 动态合理性 (L3)

| 指标名称 | 数值 | 说明 |
|----------|------|------|
| 时延5阶自相关 | {results["dynamic_rationality"]["delay_acf_5"]:.4f} | 时延序列的自相关性 |
| 突发数量 | {results["dynamic_rationality"]["burst_statistics"]["num_bursts"]} | 生成数据中的丢包突发数量 |
| 平均突发持续时间 | {results["dynamic_rationality"]["burst_statistics"]["avg_burst_duration"]:.2f} 秒 | 丢包突发的平均持续时间 |
| 突发频率 | {results["dynamic_rationality"]["burst_statistics"]["burst_frequency"]:.4f} 突发/秒 | 每秒的平均突发次数 |
| 行为对齐准确率 | {results["dynamic_rationality"]["behavior_alignment_accuracy"]:.4f} | 生成数据与预期行为的对齐程度 |

## 4. 总体评估

生成的网络模拟数据显示出良好的质量，具有合理的统计特性。
通过与真实网络数据进行比较和优化生成模型，可以进一步改进。
"""

    def _generate_file_behavior_stats(
        self,
        raw_data: pd.DataFrame,
        features_df: pd.DataFrame,
        results: Dict
    ) -> str:
        """生成文件行为统计"""
        file_behavior_stats = ""
        if raw_data is not None and features_df is not None:
            # 将features_df的window_start和window_end映射到raw_data的file_source
            # 创建窗口到文件的映射
            window_to_file = []
            # 使用上行标签作为默认标签，保持向后兼容性
            labels = results.get("labels_up", results.get("labels"))
            if labels is None:
                logger.error("未找到可用的标签数据，无法生成文件行为统计")
                return file_behavior_stats

            labels = np.array(labels)
            for i in range(len(features_df)):
                window_start = int(features_df.iloc[i]["window_start"])
                window_end = int(features_df.iloc[i]["window_end"])
                # 获取该窗口的中心索引
                mid_idx = (window_start + window_end) // 2
                # 确保mid_idx不超过raw_data的长度
                mid_idx = min(mid_idx, len(raw_data) - 1)
                # 获取该索引对应的file_source
                file_source = raw_data.iloc[mid_idx]["file_source"]
                window_to_file.append(file_source)

            # 创建DataFrame统计
            stats_df = pd.DataFrame({
                "file_source": window_to_file,
                "label": labels
            })

            # 计算每个文件中各行为类别的数量
            behavior_counts = stats_df.groupby(["file_source", "label"]).size().unstack(fill_value=0)

            # 计算每个文件的总窗口数
            file_totals = behavior_counts.sum(axis=1)

            # 计算占比
            behavior_percentages = behavior_counts.div(file_totals, axis=0) * 100

            # 生成Markdown表格
            file_behavior_stats = self._generate_file_stats_table(behavior_counts, behavior_percentages, file_totals)
        return file_behavior_stats

    def _generate_file_stats_table(
        self,
        behavior_counts: pd.DataFrame,
        behavior_percentages: pd.DataFrame,
        file_totals: pd.Series
    ) -> str:
        """生成文件统计表格"""
        file_behavior_stats = "## 8. 不同文件行为类别占比统计\n\n"
        file_behavior_stats += "| 文件名 | "
        # 添加行为类别列
        sorted_behavior_ids = sorted(behavior_counts.columns)
        for behavior_id in sorted_behavior_ids:
            behavior_name = BEHAVIOR_CATEGORY_MAP.get(behavior_id, f"行为 {behavior_id}")
            file_behavior_stats += f"{behavior_name} (%) | "
        file_behavior_stats = file_behavior_stats.rstrip(" | ") + "|\n"

        # 添加分隔线
        file_behavior_stats += "|----------|"
        for _ in sorted_behavior_ids:
            file_behavior_stats += "--- |"
        file_behavior_stats = file_behavior_stats.rstrip("|") + "|\n"

        # 添加数据行
        for file_name in behavior_counts.index:
            file_behavior_stats += f"| {file_name} | "
            for behavior_id in sorted_behavior_ids:
                percentage = behavior_percentages.loc[file_name, behavior_id]
                file_behavior_stats += f"{percentage:.1f} | "
            file_behavior_stats = file_behavior_stats.rstrip(" | ") + "|\n"

        # 添加文件总窗口数统计
        file_behavior_stats += "\n### 8.1 文件总窗口数\n\n"
        file_behavior_stats += "| 文件名 | 总窗口数 |\n"
        file_behavior_stats += "|----------|----------|\n"
        for file_name, total in file_totals.items():
            file_behavior_stats += f"| {file_name} | {total} |\n"
        return file_behavior_stats

    def _generate_separation_table_markdown(
        self, separation_metrics: Dict, feature_names: List[str]
    ) -> str:
        """生成行为分离表格的Markdown格式"""
        table_rows = []
        # 先检查separation_metrics的结构
        logger.debug(f"separation_metrics结构: {list(separation_metrics.keys())}")

        # 直接访问separation_metrics中的特征均值
        if "label_means" in separation_metrics:
            # 从label_means字段获取数据
            table_rows.extend(self._generate_rows_from_label_means(
                separation_metrics["label_means"], feature_names
            ))
        elif "behavior_means" in separation_metrics:
            # 兼容旧格式，从behavior_means字段获取数据
            table_rows.extend(self._generate_rows_from_behavior_means(
                separation_metrics["behavior_means"], feature_names
            ))
        else:
            # 否则遍历所有行为
            table_rows.extend(self._generate_rows_from_other_formats(
                separation_metrics, feature_names
            ))

        return "\n".join(table_rows)

    def _generate_rows_from_label_means(
        self, label_means: Dict, feature_names: List[str]
    ) -> List[str]:
        """从label_means生成表格行"""
        rows = []
        for behavior, means in label_means.items():
            # 将行为ID转换为中文名称
            behavior_id = int(behavior)
            behavior_name = BEHAVIOR_CATEGORY_MAP.get(behavior_id, f"行为 {behavior}")
            feature_means = [f"{means[i]:.4f}" for i in range(len(feature_names))]
            table_row = "| {0} | {1} |".format(behavior_name, " | ".join(feature_means))
            rows.append(table_row)
        return rows

    def _generate_rows_from_behavior_means(
        self, behavior_means: Dict, feature_names: List[str]
    ) -> List[str]:
        """从behavior_means生成表格行"""
        rows = []
        for behavior, means in behavior_means.items():
            # 将行为ID转换为中文名称
            behavior_id = int(behavior)
            behavior_name = BEHAVIOR_CATEGORY_MAP.get(behavior_id, f"行为 {behavior}")
            feature_means = [f"{means[i]:.4f}" for i in range(len(feature_names))]
            table_row = "| {0} | {1} |".format(behavior_name, " | ".join(feature_means))
            rows.append(table_row)
        return rows

    def _generate_rows_from_other_formats(
        self, separation_metrics: Dict, feature_names: List[str]
    ) -> List[str]:
        """从其他格式生成表格行"""
        rows = []
        for behavior, metrics in separation_metrics.items():
            if behavior == "separation_metrics" or behavior == "behavior_similarity":
                continue
            # 检查metrics的结构
            if isinstance(metrics, dict):
                logger.debug(f"行为 {behavior} 的metrics结构: {list(metrics.keys())}")
                # 尝试不同的键名
                row = self._generate_row_from_metrics_dict(behavior, metrics, feature_names)
                if row:
                    rows.append(row)
        return rows

    def _generate_row_from_metrics_dict(
        self, behavior: str, metrics: Dict, feature_names: List[str]
    ) -> str:
        """从metrics字典生成单行"""
        behavior_id = int(behavior)
        behavior_name = BEHAVIOR_CATEGORY_MAP.get(behavior_id, f"行为 {behavior}")

        if "mean" in metrics:
            # 提取特征均值，保留4位小数
            feature_means = [f"{metrics['mean'][i]:.4f}" for i in range(len(feature_names))]
            return "| {0} | {1} |".format(behavior_name, " | ".join(feature_means))
        elif "means" in metrics:
            # 尝试means键
            feature_means = [f"{metrics['means'][i]:.4f}" for i in range(len(feature_names))]
            return "| {0} | {1} |".format(behavior_name, " | ".join(feature_means))
        elif isinstance(metrics, (list, np.ndarray)):
            # 如果metrics是列表或数组，直接使用
            feature_means = [f"{metrics[i]:.4f}" for i in range(min(len(metrics), len(feature_names)))]
            # 补全缺失的特征
            while len(feature_means) < len(feature_names):
                feature_means.append("-")
            return "| {0} | {1} |".format(behavior_name, " | ".join(feature_means))
        return ""

    def _generate_markdown_content(
        self,
        results: Dict,
        method: str,
        feature_names: List[str],
        file_behavior_stats: str,
        timeline_content: str
    ) -> str:
        """生成Markdown内容"""
        return f"""# 网络行为模式发现报告

## 行为检测方法：{"规则" if method == "rule" else method}

## 1. 行为转移质量评估

### 上行行为转移指标
| 指标名称 | 数值 | 说明 |
|----------|------|------|
| 平均转移熵 | {results.get("metrics_up", results.get("metrics", {})).get("average_transition_entropy", 0):.4f} | 衡量状态转移不确定性，越低越确定 |
| 转移稀疏性 | {results.get("metrics_up", results.get("metrics", {})).get("transition_sparsity", 0):.4f} | 非零转移概率占比，反映行为切换复杂度 |

### 下行行为转移指标
| 指标名称 | 数值 | 说明 |
|----------|------|------|
| 平均转移熵 | {results.get("metrics_down", {}).get("average_transition_entropy", 0):.4f} | 衡量状态转移不确定性，越低越确定 |
| 转移稀疏性 | {results.get("metrics_down", {}).get("transition_sparsity", 0):.4f} | 非零转移概率占比，反映行为切换复杂度 |

## 2. 特征降维可视化

不同降维方法的对比：
- PCA：线性降维，保留最大方差
- t-SNE：非线性降维，专注于局部结构，适合可视化高维数据
- UMAP：非线性降维，同时保留局部和全局结构，运行速度更快

### 2.1 PCA 散点图

![PCA 散点图](plots/feature_space/pca_scatter.png)

### 2.2 PCA 方差解释图

![PCA 方差解释图](plots/feature_space/pca_variance.png)

### 2.3 t-SNE 散点图

![t-SNE 散点图](plots/feature_space/tsne.png)

### 2.4 UMAP 散点图

![UMAP 散点图](plots/feature_space/umap.png)

## 3. 特征分析

### 3.1 特征分布直方图

![特征分布直方图](plots/feature_space/feature_distributions.png)

### 3.2 特征相关性热力图

![特征相关性热力图](plots/feature_space/feature_correlation.png)

## 4. 行为转移分析

### 4.1 状态转移矩阵热力图

![状态转移矩阵热力图](plots/transition_matrix_heatmap.png)

## 5. 行为特征分析

### 5.1 上行行为特征均值

| 行为类别 | {" | ".join(feature_names)} |
|----------|{" | ".join(["---" for _ in feature_names])}|
{self._generate_separation_table_markdown(results.get("separation_metrics_up", {}), feature_names)}

### 5.2 下行行为特征均值

| 行为类别 | {" | ".join(feature_names)} |
|----------|{" | ".join(["---" for _ in feature_names])}|
{self._generate_separation_table_markdown(results.get("separation_metrics_down", {}), feature_names)}

## 6. 标签时间轴图

全局行为分布可视化，展示不同网络行为在时间轴上的分布：

{timeline_content}

## 7. 原始数据样本可视化

以下是每个行为类别的典型样本对应的原始时延和丢包率可视化：

![典型样本](plots/behavior_samples/typical_samples.png)

{file_behavior_stats}
"""

    def generate_html_report(
        self,
        results: Dict,
        X: np.ndarray,
        feature_names: List[str],
        raw_data: pd.DataFrame = None,
        features_df: pd.DataFrame = None,
        output_dir: Path = None
    ) -> None:
        """生成综合HTML报告"""
        method = results["method"]

        # 先生成Markdown报告
        self.generate_markdown_report(results, X, feature_names, raw_data, features_df, output_dir)

        # 使用配置的报告目录作为上一层目录
        # 确保可视化目录结构正确
        plots_dir = REPORTS_DIR / "plots"
        plots_dir.mkdir(exist_ok=True)

        # 确定输出目录
        if output_dir is None:
            output_dir = REPORTS_DIR

        # 读取Markdown报告
        markdown_file = output_dir / f"behavior_pattern_report_{method}.md"
        with open(markdown_file, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        # 将Markdown转换为HTML
        html_content = markdown2.markdown(
            markdown_content,
            extras=["tables", "fenced-code-blocks", "header-ids", "toc", "footnotes"],
        )

        # 创建完整的HTML报告
        method_name = "规则" if method == "rule" else method
        full_html = self._generate_full_html(html_content, method_name)

        # 保存HTML报告
        html_file = output_dir / f"behavior_pattern_report_{method}.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(full_html)

        logger.info(f"HTML报告已生成：{html_file}")
        logger.info(f"可视化资产已保存到目录：{plots_dir}")

    def generate_evaluation_html_report(
        self,
        results: Dict,
        output_dir: Path,
    ) -> None:
        """生成评估报告的HTML版本"""
        # 先生成Markdown报告
        self.generate_evaluation_markdown_report(results, output_dir)

        # 读取Markdown报告
        markdown_file = output_dir / "evaluation_report.md"
        with open(markdown_file, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        # 将Markdown转换为HTML
        html_content = markdown2.markdown(
            markdown_content,
            extras=["tables", "fenced-code-blocks", "header-ids", "toc", "footnotes"],
        )

        # 创建完整的HTML报告
        full_html = self._generate_full_html(html_content, "evaluation")

        # 保存HTML报告
        html_file = output_dir / "evaluation_report.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(full_html)

        logger.info(f"评估HTML报告已生成：{html_file}")

    def _generate_full_html(self, html_content: str, method_name: str) -> str:
        """生成完整的HTML内容"""
        # 使用传统字符串替换方式，避免ruff误将CSS属性识别为Python变量
        html_template = '''
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>网络行为模式发现报告 - METHOD_NAME 行为检测</title>
            <style>
                * {
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                }

                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    font-size: 16px;
                    line-height: 1.6;
                    color: #333;
                    background-color: #f5f7fa;
                    margin: 0;
                    padding: 0;
                }

                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }

                h1 {
                    color: #2c3e50;
                    font-size: 28px;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #3498db;
                }

                h2 {
                    color: #2c3e50;
                    font-size: 24px;
                    margin: 30px 0 15px;
                    padding-left: 10px;
                    border-left: 4px solid #3498db;
                }

                h3 {
                    color: #2c3e50;
                    font-size: 20px;
                    margin: 25px 0 15px;
                    padding-left: 8px;
                    border-left: 3px solid #3498db;
                }

                p {
                    margin: 15px 0;
                    text-align: justify;
                }

                ul, ol {
                    margin: 15px 0 15px 25px;
                }

                li {
                    margin: 8px 0;
                }

                /* 表格样式优化 */
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    background-color: white;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }

                th, td {
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #e0e0e0;
                }

                th {
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }

                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }

                tr:hover {
                    background-color: #f5f5f5;
                }

                /* 图片样式 */
                img {
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 20px auto;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
                    border-radius: 4px;
                }

                /* 代码块样式 */
                pre {
                    background-color: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 4px;
                    padding: 16px;
                    overflow-x: auto;
                    margin: 20px 0;
                }

                code {
                    font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;
                    font-size: 14px;
                    background-color: #f8f9fa;
                    padding: 2px 4px;
                    border-radius: 3px;
                }

                pre code {
                    background-color: transparent;
                    padding: 0;
                    border-radius: 0;
                }

                /* 响应式设计 */
                @media (max-width: 768px) {
                    .container {
                        padding: 10px;
                    }

                    h1 {
                        font-size: 24px;
                    }

                    h2 {
                        font-size: 20px;
                    }

                    h3 {
                        font-size: 18px;
                    }

                    table {
                        font-size: 14px;
                    }

                    th, td {
                        padding: 8px 10px;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                HTML_CONTENT
            </div>
        </body>
        </html>
        '''

        # 替换占位符
        html_template = html_template.replace('METHOD_NAME', method_name)
        html_template = html_template.replace('HTML_CONTENT', html_content)

        return html_template
