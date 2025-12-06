# 实现step 1_4评估行为发现效果（HTML报告）

## 目标
创建一个新的脚本step 1_4，用于调用现有的评估模块评估行为发现的效果，并生成HTML格式的评估报告。

## 设计思路
1. **输入**：接收step 1_3的输出结果（特征数据、行为标签、转移矩阵等）
2. **处理**：使用现有的Evaluator类评估聚类质量、转移质量和行为分离
3. **输出**：生成HTML格式的综合评估报告，包含可视化图片

## 实现步骤

### 1. 扩展Evaluator类
- 添加`generate_html_report`方法，用于生成HTML格式的评估报告
- 保持原有的Markdown报告功能不变，实现功能扩展

### 2. 创建脚本文件
创建`scripts/step1_4_evaluate_patterns.py`文件

### 3. 脚本结构设计
- 导入必要的模块和类
- 定义主函数和辅助函数
- 实现命令行参数解析
- 实现评估逻辑
- 实现结果保存和HTML报告生成

### 4. 核心功能实现
- **加载数据**：读取特征数据和行为标签
- **加载模型和转移矩阵**：如果存在的话
- **评估聚类质量**：使用`evaluate_clustering_quality`方法
- **评估转移质量**：使用`evaluate_transition_quality`方法
- **评估行为分离**：使用`evaluate_behavior_separation`方法
- **保存评估结果**：使用`save`方法，并确保生成HTML报告

### 5. 命令行参数设计
- 支持输入特征文件或目录
- 支持输入模式结果目录
- 支持输出评估结果目录
- 支持选择报告格式（默认HTML）

### 6. 输出结果设计
- JSON格式的详细评估结果
- 文本格式的评估摘要
- HTML格式的综合评估报告（支持可视化图片）

## 关键代码片段

### 1. 扩展Evaluator类
```python
def generate_html_report(self, results: Dict, output_path: Path) -> None:
    """Generate a comprehensive HTML report of evaluation results"""
    with open(output_path, "w") as f:
        # Write HTML header
        f.write("<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n")
        f.write("<meta charset=\"UTF-8\">\n")
        f.write("<title>Network Simulation Parameter Generation Evaluation Report</title>\n")
        f.write("<style>\n")
        f.write("/* CSS样式定义 */\n")
        f.write("body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }\n")
        f.write(".container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }\n")
        f.write("h1 { color: #2c3e50; }\n")
        f.write("h2 { color: #3498db; border-bottom: 2px solid #3498db; padding-bottom: 5px; }\n")
        f.write("h3 { color: #27ae60; }\n")
        f.write("table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n")
        f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
        f.write("th { background-color: #f2f2f2; }\n")
        f.write(".metrics-section { margin: 20px 0; }\n")
        f.write(".visualization-section { margin: 20px 0; text-align: center; }\n")
        f.write(".visualization-section img { max-width: 100%; height: auto; margin: 10px 0; }\n")
        f.write("</style>\n")
        f.write("</head>\n<body>\n")
        f.write("<div class=\"container\">\n")
        
        # Write report content (similar structure to Markdown but in HTML)
        # 1. Executive Summary
        f.write("<h1>Network Simulation Parameter Generation Evaluation Report</h1>\n")
        f.write("<h2>Executive Summary</h2>\n")
        f.write("<p>This report presents the comprehensive evaluation results of the network simulation parameter generation scheme, including clustering quality, behavior transition quality, and behavior separation analysis.</p>\n")
        
        # 2. Clustering Quality Evaluation
        f.write("<h2>1. Clustering Quality Evaluation</h2>\n")
        f.write("<h3>Metrics</h3>\n")
        f.write("<table>\n")
        f.write("<tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>\n")
        
        # Add clustering metrics here...
        
        f.write("</table>\n")
        
        # 3. Behavior Transition Quality Evaluation
        f.write("<h2>2. Behavior Transition Quality Evaluation</h2>\n")
        # Add transition metrics here...
        
        # 4. Behavior Separation Evaluation
        f.write("<h2>3. Behavior Separation Evaluation</h2>\n")
        # Add separation metrics here...
        
        # 5. Visualization Section
        f.write("<h2>4. Visualization</h2>\n")
        f.write("<div class=\"visualization-section\">\n")
        f.write("<h3>Clustering Results</h3>\n")
        f.write("<img src=\"pca_scatter.png\" alt=\"PCA Scatter Plot\">\n")
        f.write("<img src=\"pca_variance.png\" alt=\"PCA Variance Explained\">\n")
        
        f.write("<h3>Transition Analysis</h3>\n")
        f.write("<img src=\"transition_matrix_heatmap.png\" alt=\"Transition Matrix Heatmap\">\n")
        
        f.write("<h3>Feature Analysis</h3>\n")
        f.write("<img src=\"feature_distributions.png\" alt=\"Feature Distributions\">\n")
        f.write("<img src=\"feature_correlation.png\" alt=\"Feature Correlation Heatmap\">\n")
        f.write("</div>\n")
        
        # 6. Conclusion and Recommendations
        f.write("<h2>5. Conclusion and Recommendations</h2>\n")
        # Add conclusion and recommendations here...
        
        # Write HTML footer
        f.write("</div>\n")
        f.write("</body>\n</html>")
```

### 2. 修改save方法
```python
def save(self, results: Dict, output_dir: Path) -> None:
    """Save evaluation results to directory"""
    logger.info(f"开始保存评估结果到目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results to JSON file
    json_path = output_dir / "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"已保存评估结果到JSON文件: {json_path}")
    
    # Save summary report only if it has the expected keys
    if "statistical_fidelity" in results:
        summary_path = output_dir / "evaluation_summary.txt"
        self._generate_summary_report(results, summary_path)
        logger.info(f"已生成评估摘要报告: {summary_path}")
    
    # Save comprehensive Markdown report
    markdown_path = output_dir / "comprehensive_evaluation_report.md"
    self.generate_comprehensive_report(results, markdown_path)
    logger.info(f"已生成综合评估报告(Markdown): {markdown_path}")
    
    # Save comprehensive HTML report
    html_path = output_dir / "comprehensive_evaluation_report.html"
    self.generate_html_report(results, html_path)
    logger.info(f"已生成综合评估报告(HTML): {html_path}")
    logger.info("所有评估结果已保存完成")
```

### 3. 主脚本核心逻辑
```python
def evaluate_patterns(input_features_file: Path, input_patterns_dir: Path, output_eval_dir: Path):
    """评估行为发现效果"""
    # 加载特征数据
    features_df = pd.read_csv(input_features_file)
    
    # 加载行为发现结果
    labels_data, transition_data = load_behavior_data(input_patterns_dir)
    
    # 初始化评估器
    evaluator = Evaluator()
    
    # 准备评估数据
    X = features_df.values
    labels = np.array(labels_data["labels"])
    transition_matrix = np.array(transition_data["transition_matrix"])
    
    # 评估聚类质量
    clustering_quality = evaluator.evaluate_clustering_quality(X, labels)
    
    # 评估转移质量
    transition_quality = evaluator.evaluate_transition_quality(transition_matrix)
    
    # 评估行为分离
    behavior_separation = evaluator.evaluate_behavior_separation(X, labels)
    
    # 整合评估结果
    evaluation_results = {
        "clustering_quality": clustering_quality,
        "transition_quality": transition_quality,
        "behavior_separation": behavior_separation,
        "transition_matrix": transition_matrix.tolist()
    }
    
    # 保存评估结果（会自动生成HTML报告）
    evaluator.save(evaluation_results, output_eval_dir)
    
    return output_eval_dir
```

## 预期输出
- JSON格式的详细评估结果
- 文本格式的评估摘要
- HTML格式的综合评估报告，包含可视化图片
- 保持原有的Markdown报告功能不变

## 遵循的项目规范
- 使用中文谷歌风格注释
- 遵循脚本命名规范（step1_4_*.py）
- 使用argparse处理命令行参数
- 使用项目统一的日志模块
- 遵循输出文件命名规范
- 支持清理旧文件功能
- 实现功能扩展，保持向后兼容

## 与其他脚本的关系
- **输入**：接收step 1_3_discover_patterns.py的输出结果
- **输出**：生成的HTML评估报告可直接在浏览器中查看，方便查看可视化图片
- **集成**：可被e2e_pipeline.py和optimized_e2e_pipeline.py调用

## 测试计划
1. 确保脚本可以处理单个文件输入
2. 确保脚本可以处理目录输入
3. 确保评估结果正确生成
4. 确保HTML报告格式符合要求，支持可视化图片
5. 确保脚本可以被端到端流水线调用

## 实现时间
预计2-3小时完成实现和测试