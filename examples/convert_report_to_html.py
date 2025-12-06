#!/usr/bin/env python3
"""
Convert Markdown report to HTML format
"""

import markdown
from pathlib import Path


def convert_markdown_to_html(markdown_path: Path, html_path: Path) -> None:
    """Convert Markdown file to HTML file"""
    # Read the Markdown content
    with open(markdown_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    # Replace image references with actual image tags in Markdown content first
    import re
    import glob

    # Pattern to match image references like: - **聚类指标**：`clustering_metrics.png`
    image_pattern = r"- \*\*(.*?)\*\*：`(.*?)`"

    def replace_image_reference(match):
        title = match.group(1)
        file_path = match.group(2)

        # Check if it's a PNG image
        if file_path.endswith(".png"):
            # Return image tag for PNG files
            return f'<div style="margin: 20px 0; text-align: center;"><h4>{title}</h4><img src="{file_path}" alt="{title}" style="max-width: 100%; height: auto; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23); border-radius: 4px;"></div>'
        elif file_path.endswith(".html"):
            # Return link for HTML files
            return f'<div style="margin: 20px 0; text-align: center;"><h4>{title}</h4><a href="{file_path}" target="_blank" style="color: #3498db; text-decoration: none; font-weight: bold;">查看{title}</a></div>'
        else:
            # For other file types, return original format
            return f"- **{title}**：`{file_path}`"

    # Pattern to match random sample sections like: - 行为 0：`behavior_0_sample_*.png`
    random_sample_pattern = r"- 行为 (\d+)：`(behavior_\d+_sample_\*\.png)`"

    def replace_random_samples(match):
        behavior_id = match.group(1)
        wildcard_pattern = match.group(2)

        # Get the actual files matching the wildcard pattern
        report_dir = str(Path(markdown_path).parent)
        actual_files = sorted(glob.glob(f"{report_dir}/{wildcard_pattern}"))

        if not actual_files:
            # If no files match, return original text
            return f"- 行为 {behavior_id}：`{wildcard_pattern}`"

        # Generate image tags for each actual file
        result = f"<h4>行为 {behavior_id} 样本</h4>"
        result += '<div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;">'

        for file_path in actual_files:
            # Get just the filename without the directory
            filename = Path(file_path).name
            # Generate image tag
            result += f'<div style="flex: 1 1 200px; max-width: 300px;"><img src="{filename}" alt="行为 {behavior_id} 样本" style="max-width: 100%; height: auto; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 4px;"></div>'

        result += "</div>"
        return result

    # Replace all image references in the Markdown content
    markdown_content = re.sub(
        image_pattern, replace_image_reference, markdown_content, flags=re.MULTILINE
    )

    # Replace random sample wildcards with actual image tags
    markdown_content = re.sub(
        random_sample_pattern,
        replace_random_samples,
        markdown_content,
        flags=re.MULTILINE,
    )

    # Convert Markdown to HTML
    html_content = markdown.markdown(
        markdown_content, extensions=["tables", "fenced_code"]
    )

    # Add HTML header and footer
    # Use a safer approach with string concatenation
    html_header = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>网络仿真参数生成评估报告</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 24px;
            margin-bottom: 16px;
        }
        h1 {
            font-size: 2.5em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            font-size: 2em;
            border-bottom: 1px solid #ddd;
            padding-bottom: 8px;
        }
        h3 {
            font-size: 1.5em;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        pre {
            background-color: #f4f4f4;
            padding: 16px;
            border-radius: 4px;
            overflow-x: auto;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        }
        ul, ol {
            margin: 16px 0;
            padding-left: 24px;
        }
        li {
            margin-bottom: 8px;
        }
        a {
            color: #3498db;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);
        }
        .section {
            margin-bottom: 30px;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>网络仿真参数生成评估报告</h1>
        </header>
        
        <main>
"""

    html_footer = """
        </main>
        
        <footer class="footer">
            <p>生成时间: {}</p>
        </footer>
    </div>
</body>
</html>"""

    # Combine header, content, and footer
    full_html = (
        html_header
        + html_content
        + html_footer.format(Path(markdown_path).stat().st_mtime)
    )

    # Write the HTML content to file
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Successfully converted {markdown_path} to {html_path}")


def main():
    # Define paths
    data_dir = Path("./data")
    markdown_path = (
        data_dir / "results" / "evaluation" / "comprehensive_evaluation_report_zh.md"
    )
    html_path = (
        data_dir / "results" / "evaluation" / "comprehensive_evaluation_report_zh.html"
    )

    # Convert Markdown to HTML
    convert_markdown_to_html(markdown_path, html_path)


if __name__ == "__main__":
    main()
