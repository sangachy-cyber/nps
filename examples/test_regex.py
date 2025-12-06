#!/usr/bin/env python3
"""
Test regex pattern for matching image references in Markdown
"""

import re

# Test content from the Markdown file
test_content = """
### 可视化
- **聚类指标**：`clustering_metrics.png`
- **PCA散点图**：`pca_scatter.png`
- **PCA方差解释**：`pca_variance.png`
"""

# Pattern to match image references like: - **聚类指标**：`clustering_metrics.png`
image_pattern = r"- \*\*(.*?)\*\*：`(.*?)`"

# Test the pattern
matches = re.findall(image_pattern, test_content, flags=re.MULTILINE)
print(f"Found {len(matches)} matches:")
for match in matches:
    print(f"Title: {match[0]}, Image Path: {match[1]}")


# Test replacement
def replace_image_reference(match):
    title = match.group(1)
    image_path = match.group(2)
    return f'<div style="margin: 20px 0; text-align: center;"><h4>{title}</h4><img src="{image_path}" alt="{title}" style="max-width: 100%; height: auto; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23); border-radius: 4px;"></div>'


replaced_content = re.sub(
    image_pattern, replace_image_reference, test_content, flags=re.MULTILINE
)
print("\nReplaced content:")
print(replaced_content)
