# 网络行为模式发现报告

## 行为检测方法：规则

## 1. 行为转移质量评估

### 上行行为转移指标
| 指标名称 | 数值 | 说明 |
|----------|------|------|
| 平均转移熵 | 0.5502 | 衡量状态转移不确定性，越低越确定 |
| 转移稀疏性 | 0.1562 | 非零转移概率占比，反映行为切换复杂度 |

### 下行行为转移指标
| 指标名称 | 数值 | 说明 |
|----------|------|------|
| 平均转移熵 | 0.4190 | 衡量状态转移不确定性，越低越确定 |
| 转移稀疏性 | 0.1406 | 非零转移概率占比，反映行为切换复杂度 |

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

| 行为类别 | feat_delay1_std | feat_delay1_mean | feat_loss1_std | feat_loss2_nonzero_ratio | feat_loss2_high_ratio | feat_loss2_mean | feat_loss2_std | feat_max_consec_loss2 | feat_max_congestion_run2 | feat_delay1_trend | feat_delay1_acf_5 | feat_loss2_mode_encoded | feat_delay_ratio |
|----------|--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---|
| 稳定 | 0.2146 | 4.3642 | 0.1262 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0013 | 0.1114 | 0.0000 | 1.0028 |
| 强突发 | 0.3190 | 6.4506 | 0.3662 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0038 | 0.4442 | 0.0000 | 1.0034 |
| 瞬时峰值 | 0.2086 | 3.2497 | 0.0995 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0008 | 0.0722 | 0.0000 | 0.9873 |
| 高丢包低延迟 | 1.1994 | 4.8308 | 0.3243 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0065 | 0.7549 | 0.0000 | 0.9990 |

### 5.2 下行行为特征均值

| 行为类别 | feat_delay1_std | feat_delay1_mean | feat_loss1_std | feat_loss2_nonzero_ratio | feat_loss2_high_ratio | feat_loss2_mean | feat_loss2_std | feat_max_consec_loss2 | feat_max_congestion_run2 | feat_delay1_trend | feat_delay1_acf_5 | feat_loss2_mode_encoded | feat_delay_ratio |
|----------|--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---|
| 稳定 | 0.1938 | 5.5722 | 0.2728 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0018 | 0.2517 | 0.0000 | 0.9998 |
| 弱突发 | 1.2831 | 5.0470 | 0.4046 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0340 | 0.7878 | 0.0000 | 1.0077 |
| 瞬时峰值 | 1.0182 | 5.2350 | 0.3072 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0281 | 0.7656 | 0.0000 | 0.9985 |
| 高延迟无丢包 | 0.0408 | 6.7209 | 0.2845 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 | 0.4534 | 0.0000 | 1.0014 |

## 6. 标签时间轴图

全局行为分布可视化，展示不同网络行为在时间轴上的分布：

![时间轴图](plots/timelines/20251203_230356_b6x-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/20251203_230356_b6x-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/20251203_230356_b6x-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/20251203_230356_b6x-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/20251203_230356_b6x-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/20251203_230356_b6x-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/20251204_001204_b6x-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/20251204_001204_b6x-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/20251204_001204_b6x-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/20251204_001204_b6x-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/20251204_001204_b6x-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/20251204_001204_b6x-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/20251204_001204_b6x-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/20251204_001204_b6x-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/20251204_001204_b6x-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/20251204_001204_b6x-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_11.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_12.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_13.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_14.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_15.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_16.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_17.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_18.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_19.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_20.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/20251204_012208_co0-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/20251204_222027_ZGp-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/20251204_222027_ZGp-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/20251204_222027_ZGp-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/20251204_222027_ZGp-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/20251204_222027_ZGp-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/20251204_222027_ZGp-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/20251204_222027_ZGp-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/20251204_222027_ZGp-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/20251204_222027_ZGp-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/20251204_222027_ZGp-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/20251204_232415_ZGp-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/20251204_232415_ZGp-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/20251205_004119_ZGp-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/20251205_004119_ZGp-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/20251205_004119_ZGp-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/20251205_004119_ZGp-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/20251205_004119_ZGp-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/20251205_004119_ZGp-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/20251205_004119_ZGp-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/20251205_004119_ZGp-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/20251205_004119_ZGp-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/20251205_004119_ZGp-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/20251206_220111_8R7-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/20251206_220111_8R7-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/20251206_220111_8R7-playback.txt_timeline_window_11.png)

![时间轴图](plots/timelines/20251206_220111_8R7-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/20251206_220111_8R7-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/20251206_220111_8R7-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/20251206_220111_8R7-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/20251206_220111_8R7-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/20251206_220111_8R7-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/20251206_220111_8R7-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/20251206_220111_8R7-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/20251206_234532_8R7-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/20251206_234532_8R7-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/20251206_234532_8R7-playback.txt_timeline_window_11.png)

![时间轴图](plots/timelines/20251206_234532_8R7-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/20251206_234532_8R7-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/20251206_234532_8R7-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/20251206_234532_8R7-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/20251206_234532_8R7-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/20251206_234532_8R7-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/20251206_234532_8R7-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/20251206_234532_8R7-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/20251207_223333_vXS-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/20251208_000505_Tb7-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/20251208_012110_Tb7-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/20251208_012110_Tb7-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/20251208_012110_Tb7-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/20251208_012110_Tb7-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/20251208_012110_Tb7-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/20251208_012110_Tb7-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/20251208_012110_Tb7-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/20251208_012110_Tb7-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/20251208_012110_Tb7-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/20251208_012110_Tb7-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/20251208_224754_nXj-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/down_20251203_230356_b6x-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/down_20251203_230356_b6x-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/down_20251203_230356_b6x-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/down_20251203_230356_b6x-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/down_20251203_230356_b6x-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/down_20251203_230356_b6x-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/down_20251204_001204_b6x-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/down_20251204_001204_b6x-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/down_20251204_001204_b6x-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/down_20251204_001204_b6x-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/down_20251204_001204_b6x-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/down_20251204_001204_b6x-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/down_20251204_001204_b6x-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/down_20251204_001204_b6x-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/down_20251204_001204_b6x-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/down_20251204_001204_b6x-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_11.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_12.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_13.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_14.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_15.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_16.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_17.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_18.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_19.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_20.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/down_20251204_012208_co0-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/down_20251204_222027_ZGp-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/down_20251204_222027_ZGp-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/down_20251204_222027_ZGp-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/down_20251204_222027_ZGp-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/down_20251204_222027_ZGp-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/down_20251204_222027_ZGp-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/down_20251204_222027_ZGp-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/down_20251204_222027_ZGp-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/down_20251204_222027_ZGp-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/down_20251204_222027_ZGp-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/down_20251204_232415_ZGp-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/down_20251204_232415_ZGp-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/down_20251205_004119_ZGp-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/down_20251205_004119_ZGp-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/down_20251205_004119_ZGp-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/down_20251205_004119_ZGp-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/down_20251205_004119_ZGp-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/down_20251205_004119_ZGp-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/down_20251205_004119_ZGp-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/down_20251205_004119_ZGp-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/down_20251205_004119_ZGp-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/down_20251205_004119_ZGp-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/down_20251206_220111_8R7-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/down_20251206_220111_8R7-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/down_20251206_220111_8R7-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/down_20251206_220111_8R7-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/down_20251206_220111_8R7-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/down_20251206_220111_8R7-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/down_20251206_220111_8R7-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/down_20251206_220111_8R7-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/down_20251206_220111_8R7-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/down_20251206_220111_8R7-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/down_20251206_234532_8R7-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/down_20251206_234532_8R7-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/down_20251206_234532_8R7-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/down_20251206_234532_8R7-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/down_20251206_234532_8R7-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/down_20251206_234532_8R7-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/down_20251206_234532_8R7-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/down_20251206_234532_8R7-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/down_20251206_234532_8R7-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/down_20251206_234532_8R7-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/down_20251207_223333_vXS-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/down_20251208_000505_Tb7-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/down_20251208_012110_Tb7-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/down_20251208_012110_Tb7-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/down_20251208_012110_Tb7-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/down_20251208_012110_Tb7-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/down_20251208_012110_Tb7-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/down_20251208_012110_Tb7-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/down_20251208_012110_Tb7-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/down_20251208_012110_Tb7-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/down_20251208_012110_Tb7-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/down_20251208_012110_Tb7-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/down_20251208_224754_nXj-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/down_20251208_224754_nXj-playback_processed.csv_timeline_window_1.png)

![时间轴图](plots/timelines/up_20251203_230356_b6x-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/up_20251203_230356_b6x-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/up_20251203_230356_b6x-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/up_20251203_230356_b6x-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/up_20251203_230356_b6x-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/up_20251203_230356_b6x-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/up_20251204_001204_b6x-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/up_20251204_001204_b6x-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/up_20251204_001204_b6x-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/up_20251204_001204_b6x-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/up_20251204_001204_b6x-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/up_20251204_001204_b6x-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/up_20251204_001204_b6x-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/up_20251204_001204_b6x-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/up_20251204_001204_b6x-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/up_20251204_001204_b6x-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_11.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_12.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_13.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_14.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_15.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_16.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_17.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_18.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_19.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_20.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/up_20251204_012208_co0-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/up_20251204_222027_ZGp-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/up_20251204_222027_ZGp-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/up_20251204_222027_ZGp-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/up_20251204_222027_ZGp-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/up_20251204_222027_ZGp-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/up_20251204_222027_ZGp-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/up_20251204_222027_ZGp-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/up_20251204_222027_ZGp-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/up_20251204_222027_ZGp-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/up_20251204_222027_ZGp-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/up_20251204_232415_ZGp-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/up_20251204_232415_ZGp-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/up_20251205_004119_ZGp-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/up_20251205_004119_ZGp-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/up_20251205_004119_ZGp-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/up_20251205_004119_ZGp-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/up_20251205_004119_ZGp-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/up_20251205_004119_ZGp-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/up_20251205_004119_ZGp-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/up_20251205_004119_ZGp-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/up_20251205_004119_ZGp-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/up_20251205_004119_ZGp-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/up_20251206_220111_8R7-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/up_20251206_220111_8R7-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/up_20251206_220111_8R7-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/up_20251206_220111_8R7-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/up_20251206_220111_8R7-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/up_20251206_220111_8R7-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/up_20251206_220111_8R7-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/up_20251206_220111_8R7-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/up_20251206_220111_8R7-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/up_20251206_220111_8R7-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/up_20251206_234532_8R7-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/up_20251206_234532_8R7-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/up_20251206_234532_8R7-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/up_20251206_234532_8R7-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/up_20251206_234532_8R7-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/up_20251206_234532_8R7-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/up_20251206_234532_8R7-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/up_20251206_234532_8R7-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/up_20251206_234532_8R7-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/up_20251206_234532_8R7-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/up_20251207_223333_vXS-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/up_20251208_000505_Tb7-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/up_20251208_012110_Tb7-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/up_20251208_012110_Tb7-playback.txt_timeline_window_10.png)

![时间轴图](plots/timelines/up_20251208_012110_Tb7-playback.txt_timeline_window_2.png)

![时间轴图](plots/timelines/up_20251208_012110_Tb7-playback.txt_timeline_window_3.png)

![时间轴图](plots/timelines/up_20251208_012110_Tb7-playback.txt_timeline_window_4.png)

![时间轴图](plots/timelines/up_20251208_012110_Tb7-playback.txt_timeline_window_5.png)

![时间轴图](plots/timelines/up_20251208_012110_Tb7-playback.txt_timeline_window_6.png)

![时间轴图](plots/timelines/up_20251208_012110_Tb7-playback.txt_timeline_window_7.png)

![时间轴图](plots/timelines/up_20251208_012110_Tb7-playback.txt_timeline_window_8.png)

![时间轴图](plots/timelines/up_20251208_012110_Tb7-playback.txt_timeline_window_9.png)

![时间轴图](plots/timelines/up_20251208_224754_nXj-playback.txt_timeline_window_1.png)

![时间轴图](plots/timelines/up_20251208_224754_nXj-playback_processed.csv_timeline_window_1.png)

## 7. 原始数据样本可视化

以下是每个行为类别的典型样本对应的原始时延和丢包率可视化：

![典型样本](plots/behavior_samples/typical_samples.png)

## 8. 不同文件行为类别占比统计

| 文件名 | 稳定 (%) | 强突发 (%) | 瞬时峰值 (%) | 高丢包低延迟 (%)|
|----------|--- |--- |--- |--- |
| unknown | 15.0 | 60.0 | 10.0 | 15.0|

### 8.1 文件总窗口数

| 文件名 | 总窗口数 |
|----------|----------|
| unknown | 20 |

