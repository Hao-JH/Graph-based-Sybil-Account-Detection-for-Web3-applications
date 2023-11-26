import numpy as np
import pandas as pd


cluster_id = pd.read_csv('load_data\cluster_id.csv')
labels = np.array(cluster_id)
cluster_result = pd.read_csv('clustered_result.csv')
result = np.array(cluster_result)

# 计算混淆矩阵
tp = np.sum((labels != -1) & (result != -1))  # 真正例
fp = np.sum((labels == -1) & (result != -1))  # 假正例
tn = np.sum((labels == -1) & (result == -1))  # 真负例
fn = np.sum((labels != -1) & (result == -1))  # 假负例


# 计算准确率
precision = tp / (tp + fp)

# 计算召回率
recall = tp / (tp + fn)

# 打印结果
print("准确率:", precision)
print("召回率:", recall)