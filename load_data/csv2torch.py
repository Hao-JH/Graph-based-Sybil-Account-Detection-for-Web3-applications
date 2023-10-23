import pandas as pd
import torch
import os
import torch
import pandas as pd


current_folder = os.path.dirname(os.path.abspath(__file__))
print(current_folder)
os.chdir(current_folder)
# 读取csv文件
pos_pairs_df = pd.read_csv('pos_pairs.csv')
neg_pairs_df = pd.read_csv('neg_pairs.csv')

# 从DataFrame中提取node1和node2的索引
pos_pairs = pos_pairs_df[['node1', 'node2']].values
neg_pairs = neg_pairs_df[['node1', 'node2']].values

# 将索引转换为张量
pos_pairs = torch.tensor(pos_pairs, dtype=torch.long)
neg_pairs = torch.tensor(neg_pairs, dtype=torch.long)

torch.save(pos_pairs, 'pos_pairs.pt')
torch.save(neg_pairs, 'neg_pairs.pt')