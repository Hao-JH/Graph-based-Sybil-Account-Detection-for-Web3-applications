import numpy as np
import os
import torch
import pandas as pd
import time
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import itertools

current_folder = os.path.dirname(os.path.abspath(__file__))
print(current_folder)
os.chdir(current_folder)

addr_dict = pd.read_csv('../data/addr_feat.csv')
label_dict = pd.read_csv('../data/starknet_label/starknet_addr_score_label.csv')
from_addr_dict = {addr: i for i, addr in enumerate(addr_dict['from_addr'].unique())}
label_dict['node1'] = label_dict['node_addr'].map(from_addr_dict)
label_dict['cluster_id'] = label_dict['cluster_id'].astype(str)
le = LabelEncoder()
label_dict['cluster_id_encoded'] = le.fit_transform(label_dict['cluster_id'])
label_dict['cluster_id_encoded'] = label_dict['cluster_id_encoded'].astype(int)

print(label_dict.head())

pos_pairs = label_dict[(label_dict['label'] == 1) & (label_dict['cluster_id_encoded'].notnull())]
pos_pairs = pos_pairs.groupby('cluster_id_encoded')['node1'].apply(list).reset_index()
# pos_pairs['pair'] = pos_pairs['node1'].apply(lambda x: list(itertools.combinations(x, 2)))
pos_pairs['pair'] = pos_pairs['node1'].apply(lambda x: list(zip(x, x[1:] + [x[0]])))
split_ratio = 0.7
split_point = int(len(pos_pairs) * split_ratio)
train_pos_pairs = pos_pairs[:split_point]
test_pos_pairs = pos_pairs[split_point:]

train_pos_pairs = train_pos_pairs.explode('pair')
test_pos_pairs = test_pos_pairs.explode('pair')
train_pos_pairs[['node1', 'node2']] = pd.DataFrame(train_pos_pairs['pair'].tolist(), index=train_pos_pairs.index)
test_pos_pairs[['node1', 'node2']] = pd.DataFrame(test_pos_pairs['pair'].tolist(), index=test_pos_pairs.index)
train_pos_pairs = train_pos_pairs.drop(['pair'], axis=1)
test_pos_pairs = test_pos_pairs.drop(['pair'], axis=1)

# 生成负例对
# k = 2
# neg_pairs = pd.DataFrame(columns=pos_pairs.columns)

# for _ in range(k):
#     neg_pairs_subset = pos_pairs.copy()
#     neg_pairs_subset['node2'] = np.random.randint(len(from_addr_dict), size=len(neg_pairs_subset)).astype(int)
#     neg_pairs = pd.concat([neg_pairs, neg_pairs_subset], ignore_index=True)

print("正例对数量：", len(pos_pairs))
# print("负例对数量：", len(neg_pairs))

train_pos_pairs.to_csv('train_pos_pairs.csv', index=False)
test_pos_pairs.to_csv('test_pos_pairs.csv', index=False)
#neg_pairs.to_csv('neg_pairs.csv', index=False)