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
pos_pairs['pair'] = pos_pairs['node1'].apply(lambda x: list(itertools.combinations(x, 2)))
pos_pairs = pos_pairs.explode('pair')
pos_pairs[['node1', 'node2']] = pd.DataFrame(pos_pairs['pair'].tolist(), index=pos_pairs.index)
pos_pairs = pos_pairs.drop(['node1'], axis=1)

# 生成负例对
k = 1
neg_pairs = label_dict[label_dict['cluster_id_encoded'].notnull()]
neg_pairs = neg_pairs.groupby('cluster_id_encoded')['node1'].apply(list).reset_index()
neg_pairs = neg_pairs.sample(frac=k, replace=True)
neg_pairs['node1'] = neg_pairs['node1'].apply(lambda x: np.random.choice(x, size=2, replace=True))
neg_pairs[['node1', 'node2']] = pd.DataFrame(neg_pairs['node1'].tolist(), index=neg_pairs.index)
neg_pairs = neg_pairs.drop(['node1'], axis=1)

print("正例对数量：", len(pos_pairs))
print("负例对数量：", len(neg_pairs))

pos_pairs.to_csv('pos_pairs.csv', index=False)
neg_pairs.to_csv('neg_pairs.csv', index=False)