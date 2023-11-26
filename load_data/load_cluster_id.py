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

a = [-1] * len(from_addr_dict)
for i, row in tqdm(label_dict.iterrows()):
    if row['label'] ==1:
        index = row['node1']
        a[index] = row['cluster_id_encoded']


df = pd.DataFrame(a, columns=['cluster_id_encoded'])
df.to_csv('cluster_id.csv', index=False)