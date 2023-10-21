import numpy as np
import os
import torch
import pandas as pd
import time
from tqdm import tqdm

current_folder = os.path.dirname(os.path.abspath(__file__))
print(current_folder)
os.chdir(current_folder)

addr_dict = pd.read_csv('../data/addr_feat.csv')
edge_dict = pd.read_csv('../data/eth_edge.csv')

from_addr_dict = {addr: i for i, addr in enumerate(addr_dict['from_addr'].unique())}

edge_index = edge_dict[['origin_from_addr', 'to_addr']].applymap(lambda x: from_addr_dict[x])

edge_index = torch.tensor(edge_index.values, dtype=torch.long).T

torch.save(edge_index, 'edge_index.pt')
print(edge_index)

feature_cols = ['min_token_count', 'max_token_count', 'avg_token_count', 'hash_cnt', 'dt_cnt']
features = edge_dict[feature_cols]

features = features.apply(
    lambda x: (x - x.mean()) / (x.std()))

features = features.fillna(0)

features = torch.tensor(features.values, dtype=torch.float)

torch.save(features, 'edge_features.pt')

print(features)


'''


a1a2 attention1= f(e1,a1,a2, W) = relu(a1w1 + ...+ a11w11 + a2w12+  + e5w27 )

a1a3 attention2= f(e2,a1,a3, W) = 

softmax e^f(e,a1,a2, W)/sigma e^f(e,a1,ai, W)

attention1 * a2e1 + attention2 *a3e2
'''
