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

# 选择要作为节点特征的列
feature_cols = ['all_hash_cnt', 'all_date_cnt', 'all_method_cnt', 'all_to_addr_cnt',
                 'sum_actual_fee', 'first_nonce', 'last_nonce',
                 'eth_count_out_sum', 'eth_count_out_avg', 'eth_count_out_min', 'eth_count_out_max']

date_cols = ['all_min_dt', 'all_max_dt','first_block_date','last_block_date']

features = addr_dict[feature_cols]

# for col in tqdm(date_cols):
#     for element in features[col]:
#         if type(element) == str:
#             element = time.mktime(time.strptime(element, '%Y-%m-%d'))

# 对特征进行标准化
#features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
features = features.apply(
    lambda x: (x - x.mean()) / (x.std()))
print("norm finish")

#填充0
features = features.fillna(0)
print("fill 0 finish")

# 将 pd 数组转换为 PyTorch 张量
features = torch.tensor(features.values, dtype=torch.float)

print('start saving')
torch.save(features, 'features.pt')
print('finish saving')

print(features)