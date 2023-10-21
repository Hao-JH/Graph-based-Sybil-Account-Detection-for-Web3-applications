import os
import pandas as pd

current_folder = os.path.dirname(os.path.abspath(__file__))
print(current_folder)
os.chdir(current_folder)
eth_edge = pd.read_csv('eth_edge.csv')
print(eth_edge.info())
print(eth_edge.iloc[0])

'''
RangeIndex: 1416170 entries, 0 to 1416169
Data columns (total 7 columns):
 #   Column            Non-Null Count    Dtype  
---  ------            --------------    -----  
 0   origin_from_addr  1416170 non-null  object 
 1   to_addr           1416170 non-null  object 
 2   min_token_count   1416170 non-null  float64 最低一笔消费
 3   max_token_count   1416170 non-null  float64
 4   avg_token_count   1416170 non-null  float64
 5   hash_cnt          1416170 non-null  int64  交易次数
 6   dt_cnt            1416170 non-null  int64  交易天数
dtypes: float64(3), int64(2), object(2)
memory usage: 75.6+ MB
None
origin_from_addr    0x3a20d4f7b4229e7c4863dab158b4d076d7f454b893d9...
to_addr             0x314d279dab0fac4a1e7f4c2e39ea45abffe192968503...
min_token_count                                                 0.255
max_token_count                                                 0.255
avg_token_count                                                 0.255
hash_cnt                                                            1
dt_cnt                                                              1
'''

'''

'''