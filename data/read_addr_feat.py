import os
import pandas as pd

current_folder = os.path.dirname(os.path.abspath(__file__))
print(current_folder)
os.chdir(current_folder)

addr_dict = pd.read_csv('addr_feat.csv')
print(addr_dict.info())
print(addr_dict.iloc[0])
print(addr_dict.iloc[1])


'''
RangeIndex: 1476197 entries, 0 to 1476196
Data columns (total 24 columns):
 #   Column                  Non-Null Count    Dtype
---  ------                  --------------    -----
 0   from_addr               1476197 non-null  object
 1   all_hash_cnt            1476197 non-null  int64
 2   all_date_cnt            1476197 non-null  int64
 3   all_method_cnt          1476197 non-null  int64 //with which contract
 4   all_to_addr_cnt         1476197 non-null  int64 // num of contract
 5   all_min_dt              1476197 non-null  object
 6   all_max_dt              1476197 non-null  object
 7   sum_actual_fee          1476197 non-null  float64 // gas fee
 8   first_nonce             1336436 non-null  float64 //byl
 9   first_block_date        1336436 non-null  object
 10  first_transaction_hash  1336436 non-null  object
 11  first_calldata          1336436 non-null  object
 12  frist_contracts_sha     1336430 non-null  object
 13  first_functions_sha     1336430 non-null  object
 14  last_nonce              1475132 non-null  float64
 15  last_block_date         1475132 non-null  object
 16  last_transaction_hash   1475132 non-null  object
 17  last_calldata           1475132 non-null  object
 18  last_contracts_sha      1475130 non-null  object
 19  last_functions_sha      1475130 non-null  object
 20  eth_count_out_sum       1476196 non-null  float64
 21  eth_count_out_avg       1476196 non-null  float64
 22  eth_count_out_min       1476196 non-null  float64
 23  eth_count_out_max       1476196 non-null  float64
dtypes: float64(7), int64(4), object(13)
memory usage: 270.3+ MB
None
from_addr                 0x5f788435820afae4ea1bc835e0d1f9cfcd38e6f344d8...
all_hash_cnt                                                             56
all_date_cnt                                                             17
all_method_cnt                                                           10
all_to_addr_cnt                                                          16
all_min_dt                                                       2023-05-22
all_max_dt                                                       2023-08-09
sum_actual_fee                                                     0.018732
first_nonce                                                             1.0
first_block_date                                                 2023-05-22
first_transaction_hash    0x61065af33e721e36426dec82bb950cc74cf82fc83b9c...
first_calldata            [0x2, 0x49d36570d4e46f48e99674bd3fcc84644ddd6b...
frist_contracts_sha       f660737ccfdc46ecf854bc5919af26247f9bffbc8e061d...
first_functions_sha       d1f604586d6b2a38382c9e8b5284e74f1f2299160c794c...
last_nonce                                                             56.0
last_block_date                                                  2023-08-09
last_transaction_hash     0x37e2767048eb74e183d0061a7bf9de901beea60dbe97...
last_calldata             [0x1, 0x173f81c529191726c6e7287e24626fe24760ac...
last_contracts_sha        8c19151a1b04b137c43580ddb0a475c5db70cf46bcec6a...
last_functions_sha        e7aa8b7c8d4fed4a0156dc56b23d906073f18ba8f351ab...
eth_count_out_sum                                                  1.134017
eth_count_out_avg                                                  0.013186
eth_count_out_min                                                  0.000016
eth_count_out_max                                                     0.088
'''