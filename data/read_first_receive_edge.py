import os
import pandas as pd

current_folder = os.path.dirname(os.path.abspath(__file__))
print(current_folder)
os.chdir(current_folder)
first_receive_edge = pd.read_csv('first_receive_edge.csv')
print(first_receive_edge.info())
print(first_receive_edge.iloc[0])

'''
RangeIndex: 1476001 entries, 0 to 1476000
Data columns (total 8 columns):
 #   Column             Non-Null Count    Dtype  
---  ------             --------------    -----
 0   origin_from_addr   1476001 non-null  object
 1   to_addr            1476001 non-null  object
 2   txn_hash           1476001 non-null  object
 3   block_number       1476001 non-null  int64
 4   token_count        1476001 non-null  float64 ether
 5   dt                 1476001 non-null  int64 datetime
 6   transaction_index  1476001 non-null  int64 
 7   rank1              1476001 non-null  int64 
dtypes: float64(1), int64(4), object(3)
memory usage: 90.1+ MB
None
origin_from_addr     0x3a20d4f7b4229e7c4863dab158b4d076d7f454b893d9...
to_addr              0x45072e3874cef1d90a1703082a1b11b1aa5b2f2d052d...
txn_hash             0x56c419e04dd2249ff4ae2354c50346ff2ef9b2586a77...
block_number                                                    147750
token_count                                                      0.055
dt                                                            20230809
transaction_index                                                   23
rank1                                                                1
'''