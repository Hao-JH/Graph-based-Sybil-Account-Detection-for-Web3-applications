import os
import pandas as pd

current_folder = os.path.dirname(os.path.abspath(__file__))
print(current_folder)
os.chdir(current_folder)

label_dict = pd.read_csv('starknet_label/starknet_addr_score_label.csv')
print(label_dict.info())
print(label_dict.iloc[0])
print(label_dict.iloc[1])

'''
RangeIndex: 1476252 entries, 0 to 1476251
Data columns (total 4 columns):
 #   Column      Non-Null Count    Dtype  
---  ------      --------------    -----  
 0   node_addr   1476252 non-null  object 
 1   cluster_id  1281629 non-null  object 
 2   score       1281629 non-null  float64
 3   label       1476252 non-null  float64
dtypes: float64(2), object(2)
memory usage: 45.1+ MB
None
node_addr     0x11bde1fe991be923ac9cbedb3784d0b2147f952cfc57...
cluster_id                                                    0
score                                                  5.408163
label                                                       1.0
Name: 0, dtype: object
node_addr     0x35ffcf966e8886a8aef28bea5ff2a97585e6538a99e8...
cluster_id                                                    0
score                                                  5.408163
label                                                       1.0
Name: 1, dtype: object
'''