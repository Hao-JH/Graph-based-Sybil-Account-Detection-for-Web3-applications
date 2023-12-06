import numpy as np
import pandas as pd


cluster_id = pd.read_csv('load_data\cluster_id.csv')
labels = np.array(cluster_id)
cluster_result = pd.read_csv('clustered_result_10eps_min20.csv')
result = np.array(cluster_result)

tp = np.sum((labels != -1) & (result != -1))   
fp = np.sum((labels == -1) & (result != -1))  
tn = np.sum((labels == -1) & (result == -1))  
fn = np.sum((labels != -1) & (result == -1))  

# Calculate precision
precision = tp / (tp + fp)

# Calculate recall
recall = tp / (tp + fn)

# Print the results
print("Precision:", precision)
print("Recall:", recall)