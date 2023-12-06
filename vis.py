import numpy as np
import pandas as pd
import statistics


cluster_id = pd.read_csv('load_data\cluster_id.csv')
labels = np.array(cluster_id)
cluster_result = pd.read_csv('clustered_result_3eps.csv')
result = np.array(cluster_result)

# addr_dict = pd.read_csv('data/addr_feat.csv')
# edge_dict = pd.read_csv('data/eth_edge.csv')

# from_addr_dict = {addr: i for i, addr in enumerate(addr_dict['from_addr'].unique())}

# edge_index = edge_dict[['origin_from_addr', 'to_addr']].applymap(lambda x: from_addr_dict[x])

# print(labels[labels != -1])
# [186476 505681 ]
# print(labels[labels == 505681])
vis_label = 505681
label_example_index = [index for index, (label, res) in enumerate(zip(labels, result)) if label == vis_label]
label_example = [res[0] for label, res in zip(labels, result) if label == vis_label]
new_cluster_id = statistics.mode(label_example)
print(new_cluster_id)
result_example_index = [index for index, res in enumerate(result) if res == new_cluster_id]

whole_vis_node_index = list(set(label_example_index + result_example_index))

print(len(label_example_index))
print(len(result_example_index))
print(len(whole_vis_node_index))

