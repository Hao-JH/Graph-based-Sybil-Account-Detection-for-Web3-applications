import torch
import os
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from  model.gat import GATModel,GCNConvModel
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


current_folder = os.path.dirname(os.path.abspath(__file__))
print(current_folder)
os.chdir(current_folder)

node_features = torch.load('load_data/features.pt')
edge_features = torch.load('load_data/edge_features.pt')
edge_index = torch.load('load_data/edge_index.pt')

train_pos_pairs = torch.load('load_data/train_pos_pairs.pt')
test_pos_pairs = torch.load('load_data/test_pos_pairs.pt')



data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
in_channels = node_features.shape[1]
hidden_channels = 32
edge_dim = edge_features.shape[1]
out_channels = 32

gatmodel = GATModel(in_channels=in_channels,hidden_channels=hidden_channels,num_layers=1,out_channels=out_channels,edge_dim=edge_dim)
gcnmodel = GCNConvModel(in_channels=in_channels,out_channels=out_channels)
model = gcnmodel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
node_features = node_features.to(device)
edge_index = edge_index.to(device)
edge_features = edge_features.to(device)

train_pos_pairs = train_pos_pairs.to(device)
test_pos_pairs = test_pos_pairs.to(device)

state_dict = torch.load('model_params.pth')
model.load_state_dict(state_dict)
out = model(node_features,edge_index)
cluster_id = pd.read_csv('load_data\cluster_id.csv')
labels = np.array(cluster_id)


# 0.018972053269412682
eps = 0.018972053269412682

dbscan = DBSCAN(eps=eps*10, min_samples=20)

clustered_labels = dbscan.fit_predict(node_features.cpu().numpy())

print(clustered_labels)

# 创建一个包含标签的DataFrame
df = pd.DataFrame({'Cluster_Labels': clustered_labels})

# 将DataFrame保存为CSV文件default min samples = 5 default eps = 5
df.to_csv('clustered_result_10eps_min20.csv', index=False)

#####problem in dbscan