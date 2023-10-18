import torch
import os
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


current_folder = os.path.dirname(os.path.abspath(__file__))
print(current_folder)
os.chdir(current_folder)

node_features = torch.load('load_data/features.pt')
edge_features = torch.load('load_data/edge_features.pt')
edge_index = torch.load('load_data/edge_index.pt')

print(node_features.shape)
print(edge_features.shape)
print(edge_index.shape)
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
in_channels = node_features.shape[1]
hidden_channels = 16
edge_dim = edge_features.shape[1]
out_channels = 16
model1 = GATConv(in_channels=in_channels,hidden_channels=hidden_channels,num_layers=1,out_channels=out_channels)

model2 = GATConv(in_channels=in_channels,hidden_channels=hidden_channels,num_layers=1,out_channels=out_channels,edge_dim=edge_dim)

out1 = model1(node_features, edge_index)

out2 = model2(node_features,edge_index,edge_features)

print(out1.shape)
print(out2.shape)