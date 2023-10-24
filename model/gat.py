import torch.nn as nn
from torch_geometric.nn import GATConv,SAGEConv,GCNConv


class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, edge_dim):
        super(GATModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU()
        )
        self.conv = GATConv(in_channels=hidden_channels, hidden_channels=hidden_channels, num_layers=num_layers, out_channels=out_channels, edge_dim=edge_dim)

    def forward(self, node_features, edge_index, edge_features):
        node_features = self.fc(node_features)
        out = self.conv(node_features, edge_index, edge_features)
        return out
    
class GCNConvModel(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(GCNConvModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 2*in_channels),
            nn.ReLU()
        )
        self.conv = GCNConv(in_channels=2*in_channels, out_channels=out_channels)

    def forward(self, node_features, edge_index):
        node_features = self.fc(node_features)
        out = self.conv(node_features, edge_index)
        return out