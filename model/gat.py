import torch.nn as nn
from torch_geometric.nn import GATConv,SAGEConv,GCNConv
import torch

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
    def __init__(self, in_channels,out_channels,dropout_rate=0.1):
        super(GCNConvModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32,64)
        )
        self.conv1 = GCNConv(in_channels=64, out_channels=64)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = GCNConv(in_channels=64, out_channels=64)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = GCNConv(in_channels=64, out_channels=64)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.conv4 = GCNConv(in_channels=64, out_channels=64)
        self.final_linear = nn.Linear(64,out_channels)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, node_features, edge_index):
        node_features = self.fc(node_features)
        out1 = self.conv1(node_features, edge_index)
        out1 = self.relu1(out1)
        out2 = self.conv2(out1, edge_index)
        out2 = self.relu2(out2)
        out2 = out2 + out1
        out3 = self.conv3(out2, edge_index)
        out3 = self.relu3(out3)
        out4 = self.conv4(out3, edge_index)
        out4 = out4 + out2
        out4 = self.final_linear(out4)
        return out4
    