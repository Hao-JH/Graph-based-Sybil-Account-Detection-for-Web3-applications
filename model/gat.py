import torch.nn as nn
from torch_geometric.nn import GATConv


class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, edge_dim):
        super(GATModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv = GATConv(in_channels=hidden_channels, hidden_channels=hidden_channels, num_layers=num_layers, out_channels=out_channels, edge_dim=edge_dim)

    def forward(self, node_features, edge_index, edge_features):
        node_features = self.fc(node_features)
        out = self.conv(node_features, edge_index, edge_features)
        return out