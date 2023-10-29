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
    def __init__(self, in_channels,out_channels, weight_decay=1e-4):
        super(GCNConvModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 2*in_channels)
        )
        self.conv = GCNConv(in_channels=2*in_channels, out_channels=out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = GCNConv(in_channels=out_channels, out_channels=out_channels)
        self.sig1 =nn.Sigmoid()
        self.weight_decay = weight_decay

    def forward(self, node_features, edge_index):
        node_features = self.fc(node_features)
        out = self.conv(node_features, edge_index)
        out = self.relu1(out)
        out = self.conv2(out, edge_index)
        out = self.sig1(out)
        return out
    
    def l2_regularization(self,device):
        l2_reg = torch.tensor(0., device=device)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param)
        return self.weight_decay * l2_reg