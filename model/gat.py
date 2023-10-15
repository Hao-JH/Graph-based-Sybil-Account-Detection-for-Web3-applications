import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttention(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        self.attention = nn.Parameter(torch.FloatTensor(2*out_features, 1))
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x, adj, edge_attr):
        support = torch.mm(x, self.weight)
        adj = adj.unsqueeze(2)
        edge_attr = edge_attr.unsqueeze(2)
        h = torch.cat([torch.bmm(adj, support.unsqueeze(2)).squeeze(2), torch.bmm(edge_attr, support.unsqueeze(2)).squeeze(2)], dim=1)
        attention = torch.matmul(h, self.attention).squeeze(1)
        attention = F.softmax(attention, dim=0)
        h = torch.mul(support, attention.unsqueeze(1))
        h = torch.sum(h, dim=1)
        return F.relu(h)

class GraphNet(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GraphNet, self).__init__()
        self.ga1 = GraphAttention(in_features, hidden_features)
        self.ga2 = GraphAttention(hidden_features, out_features)

    def forward(self, x, adj, edge_attr):
        x = self.ga1(x, adj, edge_attr)
        x = F.dropout(x, training=self.training)
        x = self.ga2(x, adj, edge_attr)
        return F.log_softmax(x, dim=1)


'''
# 创建GraphNet模型
graphnet = GraphNet(in_features=features.shape[1], hidden_features=16, out_features=labels.max().item()+1)

# 使用交叉熵损失函数和Adam优化器训练模型
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(graphnet.parameters(), lr=0.01, weight_decay=5e-4)
for epoch in range(200):
    graphnet.train()
    optimizer.zero_grad()
    output = graphnet(features, adj, edge_attr)
    loss = criterion(output[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()

    graphnet.eval()
    output = graphnet(features, adj, edge_attr)
    acc_train = accuracy(output[idx_train], labels[idx_train])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch {:03d} | Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f}'.format(epoch, loss.item(), acc_train.item(), acc_val.item()))
'''