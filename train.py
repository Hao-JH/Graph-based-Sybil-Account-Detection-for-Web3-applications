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

current_folder = os.path.dirname(os.path.abspath(__file__))
print(current_folder)
os.chdir(current_folder)

node_features = torch.load('load_data/features.pt')
edge_features = torch.load('load_data/edge_features.pt')
edge_index = torch.load('load_data/edge_index.pt')
# pos_pairs = torch.load('load_data/pos_pairs.pt')
# neg_pairs = torch.load('load_data/neg_pairs.pt')
train_pos_pairs = torch.load('load_data/train_pos_pairs.pt')
test_pos_pairs = torch.load('load_data/test_pos_pairs.pt')

print(node_features.shape)
print(edge_features.shape)
print(edge_index.shape)
print(train_pos_pairs.shape)
print(test_pos_pairs.shape)

data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
in_channels = node_features.shape[1]
hidden_channels = 16
edge_dim = edge_features.shape[1]
out_channels = 16

gatmodel = GATModel(in_channels=in_channels,hidden_channels=hidden_channels,num_layers=1,out_channels=out_channels,edge_dim=edge_dim)
gcnmodel = GCNConvModel(in_channels=in_channels,out_channels=out_channels)
model = gcnmodel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
node_features = node_features.to(device)
edge_index = edge_index.to(device)
edge_features = edge_features.to(device)
# pos_pairs = pos_pairs.to(device)
# neg_pairs = neg_pairs.to(device)
train_pos_pairs = train_pos_pairs.to(device)
test_pos_pairs = test_pos_pairs.to(device)

out = model(node_features,edge_index)
# state_dict = torch.load('model_params.pth')
# model.load_state_dict(state_dict)
print(out.shape)

num_epochs = 10
batch_size = 64
k = 5

dataset = TensorDataset(train_pos_pairs[:,0],train_pos_pairs[:,1])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# testdataset = TensorDataset(test_pos_pairs[:,0],test_pos_pairs[:,1])
# testdataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def compute_loss(model, node_features, edge_index, edge_features, node1, node2, k):
    out = model(node_features, edge_index)
    pos_loss = torch.mean(F.sigmoid(torch.cosine_similarity(out[node1], out[node2])))
    neg_loss = 0
    for i in range(k):
        neg_loss += torch.mean(
            F.sigmoid(torch.cosine_similarity(
                out[node1], 
                out[torch.randint(0, node_features.shape[0], size=node1.shape)]
            )))
    loss = -1 * (pos_loss - neg_loss)
    return loss

train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0 
    for batch in tqdm(dataloader):
        node1,node2 =batch
        optimizer.zero_grad()
        # out = model(node_features, edge_index, edge_features)
        # pos_loss = torch.mean(F.sigmoid(torch.cosine_similarity(out[node1], out[node2])))
        # neg_loss = 0
        # for i in range(k):
        #     neg_loss += torch.mean(
        #         F.sigmoid(torch.cosine_similarity(
        #             out[node1], 
        #             out[torch.randint(0, node_features.shape[0], size=node1.shape)]
        #         )))
        # loss = -1 * (pos_loss - neg_loss)
        loss = compute_loss(model,node_features,edge_index,edge_features,node1,node2,k)
        l2_reg = model.l2_regularization(device=device)
        loss = loss + l2_reg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(dataloader)
    model.eval()
    with torch.no_grad():
        test_loss = compute_loss(model,node_features,edge_index,edge_features,test_pos_pairs[:,0],test_pos_pairs[:,1],k)
    print('Epoch [{}/{}], train Loss: {:.4f}, test Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss,test_loss.item()))

    train_losses.append(train_loss)
    test_losses.append(test_loss.item())

    
np.savez('losses.npz', train_losses=train_losses, test_losses=test_losses)
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')
plt.show()



torch.save(model.state_dict(), 'model_params.pth')

