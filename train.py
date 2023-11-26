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

# print(node_features.shape)
# print(edge_features.shape)
# print(edge_index.shape)
# print(train_pos_pairs.shape)
# print(test_pos_pairs.shape)

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
# print(out.shape)

num_epochs = 3
batch_size = 4096
k = 2

dataset = TensorDataset(train_pos_pairs[:,0],train_pos_pairs[:,1])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# testdataset = TensorDataset(test_pos_pairs[:,0],test_pos_pairs[:,1])
# testdataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-4)

def compute_loss(model, node_features, edge_index, edge_features, node1, node2, k):
    out = model(node_features, edge_index)

    N = torch.exp(torch.cosine_similarity(out[node1], out[node2]))
    D = torch.zeros_like(N)
    for i in range(k):
        D +=torch.exp(torch.cosine_similarity(
                 out[node1], 
                 out[torch.randint(0, node_features.shape[0], size=node1.shape)]
             ))
    loss = -torch.mean(torch.log(N/(N+D)))
    return loss
    # pos_loss = torch.mean(torch.log(F.sigmoid(torch.cosine_similarity(out[node1], out[node2]))))
    # neg_loss = 0
    # for i in range(k):
    #     neg_loss += torch.mean(torch.log(
    #         F.sigmoid(torch.cosine_similarity(
    #             out[node1], 
    #             out[torch.randint(0, node_features.shape[0], size=node1.shape)]
    #         ))))
    # loss = -1 * (pos_loss - neg_loss)
    # return loss

train_losses = []
test_losses = []
train_sim = []
test_sim = []
# train_neg_sim = []
# test_neg_sim = []

train_losses.append(compute_loss(model,node_features,edge_index,edge_features,train_pos_pairs[:,0],train_pos_pairs[:,1],k).item())
test_losses.append(compute_loss(model,node_features,edge_index,edge_features,test_pos_pairs[:,0],test_pos_pairs[:,1],k).item())

for epoch in range(num_epochs):
    model.train()
    train_loss = 0 
    for batch in tqdm(dataloader):
        node1,node2 =batch
        optimizer.zero_grad()
        loss = compute_loss(model,node_features,edge_index,edge_features,node1,node2,k) 
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(dataloader)
    model.eval()
    with torch.no_grad():
        test_loss = compute_loss(model,node_features,edge_index,edge_features,test_pos_pairs[:,0],test_pos_pairs[:,1],k)
    print('Epoch [{}/{}], train Loss: {:.4f}, test Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss,test_loss.item()))

    # writer.add_scalar('train_loss', train_loss, epoch)
    # writer.add_scalar('test_loss', test_loss.item(), epoch)
    out = model(node_features,edge_index)
    train_sim.append(torch.mean(torch.cosine_similarity(out[train_pos_pairs[:,0]], out[train_pos_pairs[:,1]])).item())
    test_sim.append(torch.mean(torch.cosine_similarity(out[test_pos_pairs[:,0]], out[test_pos_pairs[:,1]])).item())
    # train_neg_sim.append(torch.mean(torch.cosine_similarity(out[train_pos_pairs[:,0]], out[torch.randint(0, node_features.shape[0], size=train_pos_pairs[:,0].shape)])).item())
    # test_neg_sim.append(torch.mean(torch.cosine_similarity(out[test_pos_pairs[:,0]], out[torch.randint(0, node_features.shape[0], size=test_pos_pairs[:,0].shape)])).item())
    train_losses.append(train_loss)
    test_losses.append(test_loss.item())
    torch.save(model.state_dict(), 'model_params.pth')

    
np.savez('losses_k5.npz', train_losses=train_losses, test_losses=test_losses,train_sim=train_sim,test_sim=test_sim)
plt.figure()
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve_k5.png')
plt.figure()
plt.plot(train_sim, label='train sim')
plt.plot(test_sim, label='test sim')
# plt.plot(train_neg_sim, label='train neg sim')
# plt.plot(test_neg_sim, label='test neg sim')
plt.xlabel('Epoch')
plt.ylabel('sim')
plt.legend()
plt.savefig('sim_curve_k5.png')

plt.show()



torch.save(model.state_dict(), 'model_params.pth')
