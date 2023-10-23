import torch
import os
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from  model.gat import GATModel
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

current_folder = os.path.dirname(os.path.abspath(__file__))
print(current_folder)
os.chdir(current_folder)

node_features = torch.load('load_data/features.pt')
edge_features = torch.load('load_data/edge_features.pt')
edge_index = torch.load('load_data/edge_index.pt')
pos_pairs = torch.load('load_data/pos_pairs.pt')
neg_pairs = torch.load('load_data/neg_pairs.pt')

print(node_features.shape)
print(edge_features.shape)
print(edge_index.shape)
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
in_channels = node_features.shape[1]
hidden_channels = 16
edge_dim = edge_features.shape[1]
out_channels = 16

model = GATModel(in_channels=in_channels,hidden_channels=hidden_channels,num_layers=1,out_channels=out_channels,edge_dim=edge_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
node_features = node_features.to(device)
edge_index = edge_index.to(device)
edge_features = edge_features.to(device)
pos_pairs = pos_pairs.to(device)
neg_pairs = neg_pairs.to(device)

out = model(node_features,edge_index,edge_features)
#state_dict = torch.load('model_params.pth')
#model.load_state_dict(state_dict)
print(out.shape)

num_epochs = 3
batch_size = 128
k =2

dataset = TensorDataset(pos_pairs[:,0],pos_pairs[:,1])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    with tqdm(total=len(dataloader)) as t:   
        for batch in dataloader:
            node1,node2 =batch
            optimizer.zero_grad()
            out = model(node_features, edge_index, edge_features)
            pos_loss = torch.mean(F.sigmoid(torch.cosine_similarity(out[node1], out[node2])))
            neg_loss = 0
            for i in range(k):
                neg_loss += torch.mean(
                    F.sigmoid(torch.cosine_similarity(
                        out[node1], 
                        out[torch.randint(0, node_features.shape[0], size=node1.shape)]
                    )))
            loss = -1 * (pos_loss - neg_loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            t.set_description(desc="Epoch %i"%i)
            t.set_postfix(loss=loss.data.item())
            t.update(1)

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

torch.save(model.state_dict(), 'model_params.pth')