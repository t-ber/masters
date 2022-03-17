import torch
import scipy.io
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
import torch.nn.functional as F




class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.GCNConv(dataset.num_node_features, 16)
        self.conv2 = nn.GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

raw_data = scipy.io.loadmat('data/indian_pines.mat')
ground_truth = scipy.io.loadmat('data/indian_pines_gt.mat')
data = raw_data['indian_pines'] # use the key for data here
target = ground_truth['indian_pines_gt'] # use the key for target here

data = data.astype(int)
target = target.astype(int)


data = torch.from_numpy(data).float()

target = torch.from_numpy(target)# change type to your use case
print(target.size())
#plt.imshow(target)
#plt.show()
batch_size = 1

dataset = TensorDataset(data, target)
train_dataloader = DataLoader(dataset, batch_size=batch_size)

edge_index = []

for node in range(145                                                                  - 1):
    edge_to = [node, node+1]
    edge_back = [node+1, node]
    edge_index.append(edge_to)
    edge_index.append(edge_back)


edge_index = torch.tensor(edge_index, dtype=torch.long)

dataloader_iterator = iter(train_dataloader)

data, target = next(dataloader_iterator)
print(data)
#convert to graph
data = Data(x=data, edge_index=edge_index, y=target)

print(data)