import torch
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from torch import batch_norm, double, nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.data import InMemoryDataset, download_url
from tqdm import tqdm



raw_data = scipy.io.loadmat('data/indian_pines.mat')
ground_truth = scipy.io.loadmat('data/indian_pines_gt.mat')
data = raw_data['indian_pines'] # use the key for data here
target = ground_truth['indian_pines_gt'] # use the key for target here

data = data.astype(int)
target = target.astype(int)

edge_index = []

for node in range(145 - 1):
    edge_to = [node, node+1]
    edge_back = [node+1, node]
    edge_index.append(edge_to)
    edge_index.append(edge_back)


edge_index = torch.tensor(edge_index, dtype=torch.long)

print(data[5])
data = torch.from_numpy(data[0]).float()

target = torch.from_numpy(target[0]).long()# change type to your use case
print(target.size())




#dataloader_iterator = iter(train_dataloader)

#data, target = next(dataloader_iterator)
print(data.size())
#convert to graph
dataset = Data(x=data, edge_index=edge_index.t().contiguous(), y=target)


m = 0.8

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 220)
        self.conv2 = GCNConv(220, 150)
        self.conv3 = GCNConv(150, 100)
        self.conv4 = GCNConv(100, 60)
        self.conv5 = GCNConv(60, 17)
        self.batch_norm1 = BatchNorm(220, eps=1e-5, momentum=0.9)
        self.batch_norm2 = BatchNorm(150, eps=1e-5, momentum=0.9)
        self.batch_norm3 = BatchNorm(100, eps=1e-5, momentum=0.9)
        self.batch_norm4 = BatchNorm(60, eps=1e-5, momentum=0.9)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.batch_norm2(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.batch_norm3(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.batch_norm4(x)
        x = self.conv5(x, edge_index)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = dataset.to(device)
print(data.x.type())
model = GCN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss = 100
best_loss = 100
model.train()
pbar = tqdm(range(2000))
for epoch in pbar:
    pbar.set_description("Loss: %f" % loss)
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)
    if loss < best_loss:
        torch.save(model.state_dict(), "best_model")
        best_loss = loss
    loss.backward()
    optimizer.step()


model = GCN().to(device)
model.load_state_dict(torch.load("best_model"))
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred == data.y).sum()
print("wrongs: ", len(data.y) - int(correct))
acc = int(correct) / int((len(data.y)))
print(f'Accuracy: {acc:.4f}')

pred = pred.cpu()
data.y = data.y.cpu()
plt.imshow(np.expand_dims(pred, axis=1))


plt.show()

plt.imshow(np.expand_dims(data.y, axis=1))

plt.show()