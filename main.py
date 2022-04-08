from this import d
from scipy import rand
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
import random


raw_data = scipy.io.loadmat('data/indian_pines.mat')
ground_truth = scipy.io.loadmat('data/indian_pines_gt.mat')
dataset = raw_data['indian_pines'] # use the key for data here
target = ground_truth['indian_pines_gt'] # use the key for target here

dataset = dataset.astype(int)
target = target.astype(int)

edge_index = []

for node in range(145 - 1):
    edge_to = [node, node+1]
    edge_back = [node+1, node]
    edge_index.append(edge_to)
    edge_index.append(edge_back)


edge_index = torch.tensor(edge_index, dtype=torch.long)

print(dataset[5])
datalist = []
for i in range(145):
    spectra = torch.from_numpy(dataset[i]).float()
    spectra_target = torch.from_numpy(target[i]).long()# change type to your use case

    data = Data(x=spectra, edge_index=edge_index.t().contiguous(), y=spectra_target)
    datalist.append(data)


#dataloader_iterator = iter(train_dataloader)

#data, target = next(dataloader_iterator)
print(len(datalist))
#convert to graph
#loader = DataLoader(datalist, shuffle=False)


m = 0.8

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(220, 220)
        self.conv2 = GCNConv(220, 150)
        self.conv3 = GCNConv(150, 100)
        self.conv4 = GCNConv(100, 60)
        self.conv5 = GCNConv(60, 30)
        self.conv6 = GCNConv(30, 17)
        self.batch_norm1 = BatchNorm(220, eps=1e-5, momentum=0.9)
        self.batch_norm2 = BatchNorm(150, eps=1e-5, momentum=0.9)
        self.batch_norm3 = BatchNorm(100, eps=1e-5, momentum=0.9)
        self.batch_norm4 = BatchNorm(60, eps=1e-5, momentum=0.9)
        self.batch_norm5 = BatchNorm(30, eps=1e-5, momentum=0.9)
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
        x = F.relu(x)
        x = self.batch_norm5(x)
        x = self.conv6(x, edge_index)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss = 100
best_loss = 100
model.train()
pbar = tqdm(range(10000))
number_of_spectra = 10
n = 0

for epoch in pbar:
    data = datalist[n]
    data.to(device)
    pbar.set_description("Loss: %f" % loss)
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)
    if loss < best_loss:
        torch.save(model.state_dict(), "best_model")
        best_loss = loss
    loss.backward()
    optimizer.step()
    n += 1
    if n == number_of_spectra:
        n = 0
        

model = GCN().to(device)
model.load_state_dict(torch.load("best_model"))
model.eval()
correct = 0
predicted_image = []

for spectra in range(number_of_spectra):
    data = datalist[spectra]
    pred = model(data).argmax(dim=1)
    correct += (pred == data.y).sum()
    pred = pred.tolist()
    predicted_image.append(pred)
print("wrongs: ", len(data.y)*number_of_spectra - int(correct))
acc = int(correct) / int((len(data.y)*number_of_spectra))
print(f'Accuracy: {acc:.4f}')

#predicted_image = predicted_image.cpu()
data.y = data.y.cpu()
plt.imshow(predicted_image)


plt.show()

plt.imshow(np.expand_dims(data.y, axis=1))

plt.show()