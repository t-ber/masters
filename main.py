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
from torch_geometric.nn import GCNConv, BatchNorm, ChebConv, TransformerConv, AGNNConv, TAGConv, GINConv ,GINEConv, ARMAConv, SGConv, APPNP, MFConv
from torch_geometric.data import InMemoryDataset, download_url
from tqdm import tqdm
import random

from torch_geometric_temporal.nn.recurrent import GConvGRU




def make_edge_indexes(all_edges, number_of_nodes):
    edge_index = []

    if all_edges:
        for node in range(number_of_nodes):
            for node_to_connect in range(node+1,node+2):
                if node_to_connect == number_of_nodes:
                    break
                edge_to = [node, node_to_connect]
                edge_from = [node_to_connect, node]
                edge_index.append(edge_to)
                edge_index.append(edge_from)

    """elif not all_edges:
        for node in range(number_of_nodes - 1):
            edge_to = [node, node+1]
            edge_back = [node+1, node]
            edge_index.append(edge_to)
            edge_index.append(edge_back)"""

    return torch.tensor(edge_index, dtype=torch.long)


raw_data = scipy.io.loadmat('data/indian_pines.mat')
ground_truth = scipy.io.loadmat('data/indian_pines_gt.mat')
dataset = raw_data['indian_pines'] # use the key for data here
target = ground_truth['indian_pines_gt'] # use the key for target here

dataset = dataset.astype(int)
target = target.astype(int)


#plt.imshow(target)
#plt.show()


edge_index = make_edge_indexes(True, 145)

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

"""class GCN(torch.nn.Module):
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
        return F.log_softmax(x, dim=1)"""
    
def make_conv_layer(inputs, outputs):
    return TransformerConv(inputs, outputs)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """super(GCN, self).__init__()"""
        self.hidden = GConvGRU(220, 220, 1)
        

        self.linear1 = nn.Linear(60, 30)
        self.linear2 = nn.Linear(30,17)

        self.conv1 = make_conv_layer(220, 220)
        self.conv2 = make_conv_layer(220, 150)
        
        self.conv_skip_1 = make_conv_layer(220, 150)
        self.conv_skip_2 = make_conv_layer(150, 60)
        

        self.conv3 = make_conv_layer(150, 100)
        self.conv4 = make_conv_layer(100, 60)
        self.conv5 = make_conv_layer(60, 30)
        self.conv6 = make_conv_layer(30, 10)
        self.batch_norm1 = BatchNorm(220, eps=1e-5, momentum=0.9)
        self.batch_norm2 = BatchNorm(150, eps=1e-5, momentum=0.9)
        self.batch_norm3 = BatchNorm(100, eps=1e-5, momentum=0.9)
        self.batch_norm4 = BatchNorm(60, eps=1e-5, momentum=0.9)
        self.batch_norm5 = BatchNorm(30, eps=1e-5, momentum=0.9)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        identity = x

        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)

        identity = self.conv_skip_1(identity, edge_index)
       
        x += identity
        
        x = F.relu(x)
        

        
        x = self.conv3(x, edge_index)
        x = self.batch_norm3(x)
        x = F.relu(x)
        
        x = self.conv4(x, edge_index)
        x = self.batch_norm4(x)

        #identity = self.conv_skip_2(identity, edge_index)

        #x += identity
        x = F.relu(x)
        
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

training_spectra = [8, 9, 10, 22, 23, 24, 32, 33, 34, 44, 45, 46, 52, 53, 54, 
                    65, 66, 67, 69, 70, 71, 79, 80, 81, 
                    99, 100, 101, 118, 119, 120, 132, 133, 138, 139, 140]

model.train()
pbar = tqdm(range(200))
number_of_spectra = len(training_spectra)
n = 0

spectra_number_list = training_spectra
number_of_spectra = 30
spectra_number_list = list(range(number_of_spectra))

for epoch in pbar:
    #random.shuffle(spectra_number_list)
    #print(spectra_number_list)
    for spectra_number in spectra_number_list:
        data = datalist[spectra_number]
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
        """n += 1
        if n == number_of_spectra:
            n = 0"""
        

model = GCN().to(device)
model.load_state_dict(torch.load("best_model"))
model.eval()
correct = 0
predicted_image = []

for spectra in range(number_of_spectra):
    data = datalist[spectra]
    data.to(device)
    pred = model(data).argmax(dim=1)
    correct += (pred == data.y).sum()
    pred = pred.tolist()
    predicted_image.append(pred)
print("wrongs: ", len(data.y)*number_of_spectra - int(correct))
acc = int(correct) / int((len(data.y)*number_of_spectra))
print(f'Accuracy: {acc:.4f}')

#predicted_image = predicted_image.cpu()
data.y = data.y.cpu()


f, axarr = plt.subplots(1,2)
axarr[0].imshow(predicted_image)
axarr[1].imshow(target)

plt.show()





#plt.imshow(np.expand_dims(data.y, axis=1))

#plt.show()