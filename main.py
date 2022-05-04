from operator import mod
from select import select
from statistics import mode
from this import d
import time
from scipy import rand
import torch
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from torch import batch_norm, double, nn
from torch.utils.data import TensorDataset
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, BatchNorm, ChebConv, TransformerConv, AGNNConv, TAGConv, GINConv ,GINEConv, ARMAConv, SGConv, APPNP, MFConv
from torch_geometric.data import InMemoryDataset, download_url
from tqdm import tqdm
import random

from torch_geometric_temporal.nn.recurrent import GConvGRU
from train import *

print("Making dataset")

data, targets = load_dataset('Indian_pines')

train_data, train_targets = split_training_data(data, targets, 0.3)

train_data = make_training_data(train_data, train_targets, 3, 0)

test_data = make_test_data(data, targets)

train_dataloader = DataLoader(train_data, batch_size=24, shuffle=True)

test_dataloader = DataLoader(test_data)
#plt.imshow(target)
#plt.show()

print("Initiating training")
#dataloader_iterator = iter(train_dataloader)

#data, target = next(dataloader_iterator)

#convert to graph
#loader = DataLoader(datalist, shuffle=False)


m = 0.8

    
def make_conv_layer(inputs, outputs):
    return TransformerConv(inputs, outputs)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """super(GCN, self).__init__()"""


        self.hidden = GConvGRU(220, 220, 1)
        

        self.linear1 = nn.Linear(60, 60)
        self.linear2 = nn.Linear(60, 30)
        self.linear3 = nn.Linear(30, 4)

        self.conv1 = make_conv_layer(220, 150)
        self.conv2 = make_conv_layer(150, 60)

        self.conv3 = make_conv_layer(60, 30)
        
        self.batch_norm1 = BatchNorm(150, eps=1e-5, momentum=0.9)
        self.batch_norm2 = BatchNorm(60, eps=1e-5, momentum=0.9)
        self.batch_norm3 = BatchNorm(60, eps=1e-5, momentum=0.9)
       
    def forward(self, data, hidden):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)

        combined = torch.cat((x, hidden)
        hidden = x

        x = F.relu(combined)
        
        #x = self.conv3(x, edge_index)
        #x = self.batch_norm3(x)
        
        #x = F.relu(x)

        x = self.linear1(x)
        x = self.batch_norm3(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)
        
        
        return F.log_softmax(x, dim=1), hidden



training_spectra = [8, 9, 10, 22, 23, 24, 32, 33, 34, 44, 45, 46, 52, 53, 54, 
                    65, 66, 67, 69, 70, 71, 79, 80, 81, 
                    99, 100, 101, 118, 119, 120, 132, 133, 138, 139, 140]

################################
#Training section
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss = 100
best_loss = 100
times_no_new_best_loss = 0
model.train()

number_of_epochs = 20
for i in range(number_of_epochs):
    avg_loss = train_loop(train_dataloader, model, optimizer, device)
    avg_loss = avg_loss.data.cpu().numpy()
    print(avg_loss)
    if avg_loss < best_loss:
        times_no_new_best_loss = 0
        best_loss = avg_loss
        print("Saving model")
        torch.save(model.state_dict(), "best_model")
    else:
        times_no_new_best_loss += 1
    if times_no_new_best_loss == 30:
        break
################################


model = GCN().to(device)

accuracy, predicted_image = test_loop(test_dataloader, model, device)


#predicted_image = predicted_image.cpu()



f, axarr = plt.subplots(1,2)
axarr[0].imshow(predicted_image)
axarr[1].imshow(targets)

plt.show()





#plt.imshow(np.expand_dims(data.y, axis=1))

#plt.show()