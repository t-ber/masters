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
from torch.nn import BatchNorm1d
import torch.nn.functional as F
from torch_geometric.loader import DataLoader, DenseDataLoader
from torch_geometric.nn import GCNConv, ChebConv, TransformerConv, AGNNConv, TAGConv, GINConv ,GINEConv, ARMAConv, SGConv, APPNP, MFConv
from torch_geometric.data import InMemoryDataset, download_url
from tqdm import tqdm
import random
from sklearn.decomposition import PCA

from torch_geometric_temporal.nn.recurrent import GConvGRU
from train import *


def applyPCA(X, numComponents=100):
     newX = np.reshape(X, (-1, X.shape[2]))
     pca = PCA(n_components=numComponents, whiten=True)
     newX = pca.fit_transform(newX)
     newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
     return newX, pca

print("Making dataset")

data, targets = load_dataset('SalinasA')

data, pca =applyPCA(data)

train_data, train_targets = split_training_data(data, targets, 0.3)

train_data = make_training_data(train_data, train_targets, 2, 0)

test_data = make_test_data(data, targets)

train_dataloader = DataLoader(train_data, batch_size=15, shuffle=True)

test_dataloader = DataLoader(test_data)
#plt.imshow(targets)
#plt.show()

print("Initiating training")
#dataloader_iterator = iter(train_dataloader)

#data, target = next(dataloader_iterator)

#convert to graph
#loader = DataLoader(datalist, shuffle=False)


m = 0.8

    
def make_conv_layer(inputs, outputs):
    return GCNConv(inputs, outputs)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """super(GCN, self).__init__()"""

        self.inpu_size = 100
        self.hidden_size = 50
        self.num_layers = 1
        

        self.rnn = nn.RNN(self.inpu_size, self.hidden_size, self.num_layers, batch_first=True)
        

        self.linear1 = nn.Linear(50, 50)
        self.linear2 = nn.Linear(50, 30)
        self.linear3 = nn.Linear(30, 30)
        self.linear4 = nn.Linear(25, 17)

        self.conv1 = make_conv_layer(100, 100)
        self.conv2 = make_conv_layer(100, 100)

        self.conv3 = make_conv_layer(60, 30)
        
        self.batch_norm1 = BatchNorm1d(100, eps=1e-5, momentum=0.9)
        self.batch_norm2 = BatchNorm1d(100, eps=1e-5, momentum=0.9)
        self.batch_norm3 = BatchNorm1d(50, eps=1e-5, momentum=0.9)
        self.batch_norm4 = BatchNorm1d(30, eps=1e-5, momentum=0.9)
        self.batch_norm5 = BatchNorm1d(25, eps=1e-5, momentum=0.9)
       
    def forward(self, data, test=False):
        x, edge_index = data.x, data.edge_index
        
       # edge_index = edge_index[0]
        #print(edge_index.size())
        #edge_index = edge_index[0]
        batch_size = 24
        if test:
            batch_size = 1
        #x = torch.permute(x, (0, 2, 1))
        #print(x.size())
        #print(edge_index)
        #x = x.view(seq_len, batch_size, 220)
        seq_len = x.size()[0]//batch_size
       # print(edge_index.size())
        
        hidden = self.init_hidden(batch_size)

        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        #print(x.size())
        #print(hidden.size())
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x, hidden = self.rnn(x.unsqueeze(0), hidden)
        x = torch.squeeze(x)
        
        #x = self.conv2(x, edge_index)
        #x = self.batch_norm2(x)

        

       # x = F.relu(x)
        
        x = self.linear1(x)
        x = self.batch_norm3(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
       # print(x.size())
        #x = x.view( batch_size,seq_len,100)
        
       

        x = self.linear3(x)
        
        #x = self.batch_norm4(x)
       # x = F.relu(x)
       

        
        

       # x = self.linear3(x)
        


        
        return F.log_softmax(x, dim=1)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return hidden.to(device)

training_spectra = [8, 9, 10, 22, 23, 24, 32, 33, 34, 44, 45, 46, 52, 53, 54, 
                    65, 66, 67, 69, 70, 71, 79, 80, 81, 
                    99, 100, 101, 118, 119, 120, 132, 133, 138, 139, 140]

################################
#Training section
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
loss = 100
best_loss = 100
times_no_new_best_loss = 0
model.train()

number_of_epochs = 50
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