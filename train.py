from select import select
from this import d

from matplotlib import transforms
from scipy import rand
import torch
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from torch import batch_norm, double, nn
from torch.utils.data import TensorDataset
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, ChebConv, TransformerConv, AGNNConv, TAGConv, GINConv ,GINEConv, ARMAConv, SGConv, APPNP, MFConv
from torch_geometric.data import InMemoryDataset, download_url
from tqdm import tqdm
import random

from torch_geometric_temporal.nn.recurrent import GConvGRU


def load_dataset(name):
    datapath = 'data/' + name + '.mat'
    ground_truth_path = 'data/' + name + '_gt.mat'
    raw_data = scipy.io.loadmat(datapath)
    ground_truth = scipy.io.loadmat(ground_truth_path)
    #name = name.lower()
    
    dataset = raw_data["salinasA"] # use the key for data here
    gt_key = name + '_gt'
    target = ground_truth["salinasA_gt"] # use the key for target here

    dataset = dataset.astype(int)
    target = target.astype(int)

    return dataset, target

def split_training_data(data, targets, training_size):
    number_of_training_samples = np.round(training_size*len(data)).astype(int)
    train_indexes = np.round(np.linspace(0, len(data) - 1, number_of_training_samples)).astype(int)
    
    
    train_data = []
    train_targets = []

    for index in train_indexes:
        
        train_data.append(data[index])
        train_targets.append(targets[index])


    return train_data, train_targets


def make_edge_indexes(number_of_nodes):
    edge_index = []

    for node in range(number_of_nodes - 1):
        edge_to = [node, node+1]
        edge_back = [node+1, node]
        edge_index.append(edge_to)
        edge_index.append(edge_back)

    return torch.tensor(edge_index, dtype=torch.long)


def make_training_data(raw_data, target, nodes_in_data, node_overlap):

    edge_indexes = make_edge_indexes(nodes_in_data)
      
    datalist = []
   
    for i in range(len(raw_data)):
        
        spectra = torch.from_numpy(raw_data[i]).float()
        spectra_target = torch.from_numpy(target[i]).long()# change type to your use case
        
        n = 0
        break_loop  = False
        while(not break_loop):
            if n == len(spectra):
                break
            while n + nodes_in_data >= len(spectra):
                n -= 1
                break_loop = True
            
            mini_spectra = spectra[n:n+nodes_in_data]
            mini_spectra_target = spectra_target [n:n+nodes_in_data]
            data = Data(x=mini_spectra, edge_index=edge_indexes.t().contiguous(), y=mini_spectra_target)
            datalist.append(data)
        
            n += nodes_in_data
            n -= node_overlap

    return datalist

def make_test_data(raw_data, targets):
    nodes_in_data = len(raw_data[0])
    edge_indexes = make_edge_indexes(nodes_in_data)

    data_list = []

    for i in range(len(raw_data)):
        
        spectra = torch.from_numpy(raw_data[i]).float()
        spectra_target = torch.from_numpy(targets[i]).long()# change type to your use case

        data = Data(x=spectra, edge_index=edge_indexes.t().contiguous(), y=spectra_target)
        data_list.append(data)
        
    return data_list


def train_loop(data_loader, model, optimizer, device):
    size = len(data_loader.dataset)
    summed_loss = 0
    
    
    for batch, data in tqdm(enumerate(data_loader)):
        data = data.to(device)
        
        #pbar.set_description("Loss: %f" % loss)
        out = model(data)
        #out = out.view(4, 100)
        #print(out.size())
        #print(data.y.size())
        #print(data.y.size())
        
        #print("out ", out.size())
        #print("y ",data.y.size())
        loss = F.nll_loss(out, data.y)
        summed_loss += loss
        """if loss < best_loss:
            torch.save(model.state_dict(), "best_model")
            best_loss = loss"""
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_loss = summed_loss/size
    return avg_loss


def test_loop(data_loader, model, device):
    size = len(data_loader.dataset)
    model.load_state_dict(torch.load("best_model"))
    model.eval()
    correct = 0
    predicted_image = []

    for batch, data in tqdm(enumerate(data_loader)):
        data.to(device)
        pred = model(data)
        #pred = pred.view(145, 100)
        #print(pred.size())
        #print(data.y.size())
        #dim_1 = pred.size()[0] * pred.size()[1]
        #pred = pred.view(dim_1,17)
        pred = pred.argmax(dim=1)
        
        
        correct += (pred == data.y).sum()
        
        pred = pred.tolist()
        predicted_image.append(pred)
        
    print("wrongs: ", len(data.y)*size - int(correct))
    acc = int(correct) / int((len(data.y)*size))
    print(f'Accuracy: {acc:.4f}')

    return acc, predicted_image

    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            """super(GCN, self).__init__()"""


            self.hidden = GConvGRU(220, 220, 1)
            

            self.linear1 = nn.Linear(220, 100)
            self.linear2 = nn.Linear(100, 50)
            self.linear3 = nn.Linear(50, 17)
            self.linear4 = nn.Linear(50, 17)

            self.conv1 = make_conv_layer(220, 220)
            self.conv2 = make_conv_layer(220, 220)

            self.conv3 = make_conv_layer(60, 30)
            
            self.batch_norm1 = BatchNorm(220, eps=1e-5, momentum=0.9)
            self.batch_norm2 = BatchNorm(220, eps=1e-5, momentum=0.9)
            self.batch_norm3 = BatchNorm(100, eps=1e-5, momentum=0.9)
            self.batch_norm4 = BatchNorm(50, eps=1e-5, momentum=0.9)
            self.batch_norm5 = BatchNorm(50, eps=1e-5, momentum=0.9)
        
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            
            x = self.conv1(x, edge_index)
            x = self.batch_norm1(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            
            x = self.conv2(x, edge_index)
            x = self.batch_norm2(x)

            

            x = F.relu(x)
            
            #x = self.conv3(x, edge_index)
            #x = self.batch_norm3(x)
            
            #x = F.relu(x)

            x = self.linear1(x)
            x = self.batch_norm3(x)
            x = F.relu(x)

            x = self.linear2(x)
            x = self.batch_norm4(x)
            x = F.relu(x)

            x = self.linear3(x)
            


            
            return F.log_softmax(x, dim=1)