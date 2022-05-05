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
    name = name.lower()
    dataset = raw_data[name] # use the key for data here
    gt_key = name + '_gt'
    target = ground_truth[gt_key] # use the key for target here

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
        spectra_target = torch.from_numpy(target[i].flatten()).long()# change type to your use case
        
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
        spectra_target = torch.from_numpy(targets[i].flatten()).long()# change type to your use case

        data = Data(x=spectra, edge_index=edge_indexes.t().contiguous(), y=spectra_target)
        data_list.append(data)
        
    return data_list


def train_loop(data_loader, model, optimizer, device):
    size = len(data_loader.dataset)
    summed_loss = 0
    hidden = torch.zeros(1, 30)
    hidden = hidden.to(device)
    for batch, data in tqdm(enumerate(data_loader)):
        data = data.to(device)
        #pbar.set_description("Loss: %f" % loss)
        
        out  = model(data)
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
        pred = model(data).argmax(dim=1)
        correct += (pred == data.y).sum()
        pred = pred.tolist()
        predicted_image.append(pred)
    print("wrongs: ", len(data.y)*size - int(correct))
    acc = int(correct) / int((len(data.y)*size))
    print(f'Accuracy: {acc:.4f}')

    return acc, predicted_image