from select import select
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


def make_


def train_batch():
    pass