from torch_geometric import datasets
from torch_geometric.utils import from_dgl
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
import os
import torch_geometric.transforms as T
from torch_geometric import EdgeIndex
from torch_geometric.typing import SparseTensor
import torch_geometric
import time

class Dataset:
    def __init__(self, name: str, device):
        self.name = name
        self.device = device
        self.get_dataset()

    def get_dataset(self):
        if self.name == 'ogbn-arxiv':
            dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data/')
            graph = dataset[0]
        elif self.name == 'flickr':
            dataset = datasets.Flickr(root='./data/Flickr')
            graph = dataset[0]
        elif self.name == 'reddit2':
            dataset = datasets.Reddit2(root='./data/Reddit2')
            graph = dataset[0]
        else:
            raise KeyError('Unknown dataset {}.'.format(self.name))
        edge_index = graph.edge_index.to(self.device)
        # add self loop to edge_index
        edge_index, _ = torch_geometric.utils.add_self_loops(edge_index)
        self.edge_index = edge_index[:, torch.argsort(edge_index[1])]

        row = self.edge_index[0]
        col = self.edge_index[1]
        # create new transpose edge_index
        self.edge_index_t = torch.stack([col, row], dim=0)
        self.adj_t = SparseTensor.from_edge_index(self.edge_index_t, is_sorted=True)
        # materialize the colptr and csr2csc
        self.adj_t.storage.colptr()
        self.adj_t.storage.csr2csc()
        # check adj_t is sorted by row
        assert torch.all(self.adj_t.storage.row() == torch.sort(self.adj_t.storage.row())[0])
        if graph.x is not None:
            self.x = graph.x.to(self.device)
        else:
            self.x = torch.zeros(graph.num_nodes, 32).to(self.device)
        # print size of self.x
        self.num_edges = graph.num_edges
        self.in_channels = graph.num_features if graph.x is not None else 32
        self.num_classes = int(graph.y.max() + 1)
        

def timeit(model, iter, x, data):
    # benchmark time
    t = torch.zeros(iter)
    for i in range(iter):
        torch.cuda.synchronize()
        t0 = time.time()
        model(x, data)
        torch.cuda.synchronize()
        t1 = time.time()
        t[i] = t1 - t0
    print(f"Average time: {t.mean():.6f} s")
    return t

