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
import torch_geometric

class Dataset:
    def __init__(self, name: str, device):
        self.name = name
        self.device = device
        self.get_dataset()

    def get_dataset(self):
        if self.name == 'yelp':
            dataset = datasets.Yelp(root='./data/Yelp')
            graph = dataset[0]
        elif self.name == 'ogbn-arxiv':
            dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data/')
            graph = dataset[0]
        elif self.name == 'ogbn-products':
            dataset = PygNodePropPredDataset(name='ogbn-products', root='./data/')
            graph = dataset[0]
        elif self.name == 'ogbn-mag':
            dataset = PygNodePropPredDataset(name='ogbn-mag', root='./data/')
            graph = dataset[0]
        elif self.name == 'flickr':
            dataset = datasets.Flickr(root='./data/Flickr')
            graph = dataset[0]
        else:
            raise KeyError('Unknown dataset {}.'.format(self.name))
        self.edge_index = graph.edge_index.to(self.device)
        # add self loop to edge_index
        self.edge_index, _ = torch_geometric.utils.add_self_loops(self.edge_index)
        row = self.edge_index[0]
        col = self.edge_index[1]
        # stack the row and col to create the edge_index
        self.adj_t = EdgeIndex(torch.stack([col, row], dim=0)).to_sparse_tensor()
        if graph.x is not None:
            self.x = graph.x.to(self.device)
        else:
            self.x = torch.zeros(graph.num_nodes, 32).to(self.device)
        # print size of self.x
        self.num_edges = graph.num_edges
        self.in_channels = graph.num_features if graph.x is not None else 32
        self.num_classes = int(graph.y.max() + 1)
        