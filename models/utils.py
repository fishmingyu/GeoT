from torch_geometric import datasets
from torch_geometric.utils import from_dgl
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
import numpy as np
import os
import torch_geometric.transforms as T
from torch_geometric import EdgeIndex


class Dataset:
    def __init__(self, name: str, device):
        self.name = name
        self.device = device
        self.get_dataset()

    def get_dataset(self):
        if self.name == 'pubmed':
            dataset = datasets.CitationFull(root='./data/', name='PubMed')
            graph = dataset[0]
        elif self.name == 'citeseer':
            dataset = datasets.CitationFull(root='./data/', name='CiteSeer')
            graph = dataset[0]
        elif self.name == 'cora':
            dataset = datasets.CitationFull(root='./data/', name='Cora')
            graph = dataset[0]
        elif self.name == "amazon_photo":
            dataset = datasets.Amazon(root='./data/', name='Photo')
            graph = dataset[0]
        elif self.name == 'ppi':
            dataset = datasets.PPI(root='./data/PPI')
            graph = dataset[0]
        elif self.name == 'yelp':
            dataset = datasets.Yelp(root='./data/Yelp')
            graph = dataset[0]
        elif self.name == 'ogbn-arxiv':
            dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data/')
            graph = dataset[0]
        elif self.name == 'ogbl-collab':
            dataset = PygLinkPropPredDataset(name='ogbl-collab', root='./data/')
            graph = dataset[0]
        else:
            raise KeyError('Unknown dataset {}.'.format(self.name))
        self.edge_index = graph.edge_index.to(self.device)
        row = self.edge_index[0]
        col = self.edge_index[1]
        # stack the row and col to create the edge_index
        self.adj_t = EdgeIndex(torch.stack([col, row], dim=0)).to_sparse_tensor()
        self.x = graph.x.to(self.device)
        self.num_edges = graph.num_edges
        self.in_channels = graph.num_features
        self.num_classes = int(graph.y.max() + 1)
