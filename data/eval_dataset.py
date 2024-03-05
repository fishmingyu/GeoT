from torch_geometric import datasets
from torch_geometric.utils import from_dgl
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
import numpy as np
import os
import dgl


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
        elif self.name == "dblp":
            dataset = datasets.CitationFull(root='./data/', name='DBLP')
            graph = dataset[0]
        elif self.name == 'ppi':
            dataset = datasets.PPI(root='./data/PPI')
            graph = dataset[0]
        elif self.name == 'yelp':  # due to PyG's broken link, we use dgl's dataset
            dataset = dgl.data.YelpDataset()
            dgl_g = dataset[0]
            graph = from_dgl(dgl_g)
        elif self.name == 'ogbn-arxiv':
            dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data/')
            graph = dataset[0]
        elif self.name == 'ogbl-collab':
            dataset = PygLinkPropPredDataset(name='ogbl-collab', root='./data/')
            graph = dataset[0]
        else:
            raise KeyError('Unknown dataset {}.'.format(self.name))
        print("Dataset: ", self.name)
        self.edge_index = graph.edge_index.to(self.device)
        self.num_edges = graph.num_edges

        idx = self.edge_index[1]
        sorted_idx = torch.argsort(idx)
        self.idx = idx[sorted_idx]

    def store_idx(self):
        # check if the directory exists
        if not os.path.exists('./eval_data'):
            os.makedirs('./eval_data')
    
        # store the idx via numpy
        np.save(f'./eval_data/{self.name}_idx.npy', self.idx.cpu().numpy())
        print(f"Finish storing {self.name} idx")

if __name__ == '__main__':
    ml_datasets = ['pubmed', 'citeseer', 'cora', 'dblp', 'ppi', 'yelp', 'ogbn-arxiv', 'ogbl-collab']

    device = "cpu"
    
    for d in ml_datasets:
        data = Dataset(d, device)
        data.store_idx()
