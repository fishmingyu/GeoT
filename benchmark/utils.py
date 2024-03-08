from torch_geometric import datasets
from torch_geometric.utils import from_dgl
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
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
        elif self.name == "amazon_photo":
            dataset = datasets.Amazon(root='./data/', name='Photo')
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
        self.size = graph.num_nodes

        idx = self.edge_index[1]
        sorted_idx = torch.argsort(idx)
        self.idx = idx[sorted_idx]

