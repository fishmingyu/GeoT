import torch
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset

class OgbDataset:
    def __init__(self, name: str, device):
        self.name = name
        self.device = device
        self.get_dataset()

    def get_dataset(self):
        if self.name == 'products':
            dataset = PygNodePropPredDataset(name='ogbn-products', root = 'dataset/')
            graph = dataset[0]
        elif self.name == 'proteins':
            dataset = PygNodePropPredDataset(name='ogbn-proteins', root = 'dataset/')
            graph = dataset[0]
        elif self.name == 'arxiv':
            dataset = PygNodePropPredDataset(name='ogbn-arxiv', root = 'dataset/')
            graph = dataset[0]
        elif self.name == "mag":
            dataset = PygNodePropPredDataset(name='ogbn-mag', root = 'dataset/')
            graph = dataset[0]
        elif self.name == "ppa":
            dataset = PygLinkPropPredDataset(name='ogbl-ppa', root = 'dataset/')
            graph = dataset[0]
        elif self.name == "collab":
            dataset = PygLinkPropPredDataset(name='ogbl-collab', root = 'dataset/')
            graph = dataset[0]
        elif self.name == "ddi":
            dataset = PygLinkPropPredDataset(name='ogbl-ddi', root = 'dataset/')
            graph = dataset[0]
        elif self.name == "citation2":
            dataset = PygLinkPropPredDataset(name='ogbl-citation2', root = 'dataset/')
            graph = dataset[0]
        elif self.name == "wikikg2":
            dataset = PygLinkPropPredDataset(name='ogbl-wikikg2', root = 'dataset/')
            graph = dataset[0]
        elif self.name == "biokg":
            dataset = PygLinkPropPredDataset(name='ogbl-biokg', root = 'dataset/')
            graph = dataset[0]
        elif self.name == "vessel":
            dataset = PygLinkPropPredDataset(name='ogbl-vessel', root = 'dataset/')
            graph = dataset[0]
        else:
            raise KeyError('Unknown dataset {}.'.format(self.name))
        self.edge_index = graph.edge_index.to(self.device)
        self.num_edges = graph.num_edges

        idx = self.edge_index[1]
        sorted_idx = torch.argsort(idx)
        self.idx = idx[sorted_idx]
