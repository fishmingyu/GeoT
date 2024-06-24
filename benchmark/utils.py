from torch_geometric import datasets
from torch_geometric.utils import from_dgl
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric import EdgeIndex
from torch_geometric.utils import add_self_loops


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
        elif self.name == 'flickr':
            dataset = datasets.Yelp(root='./data/Flickr')
            graph = dataset[0]
        elif self.name == 'ogbn-arxiv':
            dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data/')
            graph = dataset[0]
        elif self.name == 'ogbl-collab':
            dataset = PygLinkPropPredDataset(name='ogbl-collab', root='./data/')
            graph = dataset[0]
        elif self.name == 'reddit2':
            dataset = datasets.Reddit2(root='./data/Reddit2')
            graph = dataset[0]
        else:
            raise KeyError('Unknown dataset {}.'.format(self.name))
        print("Dataset: ", self.name)
        self.edge_index = graph.edge_index.to(self.device)
        self.edge_index, _ = add_self_loops(self.edge_index, num_nodes=graph.num_nodes)
        self.num_edges = graph.num_edges
        self.num_nodes = graph.num_nodes

        row = self.edge_index[0]
        col = self.edge_index[1]

        # stack the row and col to create the edge_index
        tmp = EdgeIndex(torch.stack([col, row], dim=0)).sort_by('row')[0]
        self.adj_t = tmp.to_sparse_tensor()
        # materialize
        self.adj_t.storage.csr2csc()
        self.adj_t.storage.colptr()
        assert tmp.is_sorted_by_row == True
        assert torch.all(self.adj_t.storage.row() == torch.sort(self.adj_t.storage.row())[0])
        self.idx = self.adj_t.storage.row()
        self.src_idx = self.adj_t.storage.col()
        self.dst_idx = self.adj_t.storage.row()

