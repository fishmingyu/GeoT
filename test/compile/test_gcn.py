import pytest
import torch
from torch import Tensor

from torch_geometric import EdgeIndex
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.profile import benchmark
from torch_geometric.testing import (
    onlyFullTest,
    onlyLinux,
    withPackage,
)
from torch_geometric.utils import scatter

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_nodes, num_edges = 10_000, 200_000
    x = torch.randn(num_nodes, 64, device=args.device)
    edge_index = torch.randint(num_nodes, (2, num_edges), device=args.device)
    
    # args: no self loop
    
    conv = GCNConv(64, 64, add_self_loops=False).to(args.device)
    compiled_conv = torch.compile(conv)
    
    exp_graph = compiled_conv.graph_module
    compiled_conv(x, edge_index)
