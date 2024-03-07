import os
import os.path as osp
import warnings

import torch
import torch.nn.functional as F
import logging

import torch.fx
import torch_geometric
import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.models import GCN
from torch_geometric.profile import benchmark
from torch_geometric.testing import (
    disableExtensions,
    onlyFullTest,
    onlyLinux,
    withCUDA,
    withPackage,
)

from torch._inductor import config
config.cpp.enable_kernel_profile = True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    kwargs = {}
    num_nodes, num_edges = 10_026, 200_231
    x = torch.randn(num_nodes, 64, device=args.device)
    edge_index = torch.randint(num_nodes, (2, num_edges), device=args.device)
    kwargs["add_self_loops"] = False
    model = GCN(64, 64, num_layers=3, **kwargs).to(args.device)

    compiled_model = torch_geometric.compile(model, backend="inductor")
    compiled_model(x, edge_index)