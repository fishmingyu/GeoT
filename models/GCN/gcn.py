import os
import os.path as osp
import warnings

import torch
import torch.nn.functional as F
import logging
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import torch.fx
import torch_geometric
import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from basicgnn import BasicGNN
from torch_geometric.profile import benchmark
from torch_geometric.testing import (
    disableExtensions,
    onlyFullTest,
    onlyLinux,
    withCUDA,
    withPackage,
)

from torch_geometric.nn.conv import (
    MessagePassing,
)

class GCN(BasicGNN):
    supports_edge_weight: Final[bool] = True
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return GCNConv(in_channels, out_channels, **kwargs)

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