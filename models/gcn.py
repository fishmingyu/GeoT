import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import torch_geometric
import torch_geometric.typing
from basicgnn import BasicGNN
from conv.gcnconv import GCNConv

from torch_geometric.nn.conv import (
    MessagePassing,
)
from utils import Dataset


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
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--is_sparse", type=bool, default=True)
    parser.add_argument("--hidden_channels", type=int, default=64)
    args = parser.parse_args()

    kwargs = {}
    d = Dataset(args.dataset, args.device)
    kwargs["add_self_loops"] = False
    model = GCN(d.in_channels, args.hidden_channels, d.num_classes, **kwargs).to(args.device)
    if args.is_sparse:
        model(d.x, d.adj_t)
    else:
        model(d.x, d.edge_index)