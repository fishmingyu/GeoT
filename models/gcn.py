import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import torch_geometric
import torch_geometric.typing
from basicgnn import BasicGNN
from conv.gcnconv import GCNConv_GS

from torch_geometric.nn.conv import (
    MessagePassing,
    GCNConv,
)
from utils import Dataset
import time

class GCN(BasicGNN):
    supports_edge_weight: Final[bool] = True
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return GCNConv(in_channels, out_channels, **kwargs)
    
class GCN_GS(BasicGNN):
    supports_edge_weight: Final[bool] = True
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return GCNConv_GS(in_channels, out_channels, **kwargs)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="pubmed")
    parser.add_argument("--is_sparse", type=bool, default=True)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--is_GS", type=bool, default=False)
    args = parser.parse_args()

    kwargs = {}
    d = Dataset(args.dataset, args.device)
    if args.is_GS:
        model = GCN_GS(d.in_channels, args.hidden_channels, d.num_classes, **kwargs).to(args.device)
    else:
        model = GCN(d.in_channels, args.hidden_channels, d.num_classes, **kwargs).to(args.device)
    data = d.adj_t if args.is_sparse else d.edge_index
    
    # benchmark breakdown
    # use torch profiler to profile the time taken for forward pass
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        model(d.x, data)
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))

