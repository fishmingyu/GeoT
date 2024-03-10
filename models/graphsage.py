import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import torch_geometric
import torch_geometric.typing
from basicgnn import BasicGNN
from conv.sageconv import SAGEConv_GS

from torch_geometric.nn.conv import (
    MessagePassing,
    SAGEConv,
)
from utils import Dataset


class GraphSAGE(BasicGNN):
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return SAGEConv(in_channels, out_channels, **kwargs)

class GraphSAGE_GS(BasicGNN):
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return SAGEConv_GS(in_channels, out_channels, **kwargs)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--GS", action="store_true")
    args = parser.parse_args()

    # set aggr = sum & self_lopp
    kwargs = {"aggr": "sum"}
    d = Dataset(args.dataset, args.device)
    if args.GS:
        model = GraphSAGE_GS(d.in_channels, args.hidden_channels, d.num_classes, **kwargs).to(args.device)
    else:
        model = GraphSAGE(d.in_channels, args.hidden_channels, d.num_classes, **kwargs).to(args.device)
    if args.sparse:
        data = d.adj_t
    else:
        data = d.edge_index
    print(data)
    # benchmark breakdown
    # use torch profiler to profile the time taken for forward pass
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        model(d.x, data)
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    