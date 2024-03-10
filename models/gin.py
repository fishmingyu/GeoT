import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import torch_geometric
import torch_geometric.typing
from basicgnn import BasicGNN
from conv.ginconv import GINConv_GS

from torch_geometric.nn.conv import (
    MessagePassing,
    GINConv,
)
from utils import Dataset


class GIN(BasicGNN):
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return GINConv(in_channels, out_channels, **kwargs)


class GIN_GS(BasicGNN):
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return GINConv_GS(in_channels, out_channels, **kwargs)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--is_sparse", type=bool, default=True)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--is_GS", type=bool, default=False)
    args = parser.parse_args()

    kwargs = {}
    d = Dataset(args.dataset, args.device)
    if args.is_GS:
        model = GIN_GS(d.in_channels, args.hidden_channels, d.num_classes, **kwargs).to(args.device)
    else:
        model = GIN(d.in_channels, args.hidden_channels, d.num_classes, **kwargs).to(args.device)
    data = d.adj_t if args.is_sparse else d.edge_index
    
    # benchmark breakdown
    # use torch profiler to profile the time taken for forward pass
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        model(d.x, data)
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    
