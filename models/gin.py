import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import torch_geometric
import torch_geometric.typing
from basicgnn import BasicGNN
from conv.ginconv import GINConv_GS
from torch_geometric.nn.models import MLP
from torch_geometric.nn.conv import (
    MessagePassing,
    GINConv,
)
from utils import Dataset
from utils import timeit

class GIN(BasicGNN):
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINConv(mlp, **kwargs)


class GIN_GS(BasicGNN):
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINConv_GS(mlp, **kwargs)
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="flickr")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--GS", action="store_true")
    args = parser.parse_args()

    kwargs = {"aggr": "sum"}
    d = Dataset(args.dataset, args.device)
    if args.GS:
        model = GIN_GS(d.in_channels, args.hidden_channels, d.num_classes, **kwargs).to(args.device)
    else:
        model = GIN(d.in_channels, args.hidden_channels, d.num_classes, **kwargs).to(args.device)
    data = d.adj_t if args.sparse else d.edge_index
    
    # benchmark time
    iter = 100
    # write to csv file, test model(GIN or GIN_GS) with dataset
    t = timeit(model, iter, d.x, data)
    # write with 'a' to append to the file
    with open('model_result.csv', 'a') as file:
        file.write(f"GIN,{args.dataset},{args.hidden_channels},{args.sparse},{args.GS},{t.mean():.6f}\n")
