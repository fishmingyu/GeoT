import torch
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union
from basicgnn import BasicGNN
from torch_geometric.nn.models import MLP
from torch_geometric.nn.conv import (
    MessagePassing,
    GINConv,
)

from utils import Dataset, timeit
import geot.match_replace as replace

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
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="flickr")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=64)
    args = parser.parse_args()
    
    # prepare data and models
    d = Dataset(args.dataset, args.device)
    data = d.adj_t if args.sparse else d.edge_index
    kwargs = {"aggr": "sum"}
    model = GIN(d.in_channels, args.hidden_channels, args.num_layers, d.num_classes, **kwargs).to(args.device)
    
    # get control output
    out_gin = model(d.x, data)
    
    # replace pattern
    exported = replace.pattern_transform(model, (d.x, data))
    # print(f'\nAfter:{exported.graph_module.code}')
    
    compile_exported = torch.compile(exported.module())
    out_compile = compile_exported(d.x, data)

    diff = torch.abs(out_gin - out_compile).max()
    print(f'max difference with GIN: {diff}')
    
    # benchmark time
    iter = 100
    # write to csv file, test model GIN with dataset
    t_gin = timeit(model, iter, d.x, data)
    # t_compile_gin = timeit(compile_exported, iter, d.x, data)
    
    # write with 'a' to append to the file
    with open('model_result.csv', 'a') as file:
        file.write(f"GIN,{args.dataset},{args.hidden_channels},{args.sparse},{t_gin.mean():.6f}\n")
        # file.write(f"GIN,{args.dataset},{args.hidden_channels},{args.sparse},{t_compile_gin.mean():.6f}\n")
        # file.write(f"GIN,{args.dataset},{args.hidden_channels},{args.sparse},{t_gin.mean():.6f},{t_compile_gin.mean():.6f}\n")
