import torch
from torch_geometric.nn.models import MLP
from torch_geometric.nn.models.basic_gnn import GIN
from torch_geometric.nn.conv import GINConv

from typing import Final
from utils import Dataset, timeit
import geot.match_replace as replace  

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="flickr")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=64)
    args = parser.parse_args()
        
    d = Dataset(args.dataset, args.device)
    data = d.edge_index
    model = GIN(d.in_channels, args.hidden_channels, args.num_layers, d.num_classes, aggr='sum').to(args.device)
    
    # get control output
    out_gin = model(d.x, data)
    
    # replace pattern
    # from torch.export import export
    # print(f'Before:{export(model,(d.x, data)).graph_module.code}')
    exported = replace.pattern_transform(model, (d.x, data))
    print(f'\nAfter:{exported.graph_module.code}')
    
    compile_exported = torch.compile(exported.module())
    out_compile = compile_exported(d.x, data)

    diff = torch.abs(out_gin - out_compile).max()
    print(f'max difference with GIN: {diff}')
    
    # benchmark time
    iter = 100
    t_gin = timeit(model, iter, d.x, data)
    t_compile_gin = timeit(compile_exported, iter, d.x, data)
    # write with 'a' to append to the file
    with open('model_result.csv', 'a') as file:
        file.write(f"GIN,{args.dataset},{args.hidden_channels},{t_gin.mean():.6f},{t_compile_gin.mean():.6f}\n")
