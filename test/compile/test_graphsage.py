import torch
from torch_geometric.nn import SAGEConv

from utils import Dataset, timeit
import geot.match_replace as replace

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="flickr")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=64)
    args = parser.parse_args()
    
    # prepare data and model
    d = Dataset(args.dataset, args.device)
    data = d.edge_index
    model = SAGEConv(d.in_channels, args.hidden_channels).to(args.device)
    
    # get control output
    out_sage = model(d.x, data)
    
    # replace pattern
    # from torch.export import export
    # print(f'\nBefore:{export(model, (d.x, data)).graph_module.code}')
    exported = replace.pattern_transform(model, (d.x, data))
    print(f'\nAfter:{exported.graph_module.code}')
    
    compile_exported = torch.compile(exported.module())
    out_compile = compile_exported(d.x, data)
    
    diff = torch.abs(out_sage - out_compile).max()
    print(f'max difference with SAGE: {diff}')
    
    # benchmark time
    iter = 100
    t_graphsage = timeit(model, iter, d.x, data)
    t_compile_graphsage = timeit(compile_exported, iter, d.x, data)
    # write with 'a' to append to the file
    with open('model_result.csv', 'a') as file:
        file.write(f"SAGE,{args.dataset},{args.hidden_channels},{t_graphsage.mean():.6f},{t_compile_graphsage.mean():.6f}\n")
        