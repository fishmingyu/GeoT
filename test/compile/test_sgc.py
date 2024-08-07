import torch
from torch_geometric.nn.conv import SGConv

from utils import Dataset, timeit
import geot.match_replace as replace
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="flickr")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--which", type=str, default="compiled")
    args = parser.parse_args()
    
    # prepare data and models
    d = Dataset(args.dataset, args.device)
    data = d.edge_index
    edge_sparse = d.adj_t
    model = SGConv(d.in_channels, args.hidden_channels).to(args.device)
    
    # get control output
    out_SGC = model(d.x, data)
    
    ## replace pattern
    # from torch.export import export
    # print(f'\nSGC Before:{export(model_SGC, (d_SGC.x, data_SGC)).graph_module.code}')
    exported_SGC = replace.pattern_transform(model, (d.x, data))
    print(f'\nSGC After:{exported_SGC.graph_module.code}')
    
    compile_exported = torch.compile(exported_SGC.module())
    out_compile = compile_exported(d.x, data)
    
    # compare the output with control output
    diff_SGC = torch.abs(out_SGC - out_compile).max()
    print(f'max difference with SGC: {diff_SGC}')

    # benchmark time
    iter = 100
    if args.which == "compiled":
        t = timeit(compile_exported, iter, d.x, data)
    elif args.which == "dense":
        t = timeit(model, iter, d.x, data)
    elif args.which == "sparse":
        t = timeit(model, iter, d.x, edge_sparse)
    # write with 'a' to append to the file
    with open('model_result.csv', 'a') as file:
        file.write(f"SGC,{args.dataset},{args.hidden_channels},{args.which},{t.mean():.6f}\n")
        