from gcn import GCN, GCN_GS
from gin import GIN, GIN_GS
from graphsage import GraphSAGE, GraphSAGE_GS
import GPUtil
from utils import Dataset

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--GS", action="store_true")
    args = parser.parse_args()

    # set aggr = sum & self_lopp
    kwargs = {"aggr": "sum", "add_self_loops": False}
    d = Dataset(args.dataset, args.device)
    if args.model == "GCN":
        if args.GS:
            model = GCN_GS(d.in_channels, args.hidden_channels, d.num_classes, **kwargs).to(args.device)
        else:
            model = GCN(d.in_channels, args.hidden_channels, d.num_classes, **kwargs).to(args.device)
    elif args.model == "GIN":
        if args.GS:
            model = GIN_GS(d.in_channels, args.hidden_channels, d.num_classes, **kwargs).to(args.device)
        else:
            model = GIN(d.in_channels, args.hidden_channels, d.num_classes, **kwargs).to(args.device)
    elif args.model == "GraphSAGE":
        if args.GS:
            model = GraphSAGE_GS(d.in_channels, args.hidden_channels, d.num_classes, **kwargs).to(args.device)
        else:
            model = GraphSAGE(d.in_channels, args.hidden_channels, d.num_classes, **kwargs).to(args.device)
    data = d.adj_t if args.sparse else d.edge_index

    # benchmark memory
    model.eval()
    model(d.x, data)
    # get memory usage memoryUsed
    memoryUsed = GPUtil.getGPUs()[0].memoryUsed
    
    # write with 'a' to append to the file
    with open('memory_result.csv', 'a') as file:
        file.write(f"{args.model},{args.dataset},{args.hidden_channels},{args.sparse},{args.GS},{memoryUsed:.6f}\n")