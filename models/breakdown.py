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
from utils import timeit
import csv
import os


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
    parser.add_argument("--hidden_channels", type=int, default=64)
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
    
    # benchmark breakdown with torch profiler
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        model(d.x, data)

    # Analyze the profiling results
    profiler_results = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)

    # Convert the table to CSV format
    rows = profiler_results.split('\n')
    # split by space
    csv_data = [row.split() for row in rows]

    # Extract percentages from CSV data
    percentages = {}
    total_time = None
    # ignore first two rows
    for row in csv_data:
        # skip empty rows
        if len(row) == 0:
            continue
        # if row[0] starts with "aten::" or "torch_index_scatter::" or "torch_sparse::"
        if row[0].startswith("aten::") or row[0].startswith("torch_index_scatter::") or row[0].startswith("torch_sparse::"):
            # row[7] is the percentage of cuda time spent in the function    
            # row[7] is a string end with "%"
            percentages[row[0]] = row[7][:-1]

    if args.sparse:
        filename = f"{args.dataset}_{args.hidden_channels}_sparse_breakdown.csv"
    else:
        filename = f"{args.dataset}_{args.hidden_channels}_breakdown.csv"
    if args.GS and args.sparse:
        filename = f"{args.dataset}_{args.hidden_channels}_sparse_GS_breakdown.csv"

    # create a new directory called breakdown
    # write the file to the directory
    os.makedirs("breakdown", exist_ok=True)
    filename = os.path.join("breakdown", filename)

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Function", "Percentage"])
        for key, value in percentages.items():
            writer.writerow([key, value])
        