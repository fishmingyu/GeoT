import torch
import geot
from utils import Dataset
import time

from torch_scatter import scatter
from torch_scatter import segment_coo


def pyg_scatter_reduce(index, src):
    return scatter(src, index, dim=0, reduce='sum')


def pyg_segment_coo(index, src):
    return segment_coo(src, index, reduce='sum')


def torch_scatter_reduce(index, src):  # use torch.scatter_add_ as reference
    keys = index[-1] + 1
    device = index.device
    return torch.zeros(keys, src.size(1), device=device).scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)


def torch_index_reduce(index, src):
    keys = index[-1] + 1
    device = index.device
    return torch.zeros(keys, src.size(1), device=device).index_add_(0, index, src)

def index_scatter_reduce(index, src):
    return geot.index_scatter(0, src, index, reduce='sum', sorted=False)


def timeit(func, iter, *args, **kwargs):
    start = time.time()
    for _ in range(iter):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()
    # return each func's ms
    print(f"{func.__name__} time: {(end - start) / iter * 1000:.3f} ms")
    return (end - start) / iter * 1000


def test_index_scatter(file, dataset, feature_size, device):
    g = Dataset(dataset, device)
    idx = g.idx
    src = torch.rand(idx.size(0), feature_size).to(device)
    # warm up
    for i in range(10):
        pyg_scatter_reduce(idx, src)
        pyg_segment_coo(idx, src)
        torch_scatter_reduce(idx, src)
        index_scatter_reduce(idx, src)
    # benchmark time
    iter = 100
    # file is a csv file
    t1 = timeit(pyg_scatter_reduce, iter, idx, src)
    t2 = timeit(pyg_segment_coo, iter, idx, src)
    t3 = timeit(torch_scatter_reduce, iter, idx, src)
    t4 = timeit(index_scatter_reduce, iter, idx, src)
    # :.4f 
    file.write(f"{t1:.4f},{t2:.4f},{t3:.4f},{t4:.4f}")


if __name__ == '__main__':
    datasets = ['cora', 'citeseer', 'pubmed', 'amazon_photo', 'ppi', 'flickr', 'ogbn-arxiv', 'ogbl-collab']
    feature_size = [1, 2, 4, 8, 16, 32, 64, 128]
    device = "cuda"
    # write benchmark result to csv file
    with open("benchop_index_scatter.csv", "w") as file:
        file.write("dataset,feature_size,pyg_scatter_reduce,pyg_segment_coo,torch_scatter_reduce,index_scatter_reduce\n")
        for d in datasets:
            for f in feature_size:
                print(f"Testing {d} with feature size {f}")
                file.write(f"{d},{f},")
                test_index_scatter(file, d, f, device)
                file.write("\n")