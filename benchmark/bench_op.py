import torch
import torch_index_scatter
from utils import OgbDataset
import time

from torch_scatter import scatter
from torch_scatter import segment_coo


def pyg_scatter(index, src):
    return scatter(src, index, dim=0, reduce='sum')


def pyg_segment_coo(index, src):
    return segment_coo(src, index, reduce='sum')


def torch_scatter(index, src):  # use torch.scatter_add_ as reference
    keys = index[-1] + 1
    device = index.device
    return torch.zeros(keys, src.size(1), device=device).scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)


def index_scatter(index, src):
    return torch_index_scatter.index_scatter(0, src, index, reduce='sum', sorted=True)


def timeit(func, iter, *args, **kwargs):
    start = time.time()
    for _ in range(iter):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()
    # return each func's ms
    print(f"{func.__name__} time: {(end - start) / iter * 1000:.3f} ms")


def test_index_scatter(dataset, feature_size, device):
    g = OgbDataset(dataset, device)
    idx = g.idx
    print(idx.size(0))
    src = torch.rand(idx.size(0), feature_size).to(device)
    # benchmark time
    iter = 100
    timeit(pyg_scatter, iter, idx, src)
    timeit(pyg_segment_coo, iter, idx, src)
    timeit(torch_scatter, iter, idx, src)
    timeit(index_scatter, iter, idx, src)


if __name__ == '__main__':
    dataset = 'products'
    feature_size = 128
    device = "cuda"
    test_index_scatter(dataset, feature_size, device)
