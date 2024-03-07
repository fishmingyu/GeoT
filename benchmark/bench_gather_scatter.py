import torch
import torch_index_scatter
from utils import Dataset
import time
import torch_sparse

def gather_scatter(src_index, dst_index, src, reduce):
    return torch_index_scatter.gather_scatter(src_index, dst_index, src, reduce)

def pytorch_spmm(A, B):
    return torch.sparse.mm(A, B)

def pyg_spmm(A, B, reduce):
    return torch_sparse.matmul(A, B, reduce)

def timeit(func, iter, *args, **kwargs):
    start = time.time()
    for _ in range(iter):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()
    # return each func's ms
    print(f"{func.__name__} time: {(end - start) / iter * 1000:.3f} ms")


def test_gather_scatter(dataset, feature_size, device):
    g = Dataset(dataset, device)
    edge_index = g.edge_index
    sorted_index = torch.argsort(edge_index[1])
    src_index = edge_index[0][sorted_index]
    dst_index = edge_index[1][sorted_index]
    src_size = dst_index.max() + 1
    src = torch.rand(src_size, feature_size).to(device)
    # create sparse tensor
    row = dst_index
    col = src_index
    value = torch.ones_like(row, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(torch.stack([row, col]), value, (dst_index.max() + 1, src_index.max() + 1))
    adj = adj.coalesce()

    # create sparse tensor for torch_sparse
    adj_torch_sparse = torch_sparse.SparseTensor(row=row, col=col, value=value, sparse_sizes=(dst_index.max() + 1, src_index.max() + 1))

    # benchmark time
    iter = 100
    timeit(gather_scatter, iter, src_index, dst_index, src, 'sum')
    timeit(pytorch_spmm, iter, adj, src)
    timeit(pyg_spmm, iter, adj_torch_sparse, src, 'sum')


if __name__ == '__main__':
    dataset = 'cora'
    feature_size = 64
    device = "cuda"
    test_gather_scatter(dataset, feature_size, device)
