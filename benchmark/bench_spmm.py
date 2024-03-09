import torch
import torch_index_scatter
from utils import Dataset
import time
import torch_sparse
from scipy.sparse import csr_matrix
import dgsparse
from dgsparse import spmm_sum

def gather_scatter(src_index, dst_index, src, reduce):
    return torch_index_scatter.gather_scatter(src_index, dst_index, src, reduce)

def pytorch_spmm(A, B):
    return torch.sparse.mm(A, B)

def pyg_spmm(A, B, reduce):
    return torch_sparse.matmul(A, B, reduce)

def dgsparse_spmm(A, B):
    return spmm_sum(A, B, 0)

def timeit(func, iter, *args, **kwargs):
    start = time.time()
    for _ in range(iter):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()
    # return each func's ms
    print(f"{func.__name__} time: {(end - start) / iter * 1000:.3f} ms")
    return (end - start) / iter * 1000


def to_rowptr(row, size):
    rowptr = torch.zeros(size + 1, dtype=torch.int64)
    rowptr[1:] = torch.bincount(row, minlength=size)
    rowptr = rowptr.cumsum(0)
    return rowptr

def test_gather_scatter(file, dataset, feature_size, device):
    g = Dataset(dataset, device)
    edge_index = g.edge_index
    sparse_size = g.size
    sorted_index = torch.argsort(edge_index[1])
    src_index = edge_index[0][sorted_index]
    dst_index = edge_index[1][sorted_index]
    src_size = sparse_size
    src = torch.rand(src_size, feature_size).to(device)
    # create sparse tensor
    row = dst_index
    col = src_index
    value = torch.ones_like(row, dtype=torch.float32)
    rowptr = to_rowptr(row, sparse_size).to(device)
    # use rowptr, col, value to create sparse tensor for torch 
    # dgsparse need int type for rowptr and col
    adj = torch.sparse_csr_tensor(rowptr.int(), col.int(), value, (sparse_size, sparse_size))
    dgsparse_adj = dgsparse.SparseTensor.from_torch_sparse_csr_tensor(
            adj.detach(), True, requires_grad=False)

    # create sparse tensor for torch_sparse
    adj_torch_sparse = torch_sparse.SparseTensor(row=row, col=col, value=value, sparse_sizes=(sparse_size, sparse_size))

    # benchmark time
    iter = 100
    # write to csv file
    timeit(gather_scatter, iter, src_index, dst_index, src, 'sum')
    timeit(pytorch_spmm, iter, adj, src)
    timeit(pyg_spmm, iter, adj_torch_sparse, src, 'sum')
    timeit(dgsparse_spmm, iter, dgsparse_adj, src)


if __name__ == '__main__':
    dataset = 'ppi'
    feature_size = 64
    device = "cuda"
    with open("benchop_spmm.csv", "w") as file:
        test_gather_scatter(file, dataset, feature_size, device)
