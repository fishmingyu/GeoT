import torch
import torch_index_scatter
from utils import Dataset
import time
import torch_sparse
from scipy.sparse import csr_matrix
# import dgsparse
# from dgsparse import spmm_sum

def gather_weight_scatter(src_index, dst_index, weight, src, reduce):
    return torch_index_scatter.gather_weight_scatter(src_index, dst_index, weight, src, reduce)

def pytorch_spmm(A, B):
    return torch.sparse.mm(A, B)

def pyg_spmm(A, B, reduce):
    return torch_sparse.matmul(A, B, reduce)

# def dgsparse_spmm(A, B):
#     return spmm_sum(A, B, 0)

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
    print(g.num_nodes, g.num_edges)
    sparse_size = g.num_nodes
    src_index = g.src_idx
    dst_index = g.dst_idx
    src_size = sparse_size
    src = torch.rand(src_size, feature_size).to(device)
    # create sparse tensor
    row = dst_index
    col = src_index
    value = torch.ones_like(row, dtype=torch.float32)
    rowptr = to_rowptr(row, sparse_size).to(device)
    # use rowptr, col, value to create sparse tensor for torch 
    # dgsparse need int type for rowptr and col
    adj = torch.sparse_csr_tensor(rowptr, col, value, (sparse_size, sparse_size))
    # dgsparse_adj = dgsparse.SparseTensor.from_torch_sparse_csr_tensor(
    #         adj.detach(), True, requires_grad=False)

    # create sparse tensor for torch_sparse
    adj_torch_sparse = g.adj_t

    # warm up
    for i in range(10):
        gather_weight_scatter(src_index, dst_index, value, src, 'sum')
        pytorch_spmm(adj, src)
        pyg_spmm(adj_torch_sparse, src, 'sum')

    # benchmark time
    iter = 100
    # write to csv file
    t1 = timeit(gather_weight_scatter, iter, src_index, dst_index, value, src, 'sum')
    t2 = timeit(pytorch_spmm, iter, adj, src)
    t3 = timeit(pyg_spmm, iter, adj_torch_sparse, src, 'sum')
    # t4 = timeit(dgsparse_spmm, iter, dgsparse_adj, src)
    # :.4f
    file.write(f"{t1:.4f},{t2:.4f},{t3:.4f}")


if __name__ == '__main__':
    datasets = ["cora", "citeseer", "pubmed", "amazon_photo", "ppi", "flickr", "ogbn-arxiv", "ogbl-collab", 'reddit2']
    features = [4, 8, 16, 32, 64, 128]
    device = "cuda"
    with open("benchop_spmm.csv", "w") as file:
        for dataset in datasets:
            for feature in features:
                file.write(f"{dataset},{feature},")
                test_gather_scatter(file, dataset, feature, device)
                file.write("\n")
                