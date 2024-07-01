import torch
import geot
from geot.triton import launch_pr_spmm, launch_sr_spmm, launch_torch_compile_spmm
from utils import Dataset
import time
import torch_sparse
from scipy.sparse import csr_matrix
# import dgsparse
# from dgsparse import spmm_sum

def gather_weight_scatter(src_index, dst_index, weight, src, reduce):
    return geot.gather_weight_scatter(src_index, dst_index, weight, src, reduce)

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

def test_triton_spmm(file, dataset, feature_size, device, group_size):
    g = Dataset(dataset, device)
    print(f'feature_size: {feature_size}')
    print(g.num_nodes, g.num_edges)
    sparse_size = g.num_nodes
    src_index = g.src_idx
    dst_index = g.dst_idx
    src_size = sparse_size
    src = torch.rand(src_size, feature_size).to(device)

    # create sparse tensor
    row = dst_index
    col = src_index
    
    edge_raw = torch.stack([col, row], dim=0)
    _ , indices = torch.sort(edge_raw[1, :])
    edges = edge_raw[ : , indices]
    num_nodes = sparse_size
    num_edges = col.size()[0]

    output_sr = torch.zeros(num_nodes, feature_size, dtype=torch.float32, device='cuda')
    output_pr = torch.zeros(num_nodes, feature_size, dtype=torch.float32, device='cuda')
    output_compile = torch.zeros(num_nodes, feature_size, dtype=torch.float32, device='cuda')
    
    # warm up
    for _ in range(10):
        launch_pr_spmm(edges, src, output_pr, num_edges, feature_size, group_size)
        launch_sr_spmm(edges, src, output_sr, num_edges, feature_size, group_size)
        launch_torch_compile_spmm(edges, src, output_compile, num_edges, feature_size, group_size)
        
    # benchmark time
    iter = 100
    # write to csv file
    t1 = timeit(launch_pr_spmm, iter, edges, src, output_pr, num_edges, feature_size, group_size)
    t2 = timeit(launch_sr_spmm, iter, edges, src, output_sr, num_edges, feature_size, group_size)
    t3 = timeit(launch_torch_compile_spmm, iter, edges, src, output_compile, num_edges, feature_size, group_size)
    # :.4f
    if file:
        file.write(f"{t1:.4f},{t2:.4f},{t3:.4f}")
        

def test_gather_scatter(file, dataset, feature_size, device, group_size):
    g = Dataset(dataset, device)
    print(f'feature_size: {feature_size}')
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
    edge_raw = torch.stack([col, row], dim=0)
    _ , indices = torch.sort(edge_raw[1, :])
    edges = edge_raw[ : , indices]
    num_nodes = sparse_size
    num_edges = col.size()[0]

    output_SR = torch.zeros(num_nodes, feature_size, dtype=torch.float32, device='cuda')
    output_PR = torch.zeros(num_nodes, feature_size, dtype=torch.float32, device='cuda')
    output_compile = torch.zeros(num_nodes, feature_size, dtype=torch.float32, device='cuda')
    
    # use rowptr, col, value to create sparse tensor for torch 
    # dgsparse need int type for rowptr and col
    adj = torch.sparse_csr_tensor(rowptr, col, value, (sparse_size, sparse_size))
    # dgsparse_adj = dgsparse.SparseTensor.from_torch_sparse_csr_tensor(
    #         adj.detach(), True, requires_grad=False)

    # create sparse tensor for torch_sparse
    adj_torch_sparse = g.adj_t

    # warm up
    for _ in range(10):
        gather_weight_scatter(src_index, dst_index, value, src, 'sum')
        pytorch_spmm(adj, src)
        pyg_spmm(adj_torch_sparse, src, 'sum')
        launch_pr_spmm(edges, src, output_PR, num_edges, feature_size, group_size)
        launch_sr_spmm(edges, src, output_SR, num_edges, feature_size, group_size)
        launch_torch_compile_spmm(edges, src, output_compile, num_edges, feature_size, group_size)
        
    # benchmark time
    iter = 100
    # write to csv file
    t1 = timeit(gather_weight_scatter, iter, src_index, dst_index, value, src, 'sum')
    t2 = timeit(pytorch_spmm, iter, adj, src)
    t3 = timeit(pyg_spmm, iter, adj_torch_sparse, src, 'sum')
    t4 = timeit(launch_pr_spmm, iter, edges, src, output_PR, num_edges, feature_size, group_size)
    t5 = timeit(launch_sr_spmm, iter, edges, src, output_SR, num_edges, feature_size, group_size)
    t6 = timeit(launch_torch_compile_spmm, iter, edges, src, output_compile, num_edges, feature_size, group_size)
    # t4 = timeit(dgsparse_spmm, iter, dgsparse_adj, src)
    # :.4f
    if file:
        file.write(f"{t1:.4f},{t2:.4f},{t3:.4f},{t4:.4f},{t5:.4f},{t6:.4f}")
    print('\n')

if __name__ == '__main__':
    mode = 'triton'
    group_size = 32
    datasets = ["cora", "citeseer", "pubmed", "amazon_photo", "ppi", "flickr", "ogbn-arxiv", "ogbl-collab", 'reddit2']
    features = [1, 2, 4, 8, 16, 32, 64, 128]
    device = "cuda"
        
    # original test
    if mode == 'original':
        with open(f"benchop_spmm.csv", "w") as file:
            file.write("dataset,feature_size,gather_weight_scatter,pytorch_spmm,pyg_spmm,triton_pr,triton_sr,torch_compile\n")
            for dataset in datasets:
                for feature in features:
                    file.write(f"{dataset},{feature},")
                    test_gather_scatter(file, dataset, feature, device, group_size)
                    file.write("\n")

    # comparison for 3 triton methods
    if mode == 'triton':
        with open(f"benchop_spmm_triton_backup.csv", "w") as file:
            file.write("dataset,feature_size,triton_pr,triton_sr,torch_compile\n")
            for dataset in datasets:
                for feature in features:
                    file.write(f"{dataset},{feature},")
                    test_triton_spmm(file, dataset, feature, device, group_size)
                    file.write("\n")
