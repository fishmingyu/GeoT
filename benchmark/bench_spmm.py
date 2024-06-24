import torch
import geot
from geot.triton import launch_parallel_spmm, launch_serial_spmm, launch_torch_compile_spmm
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

def test_gather_scatter(file, dataset, feature_size, device):
    g = Dataset(dataset, device)
    print(f'feature_size: {feature_size}')
    print(g.num_nodes, g.num_edges)
    sparse_size = g.num_nodes
    src_index = g.src_idx
    dst_index = g.dst_idx
    src_size = sparse_size
    src = torch.rand(src_size, feature_size).to(device)

    # output_pyg = torch.zeros(src_size, feature_size, dtype=torch.float32, device='cuda')
    # create sparse tensor
    row = dst_index
    col = src_index
    value = torch.ones_like(row, dtype=torch.float32)
    rowptr = to_rowptr(row, sparse_size).to(device)
    edge_raw = torch.stack([row, col], dim=0)
    _ , indices = torch.sort(edge_raw[1, :])
    edges = edge_raw[ : , indices]
    num_nodes = sparse_size
    num_edges = col.size()[0]
    # print(f'num_edges: {num_edges}')
    # print(f'num_nodes: {num_nodes}')

    output_SR = torch.zeros(num_nodes, feature_size, dtype=torch.float32, device='cuda')
    output_PR = torch.zeros(num_nodes, feature_size, dtype=torch.float32, device='cuda')
    output_compile = torch.zeros(num_nodes, feature_size, dtype=torch.float32, device='cuda')
    # output_torch = torch.zeros(num_nodes, feature_size, dtype=torch.float32, device='cuda')
    # output_geot = torch.zeros(num_nodes, feature_size, dtype=torch.float32, device='cuda')
    
    # use rowptr, col, value to create sparse tensor for torch 
    # dgsparse need int type for rowptr and col
    adj = torch.sparse_csr_tensor(rowptr, col, value, (sparse_size, sparse_size))
    # dgsparse_adj = dgsparse.SparseTensor.from_torch_sparse_csr_tensor(
    #         adj.detach(), True, requires_grad=False)

    # create sparse tensor for torch_sparse
    adj_torch_sparse = g.adj_t

    # test correctness
    output_torch = pytorch_spmm(adj, src)
    launch_serial_spmm(edges, src, output_SR, num_edges, feature_size, 32)
    launch_parallel_spmm(edges, src, output_PR, num_edges, feature_size, 32)
    launch_torch_compile_spmm(edges, src, output_compile, num_edges, feature_size, 32)
    output_cuda = gather_weight_scatter(src_index, dst_index, value, src, 'sum')

    diff_SR = torch.abs(output_torch - output_SR)
    diff_PR = torch.abs(output_torch - output_PR)
    diff_compile = torch.abs(output_torch - output_compile)
    diff_cuda = torch.abs(output_torch - output_cuda)
    
    # print out where the difference is
    print(f'diff_SR: {diff_SR.max()}, location: {torch.argmax(diff_SR)}')
    print(f'diff_PR: {diff_PR.max()}, location: {torch.argmax(diff_PR)}')
    print(f'diff_cuda: {diff_cuda.max()}, location: {torch.argmax(diff_cuda)}')
    print(f'diff_compile: {diff_compile.max()}, location: {torch.argmax(diff_compile)}')

    # assert torch.allclose(output_torch, output_SR, atol=1e-4)
    # assert torch.allclose(output_torch, output_PR, atol=1e-4)
    # assert torch.allclose(output_torch, output_cuda, atol=1e-4)
    # assert torch.allclose(output_torch, output_compile, atol=1e-4)

    # warm up
    for _ in range(10):
        gather_weight_scatter(src_index, dst_index, value, src, 'sum')
        pytorch_spmm(adj, src)
        pyg_spmm(adj_torch_sparse, src, 'sum')
        launch_parallel_spmm(edges, src, output_PR, num_edges, feature_size, 32)
        launch_serial_spmm(edges, src, output_SR, num_edges, feature_size, 32)
        launch_torch_compile_spmm(edges, src, output_compile, num_edges, feature_size, 32)
        

    # benchmark time
    iter = 100
    # write to csv file
    t1 = timeit(gather_weight_scatter, iter, src_index, dst_index, value, src, 'sum')
    t2 = timeit(pytorch_spmm, iter, adj, src)
    t3 = timeit(pyg_spmm, iter, adj_torch_sparse, src, 'sum')
    t4 = timeit(launch_parallel_spmm, iter, edges, src, output_PR, num_edges, feature_size, 32)
    t5 = timeit(launch_serial_spmm, iter, edges, src, output_SR, num_edges, feature_size, 32)
    t6 = timeit(launch_torch_compile_spmm, iter, edges, src, output_compile, num_edges, feature_size, 32)
    # t4 = timeit(dgsparse_spmm, iter, dgsparse_adj, src)
    # :.4f
    if file:
        file.write(f"{t1:.4f},{t2:.4f},{t3:.4f},{t4:.4f},{t5:.4f},{t6:.4f}")
    print('\n')

if __name__ == '__main__':
    datasets = ["cora", "citeseer", "pubmed", "amazon_photo", "ppi", "flickr", "ogbn-arxiv", "ogbl-collab", 'reddit2']
    # datasets = ["ppi", "ogbn-arxiv", "ogbl-collab", 'reddit2']
    features = [4, 8, 16, 32, 64, 128]
    device = "cuda"
    # test_gather_scatter(None, "cora", 128, "cuda")
    with open("benchop_spmm.csv", "w") as file:
        file.write("dataset,feature_size,gather_weight_scatter,pytorch_spmm,pyg_spmm,triton_pr,triton_sr,torch_compile\n")
        for dataset in datasets:
            for feature in features:
                file.write(f"{dataset},{feature},")
                test_gather_scatter(file, dataset, feature, device)
                file.write("\n")


# without modifying the file
# if __name__ == '__main__':
#     datasets = ["cora", "citeseer", "pubmed", "amazon_photo", "ppi", "flickr", "ogbn-arxiv", "ogbl-collab", 'reddit2']
#     # datasets = ["ppi", "ogbn-arxiv", "ogbl-collab", 'reddit2']
#     features = [4, 8, 16, 32, 64, 128]
#     device = "cuda"
#     for dataset in datasets:
#         for feature in features:
#             test_gather_scatter(None, dataset, feature, device)
                