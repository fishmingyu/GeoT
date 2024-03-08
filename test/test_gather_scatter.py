import torch
import torch_index_scatter

def ref_spmm(src_index, dst_index, src):
    sparse_size = dst_index[-1] + 1
    # create sparse tensor
    row = dst_index
    col = src_index
    value = torch.ones_like(row, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(torch.stack([row, col]), value, (sparse_size, sparse_size))
    adj = adj.coalesce()
    return torch.sparse.mm(adj, src)

def test_gather_scatter():
    src_size = 100
    nnz_size = 1000
    feature_size = 32   
    src_index = torch.randint(0, src_size, (nnz_size,)).to("cuda")
    dst_index = torch.randint(0, src_size, (nnz_size,)).to("cuda")
    # sort dst_index
    sorted_index = torch.argsort(dst_index)
    dst_index = dst_index[sorted_index]
    src = torch.rand(src_size, feature_size).to("cuda")
    reduce = 'sum'
    out = torch_index_scatter.gather_scatter(src_index, dst_index, src, reduce)
    ref = ref_spmm(src_index, dst_index, src)
    assert torch.allclose(out, ref, atol=1e-4)

if __name__ == '__main__':
    test_gather_scatter()
