import torch
import geot
from utils import Dataset

def coo_to_csr(nrow, row):
    csr_row = torch.zeros(nrow + 1).to(row.device)
    csr_row[1:] = torch.cumsum(torch.bincount(row, minlength=nrow), 0)
    csr_row = csr_row.int()
    return csr_row

def ref_spmm(src_index, dst_index, weight, src):
    sparse_size = dst_index[-1] + 1
    # create sparse tensor
    row = dst_index
    col = src_index
    adj = torch.sparse_coo_tensor(torch.stack([row, col]), weight, (sparse_size, sparse_size))
    adj = adj.coalesce()
    return torch.sparse.mm(adj, src)

def test_csr_gws_rand():
    nrow = 10000
    nnz = 100000
    feature_size = 100
    src_index = torch.randint(0, nrow, (nnz,)).to("cuda")  # col
    dst_index = torch.randint(0, nrow, (nnz,)).to("cuda")  # row
    weight = torch.rand(nnz).to("cuda")
    src = torch.rand((nrow, feature_size), device = 'cuda')
    # sort
    row, indices = torch.sort(dst_index)
    col = src_index[indices]
    weight_sorted = weight[indices]
    
    rowptr = torch.ops.geot.coo_to_csr(row)
    
    out = torch.ops.geot.csr_gws(rowptr, col, weight_sorted, src)
    ref = ref_spmm(col, row, weight_sorted, src)
    diff = torch.abs(out - ref).flatten()
    print("diff: ", diff.topk(10).values)
    assert torch.allclose(out, ref, atol=1e-4)
    
def test_csr_gws_dataset(dataset='flickr'):
    d = Dataset(dataset, 'cuda')
    data = d.edge_index
    src = d.x
    row = data[0]
    col = data[1]
    
    nrow = row[-1] + 1
    nnz = row.size(0)
    weight = torch.rand(nnz, device = 'cuda')
    
    # test correctness of coo_to_csr
    rowptr_control = coo_to_csr(nrow, row)
    rowptr_geot = torch.ops.geot.coo_to_csr(row)
    diff = torch.abs(rowptr_control - rowptr_geot)
    print("diff: ", diff.topk(10).values)
    assert torch.allclose(rowptr_control, rowptr_geot)

    # test correctness of csr_gws
    out = torch.ops.geot.csr_gws(rowptr_geot, col, weight, src)
    ref = ref_spmm(col, row, weight, src)
    diff = torch.abs(out - ref)
    print("diff: ", diff.topk(10).values)
    assert torch.allclose(out, ref, atol=1e-4)


if __name__ == "__main__":
    print("Testing csr_gws by random data")
    test_csr_gws_rand()
    print("\nTesting csr_gws by dataset Flickr")
    test_csr_gws_dataset('flickr')
