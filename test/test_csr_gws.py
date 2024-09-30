import geot.csr_gws
import torch
import geot
from geot import csr_gws

def coo_to_csr(nrow, row, col, val):
    csr_row = torch.zeros(nrow + 1).to(row.device)
    csr_val = torch.zeros(val.size()).to(val.device)
    csr_row[1:] = torch.cumsum(torch.bincount(row, minlength=nrow), 0)
    csr_row = csr_row.int()
    # csr_val = val[torch.argsort(row)]
    return csr_row, col, val


def ref_spmm(sparse_size, src_index, dst_index, weight, src):
    # sparse_size = dst_index[-1] + 1
    # create sparse tensor
    row = dst_index
    col = src_index
    adj = torch.sparse_coo_tensor(
        torch.stack([row, col]), weight, (sparse_size, sparse_size)
    )
    adj = adj.coalesce()
    return torch.sparse.mm(adj, src)


def test_csr_gws():
    nrow = 100
    nnz = 1000
    feature_size = 32
    src_index = torch.randint(0, nrow, (nnz,)).to("cuda")  # col
    dst_index = torch.randint(0, nrow, (nnz,)).to("cuda")  # row
    weight = torch.rand(nnz).to("cuda")
    # sort
    dst_sorted, indices = torch.sort(dst_index)
    src_sorted = src_index[indices]
    weight_sorted = weight[indices]
    
    rowptr, colidx, val = coo_to_csr(nrow, dst_sorted, src_sorted, weight_sorted)

    src = torch.rand(nrow, feature_size).to("cuda")

    out = csr_gws(rowptr, colidx, val, src)
    ref = ref_spmm(nrow, src_sorted, dst_sorted, weight_sorted, src)
    diff = torch.abs(out - ref)
    print("diff: ", diff.max())
    assert torch.allclose(out, ref, atol=1e-4)


if __name__ == "__main__":
    test_csr_gws()
