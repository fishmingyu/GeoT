import torch
import geot

def ref_spmm(src_index, dst_index, weight, src):
    index_select = src.index_select(0, src_index)

    mul = weight.unsqueeze(-1) * index_select
    dst = torch.zeros_like(src)
    out = dst.index_add(0, dst_index, mul)
    return out

def test_mh_spmm():
    src_size = 100
    nnz_size = 1000
    feature_size = 32
    num_heads = 4
    src_index = torch.randint(0, src_size, (nnz_size,)).to("cuda")
    dst_index = torch.randint(0, src_size, (nnz_size,)).to("cuda")
    weight = torch.rand((nnz_size, num_heads)).to("cuda")
    # sort dst_index
    sorted_index = torch.argsort(dst_index)
    dst_index = dst_index[sorted_index]
    
    src = torch.rand(src_size, num_heads, feature_size).to("cuda")
    reduce = 'sum'
    out = geot.mh_spmm_transposed(src_index, dst_index, weight, src, reduce)
    ref = ref_spmm(src_index, dst_index, weight, src)
    assert torch.allclose(out, ref, atol=1e-4)

if __name__ == '__main__':
    test_mh_spmm()
