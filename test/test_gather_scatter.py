import torch
import torch_index_scatter

# out = src[src_index].index_add(0, dst_index, src)
def ref_gather_scatter(src_index, dst_index, src, reduce):
    keys = dst_index[-1] + 1
    ref = torch.zeros(keys, src.size(1)).to(src.device)
    ref.scatter_add_(0, dst_index.unsqueeze(-1).expand_as(src), src[src_index])
    return ref

def test_gather_scatter():
    src_index = torch.randint(0, 10, (1000,)).to("cuda")
    dst_index = torch.randint(0, 10, (1000,)).to("cuda")
    # sort dst_index
    sorted_index = torch.argsort(dst_index)
    dst_index = dst_index[sorted_index]
    src = torch.rand(1000, 32).to("cuda")
    reduce = 'sum'
    out = torch_index_scatter.gather_scatter(src_index, dst_index, src, reduce)
    ref = ref_gather_scatter(src_index, dst_index, src, reduce)
    assert torch.allclose(out, ref, atol=1e-4)

if __name__ == '__main__':
    test_gather_scatter()
