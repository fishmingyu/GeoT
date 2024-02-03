import torch
import torch_index_scatter


def test_index_scatter():
    index_size = 100
    feature_size = 16
    reduce = 'sum'
    src = torch.rand(index_size, feature_size).to("cuda")
    index = torch.randint(0, 10, (index_size,)).to("cuda")
    sorted_index = torch.argsort(index)
    index = index[sorted_index]
    keys = index[-1] + 1
    out = torch_index_scatter.index_scatter(0, src, index, reduce, sorted=True)

    # use torch.scatter as reference
    ref = torch.zeros(keys, feature_size).to("cuda")
    ref.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
    assert torch.allclose(out, ref, atol=1e-6)


if __name__ == '__main__':
    test_index_scatter()
