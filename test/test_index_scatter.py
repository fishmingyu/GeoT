import torch
import geot


def test_index_scatter():
    index_size = 1000
    feature_size = 32
    src = torch.rand(index_size, feature_size).to("cuda")
    index = torch.randint(0, 10, (index_size,)).to("cuda")
    sorted_index = torch.argsort(index)
    index = index[sorted_index]
    keys = index[-1] + 1
    out = geot.index_scatter(0, src, index, sorted=False)

    # use torch.scatter as reference
    ref = torch.zeros(keys, feature_size).to("cuda")
    ref.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
    assert torch.allclose(out, ref, atol=1e-4)

    ref = torch.zeros(keys, feature_size).to("cuda")
    ref.index_add_(0, index, src)
    assert torch.allclose(out, ref, atol=1e-4)


if __name__ == '__main__':
    test_index_scatter()
