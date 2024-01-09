import torch
import torch_index_scatter


def test_index_scatter():
    index_size = 10
    feature_size = 4
    reduce = 'sum'
    src = torch.rand(index_size, feature_size)
    index = torch.randint(0, index_size, (index_size,))
    out = torch_index_scatter.index_scatter(0, src, index, reduce, sorted=True)

    # use torch.scatter as reference
    ref = torch.zeros(index_size, feature_size)
    ref.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
    print(out)
    print(ref)


if __name__ == '__main__':
    test_index_scatter()
