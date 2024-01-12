import torch
import torch_index_scatter


def test_index_scatter():
    index_size = 10
    feature_size = 4
    reduce = 'sum'
    src = torch.rand(index_size, feature_size).to("cuda")
    print(src)
    index = torch.randint(0, 2, (index_size,)).to("cuda")
    sorted_index = torch.argsort(index)
    index = index[sorted_index]
    print(index)
    out = torch_index_scatter.index_scatter(0, src, index, reduce, sorted=True)

    # use torch.scatter as reference
    ref = torch.zeros(index_size, feature_size).to("cuda")
    ref.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
    print(out)
    print(ref)


if __name__ == '__main__':
    test_index_scatter()
