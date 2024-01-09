# Efficient Index Scatter Reduce for Geometric Deep Learning

## Motivation
We have identified that although the scatter_reduce operation is a cornerstone in the construction of geometric deep learning systems, current deep learning frameworks have not optimized it effectively. This problem can be broken down into two main areas:

1. **Frontend Incompatibility**
   - In PyTorch, the [scatter_reduce](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html#torch.Tensor.scatter_reduce_) operation is used at the frontend. However, this inherently suggests that an implicit broadcast using an expanded index is mandatory, which contradicts the conventional usage in geometric deep learning.

2. **Backend Inefficiency**
   - State-of-the-art frameworks like pytorch_scatter recognize this frontend flaw but fail to fully optimize the code for both CPU and GPU backends. In the CPU backend of pytorch_scatter, it defaults to sequential processing, which becomes exceedingly slow for large datasets.

## Our Proposed API

To address these challenges, we propose a new paradigm and frontend for this operation:

```
dst = index_scatter_reduce(dim, index, src, reduce, sorted=True) → Tensor
```

Unlike the traditional scatter_reduce operation, we mandate that `dst` and `src` have the same number of dimensions. The index is constrained to just one dimension. It is also required that `index.size(dim) <= src.size(dim)` for the specified `dim` argument.

For a 3-D tensor with `reduce="sum"`, the output is calculated as follows:

```
dst[index[i]][j][k] += src[i][j][k]  # if dim == 0
dst[i][index[j]][k] += src[i][j][k]  # if dim == 1
dst[i][j][index[k]] += src[i][j][k]  # if dim == 2
```

Additionally, we have integrated a `sorted` flag to optimize the index_scatter_reduce kernel's performance. A sorted index enhances processing locality and minimizes atomic operations, thereby substantially improving the parallel processing efficiency for both CPUs and GPUs. This feature can be implemented from the revised frontend, as shown in the PyG framework code (#TODO). A sorted index adheres to a non-decreasing order. For instance, `index = [0, 0, 0, 1, 1, 2]` qualifies as a sorted index when sorted=True.