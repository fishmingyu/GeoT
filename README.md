# GeoT: Tensor Centric Library for Graph Neural Network via Efficient Segment Reduction on GPU

### TODO
* general reduction type
* autograd
* merge into torch_scatter

## Motivation
We have identified that although the scatter and segment operations are cornerstone in the construction of geometric deep learning systems, current deep learning frameworks have not optimized it effectively. What's more, current framework solution does not support fusion for segment reduction. In torch.compile, the scatter and message will be fused into a fully-atomic SpMM, which largely affects the performance. (See benchmark detail here)

## Our Proposed API

To address these challenges, we propose new operators and paradigms as shown in the following sections:

``` python
dst = index_scatter(dim, index, src, reduce, sorted=True) → Tensor
```

Unlike the traditional scatter_reduce operation, we mandate that `dst` and `src` have the same number of dimensions. The index is constrained to just one dimension. It is also required that `index.size(dim) <= src.size(dim)` for the specified `dim` argument.

For a 3-D tensor with `reduce="sum"`, the output is calculated as follows:

```
dst[index[i]][j][k] += src[i][j][k]  # if dim == 0
dst[i][index[j]][k] += src[i][j][k]  # if dim == 1
dst[i][j][index[k]] += src[i][j][k]  # if dim == 2
```

Additionally, we have integrated a `sorted` flag to optimize the index_scatter_reduce kernel's performance. A sorted index enhances processing locality and minimizes atomic operations, thereby substantially improving the parallel processing efficiency for both CPUs and GPUs. This also called segment reduction.

### Fusion ability

In GNN, message and aggregation fusion is a common technique. PyG use torch_sparse library, an optimized SpMM code to achieve this. Other than using sparse format, we achieve efficient SpMM on the top of segment reduction. In GeoT, the SpMM can simply be called via the ```gather_scatter```, in which we specify the adj is sorted by col.


``` python
# consider adj as edge_index
# no weight
dst = gather_scatter(edge_index[0], edge_index[1], src, reduce) → Tensor
# with weight
dst = gather_weight_scatter(edge_index[0], edge_index[1], weight, src, reduce) → Tensor
```

This format agnostic style keep the operator compatible with mainstream frameworks and compilers. We aim to add this feature to Triton lang in the feature.

For more detail, please check our paper. If you find this repo useful, please cite the following bib

``` latex

```
