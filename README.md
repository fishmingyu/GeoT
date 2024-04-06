# GeoT: Tensor Centric Library for Graph Neural Network via Efficient Segment Reduction on GPU

## This branch aims to develop and optimize the same method on Intel Processors (CPU and GPU) 

## Motivation
We have identified that although the scatter_reduce operation is a cornerstone in the construction of geometric deep learning systems, current deep learning frameworks have not optimized it effectively. This problem can be broken down into two main areas:

1. **Frontend Incompatibility**
   - In PyTorch, the [scatter_reduce](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html#torch.Tensor.scatter_reduce_) operation is used at the frontend. However, this inherently suggests that an implicit broadcast using an expanded index is mandatory, which contradicts the conventional usage in geometric deep learning.

## Motivation
We have identified that although the scatter and segment operations are cornerstone in the construction of geometric deep learning systems, current deep learning frameworks have not optimized it effectively. What's more, current framework solution does not support fusion for segment reduction. In torch.compile, the scatter and message will be fused into a fully-atomic SpMM, which largely affects the performance. (See benchmark detail [here](https://github.com/fishmingyu/inductor_test_gather_scatter))

## Our Proposed API

To address these challenges, we propose new operators and paradigms as shown in the following sections:

``` python
dst = index_scatter(dim, index, src, reduce, sorted=True) → Tensor
```

Contrary to the conventional [scatter_reduce](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html#torch.Tensor.scatter_reduce_) operation which allows for flexibility in the dimensionality of dst and src, our approach necessitates that both dst and src tensors share an identical number of dimensions. This constraint aligns our method more closely with the [index_reduce](https://pytorch.org/docs/stable/generated/torch.Tensor.index_reduce_.html#torch.Tensor.index_reduce_) operation. 

E.g. For a 3-D tensor with `reduce="sum"`, the output is calculated as follows:

```
dst[index[i]][j][k] += src[i][j][k]  # if dim == 0
dst[i][index[j]][k] += src[i][j][k]  # if dim == 1
dst[i][j][index[k]] += src[i][j][k]  # if dim == 2
```

Additionally, we have integrated a `sorted` flag to optimize the index_scatter_reduce kernel's performance. A sorted index enhances processing locality and minimizes atomic operations, thereby substantially improving the parallel processing efficiency for both CPUs and GPUs. This is formulated as implicit segment reduction (or [segment coo](https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_coo.html#torch_scatter.segment_coo)).

### Fusion ability

In Graph Neural Networks (GNNs), fusing the message and aggregation steps is a prevalent strategy. PyTorch Geometric ([PyG](https://github.com/pyg-team/pytorch_geometric)) utilizes the [torch_sparse](https://github.com/rusty1s/pytorch_sparse) library, which offers optimized Sparse Matrix-Matrix Multiplication (SpMM) to facilitate this process. Beyond traditional sparse format utilization, we have developed a method for efficient SpMM built upon segment reduction. Within GeoT, SpMM can be effortlessly executed using the gather_scatter function, where it's ensured that the adjacency matrix is sorted by column.


``` python
# consider adj as edge_index
# no weight
dst = gather_scatter(edge_index[0], edge_index[1], src, reduce) → Tensor
# with weight
dst = gather_weight_scatter(edge_index[0], edge_index[1], weight, src, reduce) → Tensor
```

This format-agnostic approach ensures compatibility with mainstream frameworks and compilers by achieving segmentation implicitly, without requiring explicit sparse format specification. We plan to integrate this feature into the [Triton](https://triton-lang.org/main/index.html) language in the future.

## Run GeoT
Setup 
```bash
python setup.py build install
```
Run benchmark
```bash
cd benchmark
python bench_index_scatter.py
python bench_spmm.py
```

For more detail, please check our paper. If you find this repo useful, please cite the following bib

``` bibtex
@article{yu2024geot,
  title={GeoT: Tensor Centric Library for Graph Neural Network via Efficient Segment Reduction on GPU},
  author={Yu, Zhongming and Zhang, Genghan and Huang, Hanxian and Chen, Xin and Zhao, Jishen},
  journal={arXiv preprint arXiv:2404.03019},
  year={2024}
}
```
