import torch
import torch_index_scatter


def spmm_no_weight(adj_t, x, reduce):
    src = adj_t.storage.col()
    dst = adj_t.storage.row()
    return torch_index_scatter.gather_scatter(src, dst, x, reduce)
    