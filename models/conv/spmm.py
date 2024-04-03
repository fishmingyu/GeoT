import torch
import geot


def spmm_no_weight(adj_t, x, reduce):
    src = adj_t.storage._col
    dst = adj_t.storage._row
    return geot.gather_scatter(src, dst, x, reduce)
    
def spmm_weight(adj_t, x, reduce):
    src = adj_t.storage._col
    dst = adj_t.storage._row
    weight = adj_t.storage._value
    return geot.gather_weight_scatter(src, dst, weight, x, reduce)