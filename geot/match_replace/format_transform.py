import torch
import torch.fx
from ..triton.coo_to_csr import coo_to_hist_wrapper

@torch.library.custom_op("geot::coo_to_csr", mutates_args=())
def coo_to_csr(coo_row: torch.Tensor) -> torch.Tensor:
    coo_row = coo_row.to(torch.int32)
    
    nrow = coo_row.max() + 1
    nnz = coo_row.size(0)

    hist = torch.zeros(nrow, dtype=torch.int32, device="cuda")
    coo_to_hist_wrapper(coo_row, hist, nnz, 128)
    
    csr_row = torch.zeros(nrow + 1, dtype=torch.int32, device="cuda")
    csr_row[1:] = torch.cumsum(hist, 0)
    csr_row = csr_row.int()
    return csr_row
        
@torch.library.register_fake("geot::coo_to_csr")
def _(coo_row):
    ctx = torch.library.get_ctx()
    nrow_plus1 = ctx.new_dynamic_size()
    shape = [nrow_plus1]
    return coo_row.new_empty(shape, dtype=torch.int32)

def format_transform_for_reuse_gs(graph_module : torch.fx.GraphModule, node_index_add, node_row, node_col) -> torch.fx.Node:
    graph = graph_module.graph
    with graph.inserting_before(node_index_add):
        node_csrptr = graph.call_function(torch.ops.geot.coo_to_csr, args=(node_row,))
        weight_ones_int = graph.call_function(torch.ops.aten.ones_like, args=(node_col,))
        # convert to float32
        node_weight_ones = graph.call_function(torch.ops.aten.to, args=(weight_ones_int, torch.float32))
    return node_csrptr, node_weight_ones
    
def format_transform_for_reuse_gws(graph_module : torch.fx.GraphModule, node_index_add, node_row) -> torch.fx.Node:
    graph = graph_module.graph
    with graph.inserting_before(node_index_add):
        node_csrptr = graph.call_function(torch.ops.geot.coo_to_csr, args=(node_row,))
    return node_csrptr