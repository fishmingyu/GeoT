import torch
import torch.fx
from torch.fx import subgraph_rewriter

def fused_weight_transform(graph_module : torch.fx.GraphModule) -> torch.fx.GraphModule:

    def pattern_weight_scatter_view(x, value, edge_index, ls, dim):
        col = torch.ops.aten.select.int(edge_index, 0, 1)
        row = torch.ops.aten.select.int(edge_index, 0, 0)
        index_select = torch.ops.aten.index_select.default(x, dim, row)
        view = torch.ops.aten.view.default(value, [-1, 1])
        mul = torch.ops.aten.mul.Tensor(view, index_select)
        new_zeros_1 = torch.ops.aten.new_zeros.default(mul, ls, pin_memory = False)
        return torch.ops.aten.index_add.default(new_zeros_1, 0, col, mul)
    
    def pattern_weight_scatter_unsqueeze(x, value, edge_index, ls, dim):
        col = torch.ops.aten.select.int(edge_index, 0, 1)
        row = torch.ops.aten.select.int(edge_index, 0, 0)
        index_select = torch.ops.aten.index_select.default(x, dim, row)
        unsqueeze = torch.ops.aten.unsqueeze.default(value, -1)
        mul = torch.ops.aten.mul.Tensor(unsqueeze, index_select)
        new_zeros_1 = torch.ops.aten.new_zeros.default(mul, ls, pin_memory = False)
        return torch.ops.aten.index_add.default(new_zeros_1, 0, col, mul)

    def replacement_weight_scatter(x, value, edge_index, ls, dim):
        row = torch.ops.aten.select.int(edge_index, 0, 0)
        col = torch.ops.aten.select.int(edge_index, 0, 1)
        return torch.ops.geot.gather_weight_scatter(row, col, value, x, 'sum')
    
    subgraph_rewriter.replace_pattern(graph_module, pattern_weight_scatter_view, replacement_weight_scatter)
    subgraph_rewriter.replace_pattern(graph_module, pattern_weight_scatter_unsqueeze, replacement_weight_scatter)
    return graph_module