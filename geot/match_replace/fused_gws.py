import torch
import torch.fx
from torch.fx import subgraph_rewriter

def fused_transform_gws(graph_module : torch.fx.GraphModule) -> torch.fx.GraphModule:
    graph = graph_module.graph
    var_mul = var_index_select = var_weight = var_x = view_or_unsqueeze = None
    for node in graph.nodes:
        # locate row and col
        if node.op == 'call_function' and node.target == torch.ops.aten.select.int:
            if node.args[2] == 0:
                var_row = node
            elif node.args[2] == 1:
                var_col = node
        # locate index_select i.e. x_j = x[row]
        if node.op == 'call_function' and node.target == torch.ops.aten.index_select.default:
            if node.args[1] == 0 or node.args[1] == -2:
                var_index_select = node
                var_x = node.args[0]   
                
        if node.op == 'call_function' and (node.target == torch.ops.aten.view.default 
                                           or node.target == torch.ops.aten.unsqueeze.default):
            var_weight = node.args[0]
            view_or_unsqueeze = node
            
        if node.op == 'call_function' and node.target == torch.ops.aten.mul.Tensor:
            if (node.args[0] == view_or_unsqueeze and node.args[1] == var_index_select
                ) or (node.args[1] == view_or_unsqueeze and node.args[0] == var_index_select):
                var_mul = node
                
        # locate index_add, then replace with gather_weight_scatter 
        if node.op == 'call_function' and node.target == torch.ops.aten.index_add.default:
            convert_to_csr = 0
            if node.args[3] == var_mul:
                if (convert_to_csr):
                    with graph.inserting_before(node):
                        csr_node = graph.call_function(torch.ops.geot.coo_to_csr, args=(var_row, var_col, var_weight,))
                        gws_node = graph.call_function(torch.ops.geot.csr_gws, args=(csr_node, var_col, var_weight, var_x,))
                        node.replace_all_uses_with(gws_node)
                        graph.erase_node(node)
                else:
                    with graph.inserting_before(node):
                        new_node = graph.call_function(
                            torch.ops.geot.gather_weight_scatter, args=(var_col, var_row, var_weight, var_x,))
                        node.replace_all_uses_with(new_node)
                        graph.erase_node(node)          # erase index_add
                    
                for node_replace in graph.nodes:
                    for n in range(len(node_replace.args)):
                        if node_replace.args[n] == var_mul:
                            args_list = list(node_replace.args) 
                            args_list[n] = var_x  
                            node_replace.args = tuple(args_list) 
                # erase mul and index_select
                # graph.erase_node(var_mul)
    return graph_module
    
    
    

    # def pattern_weight_scatter_view(x, value, edge_index, ls, dim):
    #     col = torch.ops.aten.select.int(edge_index, 0, 1)
    #     row = torch.ops.aten.select.int(edge_index, 0, 0)
    #     index_select = torch.ops.aten.index_select.default(x, dim, row)
    #     view = torch.ops.aten.view.default(value, [-1, 1])
    #     mul = torch.ops.aten.mul.Tensor(view, index_select)
    #     new_zeros_1 = torch.ops.aten.new_zeros.default(mul, ls, pin_memory = False)
    #     return torch.ops.aten.index_add.default(new_zeros_1, 0, col, mul)
    
    # def pattern_weight_scatter_unsqueeze(x, value, edge_index, ls, dim):
    #     col = torch.ops.aten.select.int(edge_index, 0, 1)
    #     row = torch.ops.aten.select.int(edge_index, 0, 0)
    #     index_select = torch.ops.aten.index_select.default(x, dim, row)
    #     unsqueeze = torch.ops.aten.unsqueeze.default(value, -1)
    #     mul = torch.ops.aten.mul.Tensor(unsqueeze, index_select)
    #     new_zeros_1 = torch.ops.aten.new_zeros.default(mul, ls, pin_memory = False)
    #     return torch.ops.aten.index_add.default(new_zeros_1, 0, col, mul)

    # def replacement_weight_scatter(x, value, edge_index, ls, dim):
    #     row = torch.ops.aten.select.int(edge_index, 0, 0)
    #     col = torch.ops.aten.select.int(edge_index, 0, 1)
    #     return torch.ops.geot.gather_weight_scatter(row, col, value, x, 'sum')
    
    # subgraph_rewriter.replace_pattern(graph_module, pattern_weight_scatter_view, replacement_weight_scatter)
    # subgraph_rewriter.replace_pattern(graph_module, pattern_weight_scatter_unsqueeze, replacement_weight_scatter)
    # return graph_module