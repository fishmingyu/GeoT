import torch
import torch.fx

def fused_transform_mh_spmm(graph_module : torch.fx.GraphModule) -> torch.fx.GraphModule:
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
            if node.args[0] == view_or_unsqueeze and node.args[1] == var_index_select:
                var_mul = node
                
        # locate index_add, then replace with mh_spmm
        if node.op == 'call_function' and node.target == torch.ops.aten.index_add.default:
            if node.args[3] == var_mul:
                # add a transpose to weight
                with graph.inserting_before(node):
                    var_weight = graph.call_function(torch.ops.aten.transpose, args=(var_weight, 0, 1))
                    
                with graph.inserting_before(node):
                    new_node = graph.call_function(
                        torch.ops.geot.mh_spmm, args=(var_row, var_col, var_weight, var_x, 'sum'))
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
    