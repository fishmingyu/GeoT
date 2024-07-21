import torch
import torch.fx

def fused_transform(graph_module : torch.fx.GraphModule) -> torch.fx.GraphModule:
    graph = graph_module.graph
    
    for node in graph.nodes:
        # locate row and col
        var_index_select : torch.fx.Node
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
        # locate index_add, then replace with gather_scatter 
        if node.op == 'call_function' and node.target == torch.ops.aten.index_add.default:
            if node.args[3] == var_index_select:
                with graph.inserting_before(node):
                    new_node = graph.call_function(
                        torch.ops.geot.gather_scatter, args=(var_row, var_col, var_x, 'sum'))
                    node.replace_all_uses_with(new_node)
                    graph.erase_node(node)          # erase index_add_1
                # replace all uses of index_select with x (only used its dtype and device)
                for node_replace in graph.nodes:
                    for n in range(len(node_replace.args)):
                        if node_replace.args[n] == var_index_select:
                            args_list = list(node_replace.args) 
                            args_list[n] = var_x  
                            node_replace.args = tuple(args_list) 
                # erase index_select
                graph.erase_node(var_index_select)

    return graph_module