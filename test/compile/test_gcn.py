import torch
from torch.export import export
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.testing import (
    onlyFullTest,
    onlyLinux,
    withPackage,
)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_nodes, num_edges = 10_000, 200_000
    x = torch.randn(num_nodes, 64, device=args.device)
    edge_index = torch.randint(num_nodes, (2, num_edges), device=args.device)
    
    # sort edge_index by col
    _ , indices = torch.sort(edge_index[1, :])
    edge_index = edge_index[ : , indices]
    
    # check if already sorted by col
    row, col = edge_index
    assert torch.all(col[:-1] <= col[1:])
    
    # args: no self loop
    kwargs = {'node_dim': 0}
    conv = SAGEConv(64, 64, **kwargs).to(args.device)
    out_conv = conv(x, edge_index)
    
    # get macros about size

    exported = export(conv,(x, edge_index))
    NS = x.size(0)
    FS = x.size(1)
    exp_graph = exported.graph_module
    reduce = 'sum'
    
    
    def transform(m : torch.fx.GraphModule) -> torch.fx.GraphModule:
        graph = m.graph
        for node in graph.nodes:
            # locate row and col
            if node.op == 'call_function' and node.target == torch.ops.aten.select.int:
                if node.args[2] == 0:
                    var_row = node
                elif node.args[2] == 1:
                    var_col = node
            # locate index_select i.e. x_j = x[row]
            if node.op == 'call_function' and node.target == torch.ops.aten.index_select.default:
                if str(node.args[0]) == "x" and node.args[2] == var_row:
                    var_index_select = node
                    var_x = node.args[0]   
            # locate index_add, then replace with gather_scatter 
            if node.op == 'call_function' and node.target == torch.ops.aten.index_add.default:
                if node.args[3] == var_index_select:
                    with graph.inserting_before(node):
                        new_node = graph.call_function(
                            torch.ops.geot.gather_scatter, args=(var_row, var_col, var_x, reduce))
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
        graph.eliminate_dead_code()
        graph.lint()
        m.recompile()
    
    print(f'\nBefore:{exp_graph.graph}')
    transform(exp_graph)
    print(f'\nAfter:{exp_graph.code}')
    
    compile_exported = torch.compile(exported.module())
    out_replace = exported.module()(x, edge_index)
    out_compile = compile_exported(x, edge_index)
    assert torch.allclose(out_replace, out_compile, atol=1e-6)
    
    diff = torch.abs(out_conv - out_compile).max()
    print(f'max difference: {diff}')
    assert torch.allclose(out_conv, out_compile, atol=1e-6)