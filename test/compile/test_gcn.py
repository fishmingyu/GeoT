import torch
from torch.export import export
from torch.fx import subgraph_rewriter
from torch_geometric.nn import GCNConv
from torch_geometric.testing import (
    onlyFullTest,
    onlyLinux,
    withPackage,
)
import geot

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    # generate random input
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
    conv = GCNConv(64, 64, **kwargs).to(args.device)
    
    # get control output
    out_gcn = conv(x, edge_index)
    
    # get macros about size
    exported = export(conv,(x, edge_index))
    NS = x.size(0)
    FS = x.size(1)
    exp_graph = exported.graph_module
    reduce = 'sum'
    
    # define pattern and replacement
    def pattern_weight_scatter(linear, mul_1, cat):
        select_4 = torch.ops.aten.select.int(cat, 0, 1)
        select_5 = torch.ops.aten.select.int(cat, 0, 0)
        index_select = torch.ops.aten.index_select.default(linear, 0, select_5)
        view_1 = torch.ops.aten.view.default(mul_1, [-1, 1])
        mul_2 = torch.ops.aten.mul.Tensor(view_1, index_select)
        new_zeros_1 = torch.ops.aten.new_zeros.default(mul_2, [10000, 64], pin_memory = False)
        return torch.ops.aten.index_add.default(new_zeros_1, 0, select_4, mul_2)
    
    def replacement_weight_scatter(linear, mul_1, cat):
        row = torch.ops.aten.select.int(cat, 0, 0)
        col = torch.ops.aten.select.int(cat, 0, 1)
        return torch.ops.geot.gather_weight_scatter(row, col, mul_1, linear, reduce)
        
    print(f'\nBefore:{exp_graph.code}')
    subgraph_rewriter.replace_pattern(exp_graph, pattern_weight_scatter, replacement_weight_scatter)
    print(f'\nAfter:{exp_graph.code}')
    
    compile_exported = torch.compile(exported.module())
    out_replace = exported.module()(x, edge_index)
    out_compile = compile_exported(x, edge_index)
    assert torch.allclose(out_replace, out_compile, atol=1e-6)
    
    diff = torch.abs(out_gcn - out_compile).max()
    print(f'max difference with GCN: {diff}')
    assert torch.allclose(out_gcn, out_compile, atol=1e-6)