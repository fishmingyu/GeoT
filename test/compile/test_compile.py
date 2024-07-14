import torch
from torch_geometric.profile import benchmark
from torch_geometric.utils import scatter
from torch.fx import symbolic_trace, subgraph_rewriter
from torch.export import export
from geot import index_scatter, gather_scatter

class TestModule(torch.nn.Module):
    def forward(self, x, edge_index, reduce='sum'):
        row, col = edge_index
        x_j = x[row]
        return scatter(x_j, col, dim_size=x.size(0), reduce=reduce)

# Basic "Gather-Apply-Scatter" patterns commonly used in PyG:
def pyg_gather_scatter(x, edge_index, reduce='sum'):
    row, col = edge_index
    x_j = x[row]
    return scatter(x_j, col, dim_size=x.size(0), reduce=reduce)


def test_torch_compile(device):
    x = torch.randn(10, 16, device=device)
    edge_index = torch.randint(0, x.size(0), (2, 40), device=device)

    expected = pyg_gather_scatter(x, edge_index)
    compiled_op = torch.compile(pyg_gather_scatter)
    out = compiled_op(x, edge_index)
    assert torch.allclose(out, expected, atol=1e-6)

# only replace index_add_
def pattern_scatter(out, dim, index, src):
    return torch.ops.aten.index_add.default(out, dim, index, src)

def replacement_scatter(out, dim, index, src):
    return index_scatter(dim, src, index, reduce='sum', sorted=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    # generate random data
    num_nodes, num_edges = 10_000, 200_000

    x = torch.randn(num_nodes, 64, device=args.device)
    edge_index = torch.randint(num_nodes, (2, num_edges), device=args.device)
    
    # sort edge_index by col
    _ , indices = torch.sort(edge_index[1, :])
    edge_index = edge_index[ : , indices]
    
    # check if already sorted by col
    row, col = edge_index
    assert torch.all(col[:-1] <= col[1:])
    
    # try replace pattern
    reduce = 'sum'
    module = TestModule()

    def compile_geot(x, edge_index, reduce):
        exported = export(module,(x, edge_index, reduce))
        NS = x.size(0)
        FS = x.size(1)
        exp_graph = exported.graph_module
        return exp_graph, NS, FS

    exp_graph, NS, FS = compile_geot(x, edge_index, reduce)
    
    # pattern replacement
    def pattern_whole(x, index):
        row = torch.ops.aten.select.int(index, 0, 0)
        col = torch.ops.aten.select.int(index, 0, 1)
        src = torch.ops.aten.index.Tensor(x, [row])
        new_zeros = torch.ops.aten.new_zeros.default(src, [NS, FS], pin_memory = False)
        return torch.ops.aten.index_add.default(new_zeros, 0, col, src)
    
    def replacement_whole(x, index):
        src_index = torch.ops.aten.select.int(index, 0, 0)
        dst_index = torch.ops.aten.select.int(index, 0, 1)
        return gather_scatter(src_index, dst_index, x, reduce='sum')
    
    print(f'\nBefore:{exp_graph.code}')
    subgraph_rewriter.replace_pattern(exp_graph, pattern_whole, replacement_whole)
    # subgraph_rewriter.replace_pattern(exp_graph, pattern_scatter, replacement_scatter)
    print(f'\nAfter:{exp_graph.code}')
    
    # # test correctness
    out_index = pyg_gather_scatter(x, edge_index, reduce)
    out_geot = exp_graph(x, edge_index, reduce)[0]
    diff = torch.abs(out_index - out_geot).max()
    print(f'max difference: {diff}') 
    assert torch.allclose(out_index, out_geot, atol=1e-5)   
    
    
    ### original
    # for reduce in ['sum']:
    #     print(f'Aggregator: {reduce}')
        
    #     benchmark(
    #         funcs=[
    #             exp_graph,
    #             torch.compile(exp_graph),
    #         ],
    #         func_names=['Vanilla', 'Compiled'],
    #         args=(x, edge_index, reduce),
    #         num_steps=50 if args.device == 'cpu' else 500,
    #         num_warmups=10 if args.device == 'cpu' else 100,
    #         backward=args.backward,
    #     )
    
    #     module = TestModule()

    #     exported = export(module,(x, edge_index, reduce))

    #     print(exported.graph_module)
