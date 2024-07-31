import torch
from torch_geometric.profile import benchmark
from torch_geometric.utils import scatter
from torch.fx import symbolic_trace, subgraph_rewriter
from torch.export import export
from geot import index_scatter, gather_scatter
from geot.match_replace import pattern_transform

# [TODO]: need to support index() besides index_select()
class TestModule(torch.nn.Module):
    def forward(self, x, edge_index, reduce='sum'):
        row, col = edge_index
        x_j = torch.index_select(x, 0, row)
        return scatter(x_j, col, dim_size=x.size(0), reduce=reduce)

# Basic "Gather-Apply-Scatter" patterns commonly used in PyG:
def pyg_gather_scatter(x, edge_index, reduce='sum'):
    row, col = edge_index
    x_j = x[row]
    return scatter(x_j, col, dim_size=x.size(0), reduce=reduce)

# for benchmarking
def test_torch_compile(device):
    x = torch.randn(10, 16, device=device)
    edge_index = torch.randint(0, x.size(0), (2, 40), device=device)

    expected = pyg_gather_scatter(x, edge_index)
    compiled_op = torch.compile(pyg_gather_scatter)
    out = compiled_op(x, edge_index)
    assert torch.allclose(out, expected, atol=1e-6)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    # generate random data
    num_nodes, num_edges = 10_000, 200_000

    x = torch.randn(num_nodes, 64, device=args.device, requires_grad=True)
    x_pyg = x
    x_geot = x.clone().detach().requires_grad_(True)
    
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
    
    # do the pattern replacement
    exported = pattern_transform(module, (x, edge_index, reduce))
    print(f'\nAfter:{exported.graph_module.code}')
    
    # compile the exported module
    model = exported.module()
    compiled = torch.compile(model)
    
    # test correctness
    out_pyg = pyg_gather_scatter(x_pyg, edge_index, reduce)
    out_geot = model(x_geot, edge_index, reduce)
    diff = torch.abs(out_pyg - out_geot).max()
    print(f'Forward max difference: {diff}\n') 
    assert torch.allclose(out_pyg, out_geot, atol=1e-5)   
    
    out_geot.sum().backward()
    out_pyg.sum().backward()
    
    print(f'Backward result of GeoT:\n{x_geot.grad}')
    print(f'Backward result of PyG:\n{x_pyg.grad}')
    
    
    
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
