import torch
import torch.fx
from torch.export import export
from .fused_gs import fused_transform 
from .fused_gws import fused_transform_gws
from .fused_mh_spmm import fused_transform_mh_spmm

def pattern_transform(model : torch.nn.Module, args, **kwargs) -> torch.export.ExportedProgram:
    exported = export(model, args, **kwargs)
    graph_module = exported.graph_module
    graph = graph_module.graph
    
    for node in graph.nodes:
        var_index_select : torch.fx.Node
        if node.op == 'call_function' and node.target == torch.ops.aten.index_add.default:
            dst = node.args[0]
            dim = node.args[1]
            var_index_select = node.args[3]
            if var_index_select.op == 'call_function' and var_index_select.target == torch.ops.aten.index_select.default:
                fused_transform(graph_module)
            elif var_index_select.op == 'call_function' and var_index_select.target == torch.ops.aten.mul.Tensor:
                if dst.op == 'call_function' and dst.target == torch.ops.aten.new_zeros.default:
                    size = dst.args[1]
                    if len(size) == 2:
                        fused_transform_gws(graph_module)
                    elif len(size) == 3:
                        fused_transform_mh_spmm(graph_module)
        
    graph.eliminate_dead_code()
    graph.lint()
    graph_module.recompile()
    
    return exported