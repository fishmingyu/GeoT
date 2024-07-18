import torch
import torch.fx
from torch.export import export
from .fused_scatter import fused_replace
from .fused_weight_scatter import fused_weight_replace

def pattern_transform(model : torch.nn.Module, args, **kwargs) -> torch.export.ExportedProgram:
    exported = export(model, args, **kwargs)
    graph_module = exported.graph_module
    graph = graph_module.graph
    
    for node in graph.nodes:
        var_index_select : torch.fx.Node
        if node.op == 'call_function' and node.target == torch.ops.aten.index_add.default:
            var_index_select = node.args[3]
            if var_index_select.op == 'call_function' and var_index_select.target == torch.ops.aten.index_select.default:
                fused_replace(graph_module)
            elif var_index_select.op == 'call_function' and var_index_select.target == torch.ops.aten.mul.Tensor:
                fused_weight_replace(graph_module)
        
    graph.eliminate_dead_code()
    graph.lint()
    graph_module.recompile()
    
    return exported