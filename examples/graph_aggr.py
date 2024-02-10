import torch
from torch_geometric.nn import aggr

# Simple aggregations:
mean_aggr = aggr.MeanAggregation()

# Feature matrix holding 1000 elements with 64 features each:
x = torch.randn(1000, 64)

# Randomly assign elements to 100 sets:
index = torch.randint(0, 100, (1000, ))

output = mean_aggr(x, index)  #  Output shape: [100, 64]