## Problem

Sometimes PyG's link will be broken. If this happens, the error is like:
``` bash
    raise ValueError("Cannot load file containing pickled data "
ValueError: Cannot load file containing pickled data when allow_pickle=False
```

To avoid this, you can change the method to dgl by using
``` python
from torch_geometric.utils import from_dgl
import dg

elif self.name == 'flickr': # due to PyG's broken link, we use dgl's dataset
    dataset = dgl.data.FlickrDataset()
    dgl_g = dataset[0]
    graph = from_dgl(dgl_g)
```
