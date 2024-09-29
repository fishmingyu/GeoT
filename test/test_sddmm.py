import torch
# import dgl.sparse as dglsp
import geot

import time
def timeit(func, iter, args, kwargs = {}):
    # benchmark time
    # warm up
    for i in range(10):
        func(*args, **kwargs)

    t = torch.zeros(iter)
    torch.cuda.synchronize()
    t1_start = time.perf_counter() 
    for i in range(iter):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    t1_end = time.perf_counter()
    t[i] = t1_end - t1_start
    print(f"Average time: {t.mean():.6f} s")
    return t


def func(src_index, dst_index, X1, X2, weight):
    ddmm = torch.ops.geot.sddmm_coo_impl(src_index, dst_index, X1, X2)
    return ddmm * weight


if __name__ == '__main__':
    nodes = 10000
    edges = 200000
    features = 128
    
    val = torch.arange(1, edges + 1).float().to("cuda")
    X1 = torch.ones(nodes, features).float().to("cuda")
    X2 = torch.ones(nodes, features).float().to("cuda")
    
    # length = 16
    src_index = torch.randint(0, nodes, (edges,)).to("cuda")
    dst_index = torch.randint(0, nodes, (edges,)).to("cuda")

    out_geot = func(src_index, dst_index, X1, X2, val)
    # print(out_geot)
    
    iter = 100
    t = timeit(func, iter, (src_index, dst_index, X1, X2, val), {})
    print(f"GFLOPS: {2 * edges * features / t.mean() / 1e9:.6f}")
    