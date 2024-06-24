import triton
import triton.language as tl

# torch compiled code for spmm
@triton.jit
def torch_compile_spmm(
    in_ptr0, in_ptr1, out_ptr0,     # edgs_index, src, dst
    num_edges : tl.constexpr, 
    feature_size : tl.constexpr, 
    XBLOCK : tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]  # weither [:] is a speedup or codegen necessary
    # xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // feature_size)
    x0 = xindex % feature_size
    # tmp0 = tl.load(in_ptr0 + (200000 + x1), None, eviction_policy='evict_last') 
    # tmp6 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    # tmp1 = tl.full([XBLOCK], 10000, tl.int32)
    # tmp2 = tmp0 + tmp1
    # tmp3 = tmp0 < 0
    # tmp4 = tl.where(tmp3, tmp2, tmp0)
    # tl.device_assert((0 <= tmp4) & (tmp4 < 10000), "index out of bounds: 0 <= tmp4 < 10000")
    # tmp7 = tmp6 + tmp1
    # tmp8 = tmp6 < 0
    # tmp9 = tl.where(tmp8, tmp7, tmp6)
    # tl.device_assert((0 <= tmp9) & (tmp9 < 10000), "index out of bounds: 0 <= tmp9 < 10000")
    
    out_idx = tl.load(in_ptr0 + (num_edges + x1), None, eviction_policy='evict_last')
    in_idx = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    value = tl.load(in_ptr1 + (x0 + (feature_size*in_idx)), None)
    tl.atomic_add(out_ptr0 + (x0 + (feature_size*out_idx)), value, None)  # all atomic add
    

def launch_torch_compile_spmm(in0, in1, out, num_edges, feature_size, XBLOCK):
    grid = (triton.cdiv(num_edges, XBLOCK),)
    torch_compile_spmm[grid](in0, in1, out, num_edges, feature_size, XBLOCK)
    
    

# torch compiled code for spmm
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12800000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]  # weither [:] is a speedup or codegen necessary
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 64)
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (200000 + x1), None, eviction_policy='evict_last') 
    tmp6 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    # tmp1 = tl.full([XBLOCK], 10000, tl.int32)
    # tmp2 = tmp0 + tmp1
    # tmp3 = tmp0 < 0
    # tmp4 = tl.where(tmp3, tmp2, tmp0)
    # tl.device_assert((0 <= tmp4) & (tmp4 < 10000), "index out of bounds: 0 <= tmp4 < 10000")
    # tmp7 = tmp6 + tmp1
    # tmp8 = tmp6 < 0
    # tmp9 = tl.where(tmp8, tmp7, tmp6)
    # tl.device_assert((0 <= tmp9) & (tmp9 < 10000), "index out of bounds: 0 <= tmp9 < 10000")
    
    tmp4 = tl.load(in_ptr0 + (200000 + x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (x0 + (64*tmp9)), None)
    tl.atomic_add(out_ptr0 + (x0 + (64*tmp4)), tmp11, None)