import triton
import triton.language as tl

@triton.jit
def combine_fn(left_values, left_indices, right_values, right_indices):
    same_segment = left_indices == right_indices
    combined_values = tl.where(same_segment, left_values + right_values, right_values)
    combined_indices = right_indices
    return combined_values, combined_indices

@triton.jit
def parallel_segment_reduction_kernel(
    index,  # the input index tensor
    in_feature,  # the input tensor
    result,  # the output value tensor
    num_edges: tl.constexpr,  # Number of elements in the input tensor (1d)
    feature_size: tl.constexpr,  # Number of features in the input tensor (2d)
    BLOCK_SIZE: tl.constexpr,  # Block size for the scan
):
    pid = tl.program_id(axis=0)
    offset_pid = pid // feature_size
    feature_id = pid % feature_size
    offsets = offset_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_edges

    # Load input data
    values = tl.load(in_feature + offsets * feature_size + feature_id, mask=mask)
    indices = tl.load(index + offsets, mask=mask)
    indices_next = tl.load(index + offsets + 1, offsets < num_edges - 1)

    # Perform an inclusive scan using tl.associative_scan
    result_values, _ = tl.associative_scan(
        (values, indices,), axis=0, combine_fn=combine_fn
    )
    # if offset % BLOCK_SIZE == -1, it means the last element of the segment
    segment_start = (indices != indices_next) | (offsets % BLOCK_SIZE == BLOCK_SIZE - 1)
    tl.atomic_add(result + indices * feature_size + feature_id, result_values, mask & segment_start)


@triton.jit
def serial_segment_reduction_kernel(
        index, 
        in_feature, 
        result, 
        num_edges: tl.constexpr, 
        feature_size: tl.constexpr, 
        group_size: tl.constexpr
):
    group_id = tl.program_id(axis=0)
    node_offset = group_id * group_size
    f_index = tl.arange(0, feature_size)
    
    accumulate = tl.zeros((feature_size,), dtype=tl.float32)
    
    for ii in range(group_size):  # Iterate over the group
        xn = ii + node_offset  # Get node index
        mask = xn < num_edges  # Check if the node index is valid
        
        node_idx = tl.load(index + xn, mask=mask)
        next_idx = tl.load(index + xn + 1, mask = (xn+1) < num_edges)
        
        val = tl.load(in_feature + xn * feature_size + f_index, mask=mask)
        accumulate += val
        # Check for end of segment
        if node_idx != next_idx or ii == group_size - 1:
            # Perform atomic addition
            tl.atomic_add(result + node_idx * feature_size +
                          f_index, accumulate, mask=mask)
            # Clear accumulate for the next segment
            accumulate = tl.zeros(accumulate.shape, dtype=accumulate.dtype)



def launch_parallel_reduction(indices, input, output, num_edges:tl.constexpr, feature_size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    grid = (triton.cdiv(num_edges, BLOCK_SIZE) * feature_size,)
    parallel_segment_reduction_kernel[grid](indices, input, output, num_edges, feature_size, BLOCK_SIZE)


def launch_serial_reduction(edges, input, output, num_edges, feature_size, group_size):
    grid = (triton.cdiv(num_edges, group_size),)
    serial_segment_reduction_kernel[grid](edges, input, output, num_edges, feature_size, group_size)
   