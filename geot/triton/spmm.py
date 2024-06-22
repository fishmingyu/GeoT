import triton
import triton.language as tl

@triton.jit
def combine_fn(left_values, left_indices, right_values, right_indices):
    same_segment = left_indices == right_indices
    combined_values = tl.where(same_segment, left_values + right_values, right_values)
    combined_indices = right_indices
    return combined_values, combined_indices

@triton.jit
def parallel_spmm_sorted_coo_kernel(
    edge_index,  # the input coo sparse matrix
    input,  # the input tensor
    output,  # the output value tensor
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
    in_idx = tl.load(edge_index + offsets, mask=mask)
    values = tl.load(input + in_idx * feature_size + feature_id, mask=mask)
    # values = tl.load(input + offsets * feature_size + feature_id, mask=mask)
    out_idx = tl.load(edge_index + offsets + num_edges, mask=mask)
    out_idx_next = tl.load(edge_index + offsets + num_edges + 1, offsets < num_edges - 1)

    # Perform an inclusive scan using tl.associative_scan
    result_values, _ = tl.associative_scan(
        (values, out_idx,), axis=0, combine_fn=combine_fn
    )
    # if offset % BLOCK_SIZE == -1, it means the last element of the segment
    segment_start = (out_idx != out_idx_next) | (offsets % BLOCK_SIZE == BLOCK_SIZE - 1)
    tl.atomic_add(output + out_idx * feature_size + feature_id, result_values, mask & segment_start)    


@triton.jit
def serial_spmm_sorted_coo_naive_kernel(
    edge_index, 
    input, 
    output, 
    num_edges: tl.constexpr, 
    feature_size: tl.constexpr, 
    group_size: tl.constexpr
):
    group_id = tl.program_id(0)
    node_offset = group_id * group_size
    f_index = tl.arange(0, feature_size)
    
    accumulate = tl.zeros((feature_size,), dtype=tl.float32)

    for ii in range(group_size):  # Iterate over the group
        xn = ii + node_offset  # Get node index
        mask = xn < num_edges  # Check if the node index is valid
        
        # Load 1st row as src, 2nd row as dst
        out_node = tl.load(edge_index + xn + num_edges, mask=mask)
        next_node = tl.load(edge_index + xn + 1 + num_edges, mask = (xn+1) < num_edges)
        
        in_node = tl.load(edge_index + xn, mask=mask)  # Load the input node
        val = tl.load(input + in_node * feature_size + f_index, mask=mask)
        accumulate += val
        # Check for end of segment
        if out_node != next_node or ii == group_size - 1:
            # Perform atomic addition
            tl.atomic_add(output + out_node * feature_size +
                          f_index, accumulate, mask=mask)
            # Reset val for the next segment
            accumulate = tl.zeros(accumulate.shape, dtype=accumulate.dtype)



def launch_parallel_spmm(indices, input, output, num_edges:tl.constexpr, feature_size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    grid = (triton.cdiv(num_edges, BLOCK_SIZE) * feature_size,)
    parallel_spmm_sorted_coo_kernel[grid](indices, input, output, num_edges, feature_size, BLOCK_SIZE)


def launch_serial_spmm(edges, input, output, num_edges, feature_size, group_size):
    grid = (triton.cdiv(num_edges, group_size),)
    serial_spmm_sorted_coo_naive_kernel[grid](edges, input, output, num_edges, feature_size, group_size)
   