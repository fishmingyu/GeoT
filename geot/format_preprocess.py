import torch

def csr_to_block_format(rowptr: torch.Tensor, col: torch.Tensor, value: torch.Tensor,
                      window_size: int, wide_size: int):
    """
    Processes a sparse matrix in CSR-like format by grouping rows into blocks.
    
    Args:
        rowptr (torch.Tensor): 1D tensor of row offsets (length = number of rows + 1).
        col (torch.Tensor): 1D tensor of column indices.
        value (torch.Tensor): 1D tensor of values.
        window_size (int): Number of rows per block.
        wide_size (int): Block width used for grouping/padding columns.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: new row pointer, column indices,
        and value tensors.
    """
    
    # Use local variable names similar to the C++ code
    window = window_size
    wide = wide_size

    # Convert tensors to Python lists for easier indexing.
    # (For large data, you might want to work with numpy arrays instead.)
    row_list = rowptr.tolist()         # row_list has length (num_rows + 1)
    column_list = col.tolist()   # list of int column indices
    value_list = value.tolist()     # list of float values

    # Total number of rows in the original matrix.
    rows = len(row_list) - 1
    rowsNew = rows // window

    # Dictionary to hold block results.
    res = {}

    # Process each block (each block contains "window" rows)
    for i in range(rowsNew):
        v = {}  # This will store fields: 'row', 'pad', 'colum', 'value'
        
        # Determine the range in the column list that covers the block.
        # Note: In CSR format, row_list[j] to row_list[j+1] gives nonzero indices for row j.
        start_idx = row_list[i * window]
        end_idx = row_list[i * window + window]
        
        # Merge columns for the entire block into a sorted set (unique columns).
        mergedSet = set(column_list[start_idx:end_idx])
        v_row = len(mergedSet)
        v['row'] = v_row

        # Determine padded number of columns.
        pad = ((v_row // wide + 1)) * wide
        if v_row % wide == 0:
            pad -= wide
        v['pad'] = pad

        # Sorted list of unique column indices.
        mergedVector = sorted(mergedSet)
        v['colum'] = mergedVector

        # Create a template for the new values.
        # Size is (number of unique columns in the block) * (number of rows in the block)
        demo = [0.0] * (v_row * window)
        
        # Build a mapping from each column to its index in the merged (sorted) vector.
        colmap = {col: idx for idx, col in enumerate(mergedVector)}
        
        # Calculate how many blocks (of width 'wide') we have based on padding.
        bIds = pad // wide

        p = 0  # row index within the current block (0 <= p < window)
        # Process each row in the current block.
        for j in range(i * window, (i + 1) * window):
            row_start = row_list[j]
            row_end = row_list[j + 1]
            for m in range(row_start, row_end):
                col_val = column_list[m]
                # Determine block id and the index inside the block (for the merged column index)
                bId = colmap[col_val] // wide
                bInId = colmap[col_val] % wide
                # Compute the product using value[j] and value[column[m]]
                prod = value_list[j] * value_list[col_val]
                if v_row % wide == 0:
                    # When the number of unique columns is an exact multiple of wide.
                    index = bId * window * wide + p * wide + bInId
                    demo[index] = prod
                else:
                    if bId < (bIds - 1):
                        index = bId * window * wide + p * wide + bInId
                        demo[index] = prod
                    else:
                        index = bId * window * wide + p * (v_row % wide) + bInId
                        demo[index] = prod
            p += 1
        
        v['value'] = demo
        res[i] = v

    # Combine the block results into final lists.
    rowNew = [0]
    colNew = []
    valueNew = []
    for i in sorted(res.keys()):
        block = res[i]
        # Update row pointer based on the number of unique columns in this block.
        rowNew.append(rowNew[-1] + block['row'])
        colNew.extend(block['colum'])
        valueNew.extend(block['value'])
    
    # Create new torch tensors (the copies own their data)
    rowTensor = torch.tensor(rowNew, dtype=torch.int32)
    colTensor = torch.tensor(colNew, dtype=torch.int32)
    valueTensor = torch.tensor(valueNew, dtype=torch.float32)

    return rowTensor, colTensor, valueTensor


# Example usage:
if __name__ == "__main__":
    # Example input tensors (adjust these as needed for your data)
    rowptr = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32)  # For a matrix with 4 rows.
    col = torch.tensor([1, 3, 0, 2, 1, 3, 0, 2], dtype=torch.int32)
    value = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float32)
    window_size = 2
    wide_size = 2

    rowTensor, colTensor, valueTensor = csr_to_block_format(rowptr, col, value, window_size, wide_size)
    print("Row Tensor:", rowTensor)
    print("Column Tensor:", colTensor)
    print("Value Tensor:", valueTensor)
