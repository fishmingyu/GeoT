import torch
import scipy.sparse as sp
import numpy as np
import FS_Block


def csr_to_block_format(row1: torch.Tensor, column1: torch.Tensor, degree1: torch.Tensor,
                      window1: int, wide1: int):
    window = window1
    wide = wide1

    # Convert tensors to Python lists for indexing (assumes CPU tensors)
    # row1 is expected to be a 1D tensor of ints with size (n+1,)
    row = row1.tolist()
    column = column1.tolist()  # assume 1D tensor of ints
    degree = degree1.tolist()  # assume 1D tensor of floats

    rows = len(row) - 1
    rowsNew = rows // window

    res = {}  # will store results per block

    for i in range(rowsNew):
        v = {}  # dictionary to mimic Value_tf32

        # Create mergedVector from column[row[i*window] : row[i*window+window]]
        start = row[i * window]
        end = row[i * window + window]
        mergedVector = column[start:end]

        # Count occurrences of each element in mergedVector
        elementCounts = {}
        for element in mergedVector:
            elementCounts[element] = elementCounts.get(element, 0) + 1

        # Create a sorted list of (column, count) pairs, sorted by column index
        countVector = sorted(elementCounts.items(), key=lambda x: x[0])
        v_row = len(countVector)
        v['row'] = v_row

        # Compute padding: pad = ((v.row / wide + 1)) * wide, subtract wide if v.row is a multiple of wide
        v_pad = ((v_row // wide) + 1) * wide
        if v_row % wide == 0:
            v_pad -= wide
        v['pad'] = v_pad

        if v_row > 0:
            # Build colum list and a mapping from column value to its index (order)
            colmap = {}
            v_colum = []
            for c, (col_val, _) in enumerate(countVector):
                v_colum.append(col_val)
                colmap[col_val] = c
            # # print colmap when debugging
            # print(colmap)
            v['colum'] = v_colum

            # Preallocate demo vector.
            # In the C++ code, demo is resized to v.row * window.
            demo = [0.0] * (v_row * window)
            bIds = v_pad // wide

            p = 0
            for j in range(i * window, (i + 1) * window):
                # For each row in the block, iterate over its entries from row[j] to row[j+1]
                for m in range(row[j], row[j + 1]):
                    col_val = column[m]
                    # Find the mapped column index and its block info
                    mapped_index = colmap[col_val]
                    bId = mapped_index // wide
                    bInId = mapped_index % wide

                    if v_row % wide == 0:
                        # When v.row is a multiple of wide, the index computation is:
                        index = bId * window * wide + p * wide + bInId
                        demo[index] = degree[m]
                    else:
                        if bId < (bIds - 1):
                            index = bId * window * wide + p * wide + bInId
                            demo[index] = degree[m]
                        else:
                            index = bId * window * wide + p * (v_row % wide) + bInId
                            demo[index] = degree[m]
                p += 1
            v['value'] = demo

        res[i] = v

    # Assemble the final rowNew, colNew, and valueNew lists.
    rowNew = [0]
    colNew = []
    valueNew = []
    # Since res keys are in order from 0 to rowsNew-1, we iterate in sorted order.
    for i in sorted(res.keys()):
        v = res[i]
        rowNew.append(rowNew[-1] + v.get('row', 0))
        if 'colum' in v:
            colNew.extend(v['colum'])
        if 'value' in v:
            valueNew.extend(v['value'])

    # Create tensors from the lists.
    # Note: We create new tensors to own the data.
    rowTensor = torch.tensor(rowNew, dtype=torch.int32)
    colTensor = torch.tensor(colNew, dtype=torch.int32)
    valueTensor = torch.tensor(valueNew, dtype=torch.float32)

    return [rowTensor, colTensor, valueTensor]


# Example usage:
if __name__ == "__main__":
    # use random sparse matrix
    m = 1000
    n = 1000
    density = 0.01
    sparse_matrix = sp.random(m, n, density=density, format='csr', dtype=np.float32)
    row = torch.tensor(sparse_matrix.indptr, dtype=torch.int32)
    col = torch.tensor(sparse_matrix.indices, dtype=torch.int32)
    value = torch.tensor(sparse_matrix.data, dtype=torch.float32)
    window = 8
    wide = 8
    rowTensor, colTensor, valueTensor = csr_to_block_format(row, col, value, window, wide)  
    # test match
    rowTensor1, colTensor1, valueTensor1 = FS_Block.blockProcess_tf32(row, col, value, window, wide)
    assert torch.equal(rowTensor, rowTensor1)
    assert torch.equal(colTensor, colTensor1)
    assert torch.equal(valueTensor, valueTensor1)

