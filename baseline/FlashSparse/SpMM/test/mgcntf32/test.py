import torch
import FS_Block
import FS_SpMM
import scipy.sparse as sp
import numpy as np

def check(row_pointers1, column_index1, value, rhs, n):
    # use torch sparse spmm
    # Convert CSR representation to torch sparse tensor
    values = value
    sparse_tensor = torch.sparse_csr_tensor(
        crow_indices=row_pointers1,
        col_indices=column_index1,
        values=values,
        size=(n, n)
    )
    
    # Perform sparse matrix multiplication
    result = torch.matmul(sparse_tensor, rhs)
    
    return result

m = 10000
n = 10000
density = 0.001
sparse_matrix = sp.random(m, n, density=density, format='csr', dtype=np.float32)
row = torch.tensor(sparse_matrix.indptr, dtype=torch.int32)
col = torch.tensor(sparse_matrix.indices, dtype=torch.int32)
value = torch.ones_like(torch.tensor(sparse_matrix.data, dtype=torch.float32))
window = 8
wide = 4

row_pointers, column_index, degrees=FS_Block.blockProcess_tf32(row, col, value, window, wide)

print(row_pointers)
print(column_index)
print(degrees.size())
# print(degrees)
print()

num_nodes_ori = m
num_nodes=m + 16 - m % 16

dimN =32
#rhs = torch.ones((30, dimN), dtype=torch.float32)
rhs = torch.randint(low=0, high=3, size=(num_nodes_ori, dimN))
# rhs = rhs.half()
rhs = rhs.float()
result, spmm_ms_avg =  FS_SpMM.forward_tf32_map(   
                        row_pointers, 
                        column_index, 
                        degrees, 
                        rhs, 
                        num_nodes, 
                        rhs.size(1), 
                        num_nodes_ori, 1)

res = check(row,col,value,rhs,num_nodes_ori)
print(result)
print(res)

for i in range(num_nodes_ori):
	if (result[i][0] - res[i][0]) != 0 :
			print("No")
			exit(0)
	if (result[i][dimN-1] - res[i][dimN-1]) != 0 :
			print("No")
			exit(0)

print("PASS")


# test time
import time

epoch = 1000
start = time.time()
# cuda synchronize
torch.cuda.synchronize()

result, spmm_ms_avg = FS_SpMM.forward_tf32_map(   
                        row_pointers, 
                        column_index, 
                        degrees, 
                        rhs, 
                        num_nodes, 
                        rhs.size(1), 
                        num_nodes_ori, epoch)
# cuda synchronize
torch.cuda.synchronize()
end = time.time()

# print result size
print("Result size: ", result.size())

print("Time: ", (end - start) / epoch)
print("SpMM time: {} ms".format(spmm_ms_avg))
# glops: 2 * nnz / time
print("GLOPS: ", 2 * sparse_matrix.nnz / ((spmm_ms_avg) / epoch) / 1e6)