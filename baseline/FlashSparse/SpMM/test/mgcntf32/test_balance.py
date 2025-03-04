import numpy
import torch
import Libra5Block
import Libra5BenchmarkGCN
from scipy.sparse import *
import FS_Block
import FS_SpMM
def check(row_pointers1, column_index1, dd, rhs, n) :
    row_pointers1 = row_pointers1[:n+1]
    dd = dd.numpy()
    value = []
    for i in range(len(row_pointers1) - 1):
        for j in range(row_pointers1[i], row_pointers1[i+1]):
            value.append(dd[i]*dd[column_index1[j]])
    # n = row_pointers1.size(0)-1
    sparse_matrix = csr_matrix((value, column_index1.numpy(), row_pointers1.numpy()), shape=(n, n))
    result = sparse_matrix.dot(rhs.numpy())
    return result

row = torch.tensor([0, 6, 7,  8, 10, 10, 11, 13, 15, 18, 19, 19, 19, 20, 20, 20, 20, 24,
        25, 26, 28, 28, 29, 31, 33, 36, 37, 37, 37, 38, 38, 38, 38],dtype=torch.int32)
col = torch.tensor([1,2,3,11,16,21, 1,1,2,24,25,16,22,0,27,0,4,25,9,27,1,3,11,21,1,1,2,24,25,16,22,0,27,0,4,25,9,27],dtype=torch.int32)

dd =  torch.ones_like(col).float()
row_pointers, column_index, degrees, t_window_rowTensor,t_atomicTensor=FS_Block.blockProcess_tf32_balance(row, col, dd, 8, 4, 2)

print(row_pointers)
print(column_index)
print(t_window_rowTensor)
print(t_atomicTensor)
# print(degrees)
print()
dimN =32
#rhs = torch.ones((30, dimN), dtype=torch.float32)
rhs = torch.randint(low=1, high=3, size=(30, dimN))
rhs = rhs.float()

result, spmm_ms_avg =  FS_SpMM.forward_tf32_balance(   
                        row_pointers, 
                        column_index, 
                        degrees, 
                        t_window_rowTensor,
                        t_atomicTensor,
                        rhs, 
                        32, 
                        rhs.size(1), 
                        30, 1)

res = check(row,col,dd,rhs,30)
print(result)
print(res)


for i in range(30):
    if (result[i][0] - res[i][0]) != 0 :
            print("No")
            exit(0)
    if (result[i][dimN-1] - res[i][dimN-1]) != 0 :
            print("No")
            exit(0)
        
print("PASS")
    