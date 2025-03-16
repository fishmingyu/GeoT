import numpy as np
import torch
import FS_Block
row = torch.tensor([0, 6, 7,  8, 10, 10, 11, 13, 15, 18, 19, 19, 19, 20, 20, 20, 20, 24,
        25, 26, 28, 28, 29, 31, 33, 36, 37, 37, 37, 38, 38, 38, 38],dtype=torch.int32)
col = torch.tensor([1,2,3,11,16,21, 1,1,2,24,25,16,22,0,27,0,4,25,9,27,1,3,11,21,1,1,2,24,25,16,22,0,27,0,4,25,9,27],dtype=torch.int32)
value1=col.float()
# value1=value
rowTensor, colTensor, valueTensor = FS_Block.blockProcess_tf32(row,col,value1,8,8)
# rowTensor, colTensor, valueTensor, window, atomic = FS_Block.blockProcess_fp16_balance(row,col,value1,8,4,2)
# rowTensor, colTensor, valueTensor = FS_Block.blockProcess_output(row,col,8)
# rowTensor, colTensor, valueTensor16, valueTensor8, valueTensor4 = FS_Block.blockProcess_csr(row,col)
print(rowTensor)
print(colTensor)
print(valueTensor)
# print valueTensor size
print(valueTensor.size())
# print(window)
# print(atomic)
# print(valueTensor16)
# print(valueTensor8)
# print(valueTensor4)
# print(valueTensor_templete)
