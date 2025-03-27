import torch
import cusparse_spmm_csr

col = torch.tensor([0, 2, 3, 1, 0, 2, 3, 1, 3],dtype=torch.int32)
row = torch.tensor([0, 3, 4, 7, 9],dtype=torch.int32)
a_value = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9],dtype=torch.half)
b_vlaue = torch.tensor([1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12],dtype=torch.half)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
row=row.to(device)
col=col.to(device)
a_value=a_value.to(device)
b_vlaue=b_vlaue.to(device)
print(row)
print(col)
print(a_value)
print(b_vlaue)
output=cusparse_spmm_csr.cuSPARSE_SPMM_CSR(row, col, a_value, b_vlaue, 4, 3, 9)
print(output)
