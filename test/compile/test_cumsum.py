import geot.match_replace
import torch
import torch.fx
import geot

def coo_to_csr(nrow, coo_row):
    # torch._check_is_size(nrow)
    csr_row = torch.zeros(nrow + 1).to("cuda")
    bin = torch.bincount(coo_row, minlength=nrow)
    # torch._dynamo.mark_dynamic(bin, 0)
    csr_row[1:] = torch.cumsum(bin, 0)
    csr_row = csr_row.int()
    return csr_row

def coo_to_csr_sequential(coo_row):
    nrow = coo_row.max() + 1
    csr_row = torch.zeros(nrow + 1).to("cuda")
    for i in range(nrow):
        csr_row[i+1] = csr_row[i] + (coo_row == i).sum()
    csr_row = csr_row.int()
    return csr_row

class CumSum(torch.nn.Module):
    def forward(self, nrow, x):
        bin = torch.bincount(x, minlength=nrow)
        return torch.cumsum(bin, 0)

class CumSumSeq(torch.nn.Module):
    def forward(self, nrow, x):
        csr_row = torch.zeros(nrow + 1).to("cuda")
        csr_row_list = list(csr_row)
        for i in range(nrow):
            csr_row_list[i+1] = csr_row_list[i] + (x == i).sum()
        csr_row = torch.tensor(csr_row_list).to("cuda")
        csr_row = csr_row.int()
        return csr_row

nnz = 100
nrow = 10
input = torch.randint(0, nrow, (nnz,)).to("cuda")
# compiled = torch.compile(coo_to_csr)
output = coo_to_csr(nrow, input)

compiled_seq = torch.compile(coo_to_csr_sequential)
out_seq = compiled_seq(input)

# model = CumSum()
# exported = torch.export.export(model, (nrow, input))
# print(f'After:{exported.graph_module.code}')
# out_exported = exported.module()(nrow, input)

# diff = torch.abs(out_exported - output).max()
# print(f'max difference of compiled: {diff}')

model_seq = CumSumSeq()
exported_seq = torch.export.export(model_seq, (nrow, input))
print(f'After:{exported_seq.graph_module.code}')
out_exported_seq = exported_seq.module()(nrow, input)

diff = torch.abs(out_exported_seq - out_seq).max()
print(f'max difference of sequential: {diff}')
