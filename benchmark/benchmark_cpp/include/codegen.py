# Generate C++ code for the benchmark
# Template for the C++ code
# 4 configurations: int NPerThread, int NThreadX, int NnzPerThread, int NnzThreadY
def generate_sr_subcode(NPerThread, NThreadX, NnzPerThread, NnzThreadY):
    function_name = f"segscan_sr_test"
    value_type = f"ValueType"
    template_list = f"<{value_type}, {NPerThread}, {NThreadX}, {NnzPerThread}, {NnzThreadY}>"
    argument_list = f"(nnz, N, keys, index, src, dst)"
    time_code = f"time = " + f"{function_name}{template_list}" + argument_list + ";\n"
    gflops_code = f"    gflops = nnz * N / time / 1e6;\n"
    profile_code = f"    out_file << data_name << \",\" << N << \",\" << {NPerThread} << \",\" << {NThreadX} << \",\" << {NnzPerThread} << \",\" << {NnzThreadY} << \",\" << time <<  \",\" << gflops << std::endl;\n"
    return time_code + gflops_code + profile_code

# 5 configurations: int NPerThread, int NThreadY, int NnzPerThread, int RNum, int RSync
def generate_pr_subcode(NPerThread, NThreadY, NnzPerThread, RNum, RSync):
    function_name = f"segscan_pr_test"
    value_type = f"ValueType"
    template_list = f"<{value_type}, {NPerThread}, {NThreadY}, {NnzPerThread}, {RNum}, {RSync}>"
    argument_list = f"(nnz, N, keys, index, src, dst)"
    time_code = f"time = " + f"{function_name}{template_list}" + argument_list + ";\n"
    gflops_code = f"    gflops = nnz * N / time / 1e6;\n"
    profile_code = f"    out_file << data_name << \",\"<< N << \",\" << {NPerThread} << \",\" << {NThreadY} << \",\" << {NnzPerThread} << \",\" << {RNum} << \",\" << {RSync} << \",\" << time <<  \",\" << gflops << std::endl;\n"
    return time_code + gflops_code + profile_code

def generate_sr_tune():
    function_head = r"""
template <typename ValueType>
void segscan_sr_tune(std::ofstream &out_file, char *data_name, int nnz, int N, int keys, util::RamArray<Index> &index,
                     util::RamArray<DType> &src, util::RamArray<DType> &dst) {
    float time = 0;
    float gflops = 0;
"""
    NPerThread_list = [1, 2]
    NThreadX_list = [8, 16, 32, 64]
    NnzPerThread_list = [4, 8, 16, 32]
    NnzThread_list = [2, 4, 8]

    function_body = ""
    for NPerThread in NPerThread_list:
        for NThreadX in NThreadX_list:
            for NnzPerThread in NnzPerThread_list:
                for NnzThreadY in NnzThread_list:
                    # add tap in the beginning of the line
                    # only (consider NThreadX * NnzThreadY >= 64)
                    if NThreadX * NnzThreadY >= 64:
                        function_body += "    " + generate_sr_subcode(NPerThread, NThreadX, NnzPerThread, NnzThreadY)
                    
    function_tail = r"}"

    return function_head + function_body + function_tail


def generate_pr_tune():
    function_head = r"""
template <typename ValueType>
void segscan_pr_tune(std::ofstream &out_file, char *data_name, int nnz, int N, int keys, util::RamArray<Index> &index,
                     util::RamArray<DType> &src, util::RamArray<DType> &dst) {
    float time = 0;
    float gflops = 0;
"""
    NPerThread_list = [1, 2, 4]
    NThreadY_list = [1, 2, 4]
    NnzPerThread_list = [1, 2, 4]
    RNum_list = [1, 2, 4, 8]
    RSync_list = [8, 16, 32]

    function_body = ""
    for NPerThread in NPerThread_list:
        for NThreadY in NThreadY_list:
            for NnzPerThread in NnzPerThread_list:
                for RNum in RNum_list:
                    for RSync in RSync_list:
                        # add tap in the beginning of the line
                        function_body += "    " + generate_pr_subcode(NPerThread, NThreadY, NnzPerThread, RNum, RSync)

    function_tail = r"}"

    return function_head + function_body + function_tail

def generate_code():
    code_header = r"""
#include "util.cuh"
"""
    sr_code = generate_sr_tune()
    pr_code = generate_pr_tune()
    return code_header + sr_code + '\n' + pr_code


if __name__ == "__main__":
    # create a hpp file and write the code into it
    with open("seg_func.hpp", "w") as f:
        f.write(generate_code())
