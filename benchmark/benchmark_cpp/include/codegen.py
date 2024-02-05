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
    return time_code + gflops_code

def generate_profile_output(NPerThread, NThreadX, NnzPerThread, NnzThreadY):
    code = f"out_file << data_name << \",\" << {NPerThread} << \",\" << {NThreadX} << \",\" << {NnzPerThread} << \",\" << {NnzThreadY} << \",\" << time <<  \",\" << gflops << std::endl;\n"
    return code

def generate_sr_tune():
    function_head = r"""
template <typename ValueType>
void segscan_sr_tune(std::ofstream &out_file, char *data_name, int nnz, int N, int keys, util::RamArray<Index> &index,
                     util::RamArray<DType> &src, util::RamArray<DType> &dst) {
    float time = 0;
    float gflops = 0;
"""
    NPerThread_list = [1, 2, 4]
    NThreadX_list = [8, 16, 32, 64]
    NnzPerThread_list = [4, 8, 16, 32, 64]
    NnzThread = [1, 2, 4, 8]

    function_body = ""
    for NPerThread in NPerThread_list:
        for NThreadX in NThreadX_list:
            for NnzPerThread in NnzPerThread_list:
                for NnzThreadY in NnzThread:
                    # add tap in the beginning of the line
                    function_body += "    " + generate_sr_subcode(NPerThread, NThreadX, NnzPerThread, NnzThreadY)
                    function_body += "    " + generate_profile_output(NPerThread, NThreadX, NnzPerThread, NnzThreadY)
                    
    function_tail = r"}"

    return function_head + function_body + function_tail

def generate_code():
    code_header = r"""
#include "util.cuh"
"""
    return code_header + generate_sr_tune()


if __name__ == "__main__":
    # create a hpp file and write the code into it
    with open("seg_func.hpp", "w") as f:
        f.write(generate_code())
