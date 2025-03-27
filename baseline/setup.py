from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='MBaseline_kernel',
    ext_modules=[
       CUDAExtension(
            #v2 without gemm in GNNAdisor
            name='GNNAdvisor_kernel', 
            sources=[
            './GNNAdvisor/GNNAdvisor_kernel.cu',
            './GNNAdvisor/GNNAdvisor.cpp',
            ]
         ),
       CUDAExtension(
            name='TCGNN', 
            sources=[
            './TCGNN/TCGNN_kernel.cu',
            './TCGNN/TCGNN.cpp',
            ]
         ) ,
        CUDAExtension(
            name='TCGNN_kernel', 
            sources=[
            './TCGNN_kernel/TCGNN_kernel.cu',
            './TCGNN_kernel/TCGNN.cpp',
            ]
         ) ,
       CUDAExtension(
            name='GESpMM_kernel', 
            sources=[
            './GESpMM/gespmmkernel.cu', 
            './GESpMM/gespmm.cpp',
            ]
         ) ,
       CUDAExtension(
            name='cuSPARSE_kernel', 
            sources=[
            './cuSPARSE/spmm_csr_kernel.cu',
            './cuSPARSE/spmm_csr.cpp',
            ]
         ) ,
       CUDAExtension(
            name = 'DTCSpMM', 
            sources =[
            './DTC-SpMM/DTCSpMM.cpp',
            './DTC-SpMM/DTCSpMM_kernel.cu',
            ]
         ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


