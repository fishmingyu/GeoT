from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


setup(
    name='FlashSparse_kernel',
    # ext_modules=[module],
    ext_modules=[
        CppExtension(
            name='FS_Block', 
            sources=[
            './block_format.cpp'
            ]
         ) ,
         CUDAExtension(
            name='FS_SpMM', 
            sources=[
            './SpMM/src/benchmark.cpp',
            './SpMM/src/spmmKernel.cu',
            ]
         ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

