from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='GESPMM',
    ext_modules=[
       CUDAExtension(
            name='GESPMM', 
            sources=[
            'gespmmkernel.cu',
            'gespmm.cpp'
            ]
            # extra_compile_args=['-O3', '-fopenmp', '-mcx16'],
            # libraries=["numa", "tcmalloc_minimal"]
         ) 
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


