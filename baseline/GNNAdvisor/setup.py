from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='GNNAdvisor_now',
    ext_modules=[
        CUDAExtension(
        name='GNNAdvisor_now', 
        sources=[   
                    'GNNAdvisor.cpp', 
                    'GNNAdvisor_kernel.cu'
                ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })