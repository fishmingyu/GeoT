import glob
import os
import os.path as osp
from itertools import product

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

__version__ = '0.0.1'
URL = ''

WITH_CUDA = True
# if torch.cuda.is_available():
#     WITH_CUDA = CUDA_HOME is not None
suffices = ['cuda', 'cpu'] if WITH_CUDA else ['cpu']
print(f'Building with CUDA: {WITH_CUDA}, ', 'CUDA_HOME:', CUDA_HOME)


def get_extensions():
    extensions = []
    extensions_dir = osp.join('csrc')
    main_files = glob.glob(osp.join(extensions_dir, '*.cpp'))
    main_files = [path for path in main_files]

    define_macros = [('WITH_PYTHON', None)]
    undef_macros = []
    libraries = []
    extra_compile_args = {'cxx': ['-O2']}
    extra_link_args = [
        '-s',
        '-lm',
        '-ldl',
    ]

    if WITH_CUDA:
        define_macros += [('WITH_CUDA', None)]
        nvcc_flags = os.getenv('NVCC_FLAGS', '')
        nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
        nvcc_flags += ['-O2']
        extra_compile_args['nvcc'] = nvcc_flags

    names = ['index_scatter', 'gather_scatter', 'gather_weight_scatter', 'mh_spmm']
    sources = main_files

    for name in names:
        path = osp.join(extensions_dir, 'cpu', f'{name}_cpu.cpp')
        if osp.exists(path):
            sources += [path]

        path = osp.join(extensions_dir, 'cuda', f'{name}_cuda.cu')
        if WITH_CUDA and osp.exists(path):
            sources += [path]

    Extension = CUDAExtension if WITH_CUDA else CppExtension
    extension = Extension(
        f'geot._C',
        sources,
        include_dirs=[extensions_dir],
        define_macros=define_macros,
        undef_macros=undef_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
    )
    extensions += [extension]
    return extensions


install_requires = [
    'scipy',
    'torch>=2.1.0',
    'torch_scatter',
    'ogb',
    'torch_geometric',
    'rdflib',
    'h5py',
    'pandas'
]

test_requires = [
    'pytest',
]

setup(
    name='geot',
    version=__version__,
    description=(
        'GeoT: Tensor Centric Library for Graph Neural Network via Efficient Segment Reduction on GPU'),
    author='Zhongming Yu, Genghan Zhang',
    author_email='zhy025@ucsd.edu',
    url=URL,
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    keywords=[
        'pytorch',
        'sparse',
        'autograd',
    ],
    python_requires='>=3.7',
    install_requires=install_requires,
    extras_require={
        'test': test_requires,
    },
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext':
        BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    },
    packages=find_packages(),
    include_package_data=True,
)
