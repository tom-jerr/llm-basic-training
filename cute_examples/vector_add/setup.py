from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Assuming setup.py is in vector_add/
cutlass_dir = os.path.abspath("../cutlass")
include_dirs = [
    os.path.join(cutlass_dir, "include"),
    os.path.join(cutlass_dir, "tools/util/include")
]

setup(
    name='vector_add_cuda',
    ext_modules=[
        CUDAExtension('vector_add_cuda', [
            'vector_add_bind.cu',
        ],
        include_dirs=include_dirs,
        extra_compile_args={'cxx': ['-std=c++17'],
                            'nvcc': ['-std=c++17', '-O3', '--expt-relaxed-constexpr', '-arch=sm_80']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
