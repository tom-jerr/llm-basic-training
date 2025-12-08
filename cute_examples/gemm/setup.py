from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Assuming we are running setup.py from gemm_binding directory
cutlass_dir = os.path.abspath("../../cutlass")
include_dirs = [
    os.path.join(cutlass_dir, "include"),
    os.path.join(cutlass_dir, "tools/util/include"),
]

setup(
    name='gemm_custom',
    ext_modules=[
        CUDAExtension(
            name='gemm_custom',
            sources=['gemm_binding.cpp', 'gemm_kernel.cu'],
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '-std=c++17', '--expt-relaxed-constexpr', '-arch=sm_80']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
