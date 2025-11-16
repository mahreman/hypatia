"""
Setup script for Hypatia CUDA extensions
Builds the fused Linear+ReLU CUDA kernel
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='hypatia_linear_relu_cuda',
    ext_modules=[
        CUDAExtension(
            name='hypatia_core._linear_relu_cuda',
            sources=[
                os.path.join(current_dir, 'linear_relu.cpp'),
                os.path.join(current_dir, 'linear_relu_cuda.cu'),
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',
                    # Add compute capability flags as needed
                    # '-gencode', 'arch=compute_75,code=sm_75',  # Turing
                    # '-gencode', 'arch=compute_80,code=sm_80',  # Ampere
                    # '-gencode', 'arch=compute_86,code=sm_86',  # Ampere (RTX 30xx)
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
