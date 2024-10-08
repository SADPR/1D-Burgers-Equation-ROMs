from setuptools import setup, Extension
import pybind11
import os

# Get the include path for pybind11
pybind11_include = pybind11.get_include()

# Define the MKL include and library paths (update these if MKL is installed in a different location)
mkl_include = "/opt/intel/mkl/include"
mkl_lib = "/opt/intel/mkl/lib/intel64"

# Extension definition
ext_modules = [
    Extension(
        "sparse_solver_parallel",  # Name of the generated module
        sources=["sparse_solver_parallel.cpp"],  # C++ source files
        include_dirs=[pybind11_include, "/usr/include/eigen3", mkl_include],  # Include directories for Pybind11, Eigen, and MKL
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],  # Optimization flags
        extra_link_args=[
            f"-L{mkl_lib}",  # Add the MKL library path
            "-lmkl_intel_lp64", "-lmkl_intel_thread", "-lmkl_core",  # Link the MKL libraries
            "-liomp5",  # Intel OpenMP
            "-lpthread", "-lm", "-ldl"  # Standard threading and dynamic linking libraries
        ]
    )
]

# Setup configuration
setup(
    name="sparse_solver_parallel",
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.6.2"],
)
