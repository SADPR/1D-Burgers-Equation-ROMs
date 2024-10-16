from setuptools import setup, Extension
import pybind11
import subprocess
import os

# Source the oneAPI setvars.sh script
def source_oneapi():
    oneapi_script = "/opt/intel/oneapi/setvars.sh"
    if os.path.exists(oneapi_script):
        subprocess.run(f"source {oneapi_script}", shell=True, executable="/bin/bash")
    else:
        print(f"Warning: {oneapi_script} not found. Make sure you have the oneAPI toolkit installed.")

# Call the function to source the environment variables
source_oneapi()

# Get the include path for pybind11
pybind11_include = pybind11.get_include()

# Define the MKL include and library paths
mkl_include = "/opt/intel/oneapi/mkl/latest/include"  # Path to MKL headers
mkl_lib = "/opt/intel/oneapi/mkl/latest/lib/intel64"  # Path to MKL libraries

# Eigen include directory
eigen_include = "/usr/include/eigen3"

# Extension definition
ext_modules = [
    Extension(
        "mkl_sparse_dense_operations",  # Name of the generated module
        sources=["cpp_files/mkl_sparse_dense_operations.cpp"],  # C++ source file
        include_dirs=[pybind11_include, eigen_include, mkl_include],  # Include directories for Pybind11, Eigen, and MKL
        language="c++",
        extra_compile_args=["-O3", "-std=c++17","-fopenmp"],  # Optimization flags
        extra_link_args=[
            f"-L{mkl_lib}",  # Add the MKL library path
            "-lmkl_intel_lp64", "-lmkl_intel_thread", "-lmkl_core",  # Link the MKL libraries
            "-liomp5",  # Intel OpenMP
            "-lpthread", "-lm", "-ldl",  # Standard threading and dynamic linking libraries
            "-fopenmp"
        ]
    )
]

# Setup configuration
setup(
    name="mkl_sparse_dense_operations",
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.6.2"],
)

