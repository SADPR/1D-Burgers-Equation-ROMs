from setuptools import setup, Extension
import pybind11

# Get the include path for pybind11
pybind11_include = pybind11.get_include()

# Define the MKL include and library paths (update these if MKL is installed in a different location)
mkl_include = "/opt/intel/mkl/include"  # Path to MKL headers
mkl_lib = "/opt/intel/mkl/lib/intel64"  # Path to MKL libraries

# Eigen include directory
eigen_include = "/usr/include/eigen3"

# Extension definition
ext_modules = [
    Extension(
        "mkl_sparse_dense_operations",  # Name of the generated module
        sources=["cpp_files/mkl_sparse_dense_operations.cpp"],  # C++ source file
        include_dirs=[pybind11_include, eigen_include, mkl_include],  # Include directories for Pybind11, Eigen, and MKL
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
    name="mkl_sparse_dense_operations",
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.6.2"],
)
