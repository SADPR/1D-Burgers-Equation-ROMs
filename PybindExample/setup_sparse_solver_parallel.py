from setuptools import setup, Extension
import pybind11
import os

# Get the include path for pybind11
pybind11_include = pybind11.get_include()

# Extension definition
ext_modules = [
    Extension(
        "sparse_solver_parallel",  # Name of the generated module
        sources=["sparse_solver_parallel.cpp"],  # C++ source files
        include_dirs=[pybind11_include, "/usr/include/eigen3"],  # Include directories for Pybind11 and Eigen
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],  # Optimization flags
    )
]

# Setup configuration
setup(
    name="sparse_solver_parallel",
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.6.2"],
)
