# File: setup.py
from setuptools import setup, Extension
import pybind11
import os

# Get the include path for pybind11
pybind11_include = pybind11.get_include()

# Extension definition
ext_modules = [
    Extension(
        "svd_solver",  # Name of the generated module
        sources=["cpp_files/svd_solver.cpp"],  # C++ source file
        include_dirs=[pybind11_include, "/usr/include/eigen3"],  # Include directories for Pybind11 and Eigen
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],  # Optimization flags
    )
]

# Setup configuration
setup(
    name="svd_solver",
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.6.2"],
)
