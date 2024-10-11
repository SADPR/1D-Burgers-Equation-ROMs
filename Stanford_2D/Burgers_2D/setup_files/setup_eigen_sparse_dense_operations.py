from setuptools import setup, Extension
import pybind11

# Get the include path for pybind11
pybind11_include = pybind11.get_include()

# Extension definition
ext_modules = [
    Extension(
        "eigen_sparse_dense_operations",  # Name of the generated module
        sources=["cpp_files/eigen_sparse_dense_operations.cpp"],  # C++ source file
        include_dirs=[pybind11_include, "/usr/include/eigen3"],  # Include directories for Pybind11 and Eigen
        language="c++",
        extra_compile_args=["-O3", "-std=c++17", "-fopenmp"],  # Enable OpenMP
        extra_link_args=["-fopenmp"]  # Link OpenMP
    )
]

# Setup configuration
setup(
    name="eigen_sparse_dense_operations",
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.6.2"],
)

