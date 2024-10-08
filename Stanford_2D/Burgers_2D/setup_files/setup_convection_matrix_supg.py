from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext  # Use setup_helpers from pybind11
import pybind11

ext_modules = [
    Pybind11Extension(
        "convection_matrix_supg_parallel",
        sources=["cpp_files/convection_matrix_supg_parallel.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "/usr/include/eigen3",  # Adjust the path to Eigen if needed
        ],
        extra_compile_args=["-fopenmp", "-O3"],  # Enable OpenMP and optimize
        extra_link_args=["-fopenmp"],  # Link with OpenMP
        language="c++",
    ),
]

setup(
    name="convection_matrix_supg_parallel",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},  # Use build_ext from pybind11.setup_helpers
)



