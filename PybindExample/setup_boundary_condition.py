from setuptools import setup, Extension
import pybind11
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "boundary_conditions_parallel",
        ["boundary_conditions_parallel.cpp"],
        include_dirs=[
            pybind11.get_include(),
            '/usr/include/eigen3',  # Eigen path on most Linux systems
        ],
        extra_compile_args=["-O3"],  # Optimization flag
    ),
]

setup(
    name="boundary_conditions_parallel",
    ext_modules=ext_modules,
    zip_safe=False,
)
