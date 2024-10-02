from setuptools import setup, Extension
import pybind11

# Define the extension module
ext_modules = [
    Extension(
        "example",  # Name of the module
        ["example.cpp"],  # Source files
        include_dirs=[pybind11.get_include()],  # Pybind11 include directory
        language="c++"
    ),
]

# Setup the module
setup(
    name="example",
    ext_modules=ext_modules,
    zip_safe=False,
)
