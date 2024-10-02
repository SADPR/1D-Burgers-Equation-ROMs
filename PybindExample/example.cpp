#include <pybind11/pybind11.h>

// Simple function to add two numbers
int add(int a, int b) {
    return a + b;
}

// Pybind11 module definition
PYBIND11_MODULE(example, m) {
    m.def("add", &add, "A function that adds two numbers");
}
