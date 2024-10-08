#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

namespace py = pybind11;

// Function to solve the linear system A * delta_U = -R using Eigen's SparseLU
Eigen::VectorXd solve_sparse_system(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& R) {
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Decomposition failed");
    }

    // Solve for delta_U
    Eigen::VectorXd delta_U = solver.solve(-R);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Solving failed");
    }

    return delta_U;
}

// Pybind11 module definition
PYBIND11_MODULE(sparse_solver_parallel, m) {
    m.doc() = "Module for solving sparse linear systems using Eigen";  // Optional module docstring
    m.def("solve_sparse_system", &solve_sparse_system,
          "Solves the sparse linear system A * delta_U = -R using Eigen's SparseLU",
          py::arg("A"), py::arg("R"));
}
