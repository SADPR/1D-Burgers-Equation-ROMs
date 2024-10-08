#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Sparse>
#include <Eigen/PardisoSupport>  // Use Intel MKL's Pardiso solver

namespace py = pybind11;

// Function to solve the linear system A * delta_U = -R using Intel MKL's Pardiso solver
Eigen::VectorXd solve_sparse_system(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& R) {
    Eigen::PardisoLU<Eigen::SparseMatrix<double>> solver;  // Pardiso LU from MKL
    solver.compute(A);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Decomposition failed");
    }

    // Create a negated version of R explicitly
    Eigen::VectorXd neg_R = -R;

    // Solve for delta_U
    Eigen::VectorXd delta_U = solver.solve(neg_R);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Solving failed");
    }

    return delta_U;
}

// Pybind11 module definition
PYBIND11_MODULE(sparse_solver_parallel, m) {
    m.doc() = "Module for solving sparse linear systems using Intel MKL's Pardiso solver";  // Optional module docstring
    m.def("solve_sparse_system", &solve_sparse_system,
          "Solves the sparse linear system A * delta_U = -R using Intel MKL's PardisoLU",
          py::arg("A"), py::arg("R"));
}
