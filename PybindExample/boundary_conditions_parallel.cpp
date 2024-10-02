#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Sparse>
#include <vector>

namespace py = pybind11;

// Function to apply boundary conditions to matrix A and vector R
std::tuple<Eigen::SparseMatrix<double>, Eigen::VectorXd> apply_boundary_conditions(
    Eigen::SparseMatrix<double>& A, Eigen::VectorXd& R, const std::vector<int>& boundary_dofs) {

    // Make a copy of A to work with non-const elements (as .coeffRef() may not directly work with SparseMatrix)
    Eigen::SparseMatrix<double> A_modified = A;

    // Loop over each boundary degree of freedom (DOF)
    for (const int& dof : boundary_dofs) {
        // Set entire row of A[dof, :] to zero
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_modified, dof); it; ++it) {
            it.valueRef() = 0.0;
        }
        // Set diagonal entry A[dof, dof] to 1.0
        A_modified.coeffRef(dof, dof) = 1.0;

        // Set the corresponding entry in the residual vector R to 0.0
        R[dof] = 0.0;
    }

    // Return both modified matrix A and vector R as a tuple
    return std::make_tuple(A_modified, R);
}

// Pybind11 module definition
PYBIND11_MODULE(boundary_conditions_parallel, m) {
    m.doc() = "Module for applying boundary conditions in parallel";  // Optional module docstring
    m.def("apply_boundary_conditions", &apply_boundary_conditions,
          "Applies boundary conditions to the sparse matrix A and residual vector R and returns modified A and R",
          py::arg("A").noconvert(), py::arg("R").noconvert(), py::arg("boundary_dofs"));
}





