#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Sparse>
#include <vector>

namespace py = pybind11;

// Function to apply boundary conditions to matrix A and vector R
std::tuple<Eigen::SparseMatrix<double>, Eigen::VectorXd> apply_boundary_conditions(
    Eigen::SparseMatrix<double>& A,          // 1
    Eigen::VectorXd& R,                      // 2
    const Eigen::VectorXd& U_current,        // 3
    const std::vector<int>& boundary_dofs,   // 4
    double mu1                               // 5
) {
    Eigen::SparseMatrix<double> A_modified = A;

    for (const int& dof : boundary_dofs) {
        // Set entire row of A[dof, :] to zero
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_modified, dof); it; ++it) {
            it.valueRef() = 0.0;
        }
        // Set diagonal entry A[dof, dof] to 1.0
        A_modified.coeffRef(dof, dof) = 1.0;

        // Set the corresponding entry in the residual vector R
        R[dof] = U_current[dof] - mu1;
    }

    return std::make_tuple(A_modified, R);
}

// Pybind11 module definition
PYBIND11_MODULE(boundary_conditions_parallel, m) {
    m.doc() = "Module for applying boundary conditions in parallel";  // Optional module docstring
    m.def("apply_boundary_conditions", &apply_boundary_conditions,
          "Applies boundary conditions to the sparse matrix A and residual vector R and returns modified A and R",
          py::arg("A").noconvert(),         // 1
          py::arg("R").noconvert(),         // 2
          py::arg("U_current"),             // 3
          py::arg("boundary_dofs"),         // 4
          py::arg("mu1")                    // 5
    );
}
