#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Sparse>
#include <vector>
#include <Eigen/Dense>

namespace py = pybind11;

// Function to assemble the mass matrix using Eigen's SparseMatrix and return it in a sparse format
Eigen::SparseMatrix<double> assemble_mass_matrix(py::array_t<double> X, py::array_t<double> Y, py::array_t<int> T,
                                                 py::array_t<double> N_values, py::array_t<double> wgp, py::array_t<double> zgp,
                                                 int n_nodes, int n_elements, int n_local_nodes) {
    // Map the input data
    auto X_buf = X.unchecked<1>();
    auto Y_buf = Y.unchecked<1>();
    auto T_buf = T.unchecked<2>();
    auto N_buf = N_values.unchecked<3>();
    auto wgp_buf = wgp.unchecked<1>();
    auto zgp_buf = zgp.unchecked<1>();

    // Initialize the sparse matrix in COO format using triplets
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(2 * n_elements * n_local_nodes * n_local_nodes);  // For u_x and u_y components

    // Loop over elements
    for (int elem = 0; elem < n_elements; ++elem) {
        // Extract the nodes for the current element
        std::array<int, 4> element_nodes;
        for (int i = 0; i < n_local_nodes; ++i) {
            element_nodes[i] = T_buf(elem, i) - 1;  // Convert to zero-based indexing
        }

        // Extract the coordinates of the element's nodes
        std::array<double, 4> x_element, y_element;
        for (int i = 0; i < n_local_nodes; ++i) {
            x_element[i] = X_buf(element_nodes[i]);
            y_element[i] = Y_buf(element_nodes[i]);
        }

        // Initialize the local mass matrix
        Eigen::MatrixXd M_element = Eigen::MatrixXd::Zero(n_local_nodes, n_local_nodes);

        // Loop over Gauss points
        for (int i = 0; i < 2; ++i) {  // ngaus = 2
            for (int j = 0; j < 2; ++j) {  // ngaus = 2
                double xi = zgp_buf(i);
                double eta = zgp_buf(j);

                // Shape function values at the Gauss point
                Eigen::VectorXd N_gp(n_local_nodes);
                for (int k = 0; k < n_local_nodes; ++k) {
                    N_gp(k) = N_buf(i, j, k);
                }

                // Compute the Jacobian matrix
                Eigen::MatrixXd dN_dxi_gp(4, 2);
                dN_dxi_gp(0, 0) = -(1 - eta) / 4.0; dN_dxi_gp(0, 1) = -(1 - xi) / 4.0;
                dN_dxi_gp(1, 0) = (1 - eta) / 4.0; dN_dxi_gp(1, 1) = -(1 + xi) / 4.0;
                dN_dxi_gp(2, 0) = (1 + eta) / 4.0; dN_dxi_gp(2, 1) = (1 + xi) / 4.0;
                dN_dxi_gp(3, 0) = -(1 + eta) / 4.0; dN_dxi_gp(3, 1) = (1 - xi) / 4.0;

                Eigen::MatrixXd J(2, 2);
                J.setZero();
                for (int k = 0; k < n_local_nodes; ++k) {
                    J(0, 0) += dN_dxi_gp(k, 0) * x_element[k];
                    J(0, 1) += dN_dxi_gp(k, 0) * y_element[k];
                    J(1, 0) += dN_dxi_gp(k, 1) * x_element[k];
                    J(1, 1) += dN_dxi_gp(k, 1) * y_element[k];
                }

                double detJ = J.determinant();
                double dV = wgp_buf(i) * wgp_buf(j) * detJ;

                // Local mass matrix contribution
                M_element += N_gp * N_gp.transpose() * dV;
            }
        }

        // Assemble the local mass matrix into the global sparse matrix
        for (int a = 0; a < n_local_nodes; ++a) {
            int global_a_x = element_nodes[a];
            int global_a_y = element_nodes[a] + n_nodes;

            for (int b = 0; b < n_local_nodes; ++b) {
                int global_b_x = element_nodes[b];
                int global_b_y = element_nodes[b] + n_nodes;

                double value = M_element(a, b);

                // For u_x component
                tripletList.emplace_back(global_a_x, global_b_x, value);
                // For u_y component
                tripletList.emplace_back(global_a_y, global_b_y, value);
            }
        }
    }

    // Convert triplet list to sparse matrix
    Eigen::SparseMatrix<double> M_global(2 * n_nodes, 2 * n_nodes);
    M_global.setFromTriplets(tripletList.begin(), tripletList.end());

    return M_global;  // Return sparse matrix directly
}

// Binding the C++ function to Python using Pybind11
PYBIND11_MODULE(mass_matrix, m) {
    m.def("assemble_mass_matrix", &assemble_mass_matrix, "Assemble sparse mass matrix");
}


