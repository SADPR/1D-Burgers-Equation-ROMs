#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Sparse>
#include <vector>
#include <Eigen/Dense>
#include <omp.h>

namespace py = pybind11;

// Function to assemble the diffusion matrix using Eigen's SparseMatrix in parallel
Eigen::SparseMatrix<double> assemble_diffusion_matrix(py::array_t<double> X, py::array_t<double> Y, py::array_t<int> T,
                                                      py::array_t<double> N_values, py::array_t<double> dN_dxi_values,
                                                      py::array_t<double> wgp, py::array_t<double> zgp,
                                                      int n_nodes, int n_elements, int n_local_nodes) {
    // Map the input data
    auto X_buf = X.unchecked<1>();
    auto Y_buf = Y.unchecked<1>();
    auto T_buf = T.unchecked<2>();
    auto N_buf = N_values.unchecked<3>();
    auto dN_dxi_buf = dN_dxi_values.unchecked<4>();
    auto wgp_buf = wgp.unchecked<1>();
    auto zgp_buf = zgp.unchecked<1>();

    // Initialize the sparse matrix in COO format using triplets
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(2 * n_elements * n_local_nodes * n_local_nodes);  // For u_x and u_y components

    // Parallel loop over elements
    #pragma omp parallel
    {
        std::vector<Eigen::Triplet<double>> tripletList_private;
        tripletList_private.reserve(2 * n_local_nodes * n_local_nodes);  // Each thread's triplet list

        #pragma omp for nowait
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

            // Initialize the local diffusion matrix
            Eigen::MatrixXd K_element = Eigen::MatrixXd::Zero(n_local_nodes, n_local_nodes);

            // Loop over Gauss points
            for (int i = 0; i < 2; ++i) {  // ngaus = 2
                for (int j = 0; j < 2; ++j) {  // ngaus = 2
                    double xi = zgp_buf(i);
                    double eta = zgp_buf(j);

                    // Shape function values and derivatives at the Gauss point
                    Eigen::VectorXd N_gp(n_local_nodes);
                    Eigen::MatrixXd dN_dxi_gp(n_local_nodes, 2);
                    for (int k = 0; k < n_local_nodes; ++k) {
                        N_gp(k) = N_buf(i, j, k);
                        dN_dxi_gp(k, 0) = dN_dxi_buf(i, j, k, 0);
                        dN_dxi_gp(k, 1) = dN_dxi_buf(i, j, k, 1);
                    }

                    // Compute the Jacobian matrix
                    Eigen::MatrixXd J(2, 2);
                    J.setZero();
                    for (int k = 0; k < n_local_nodes; ++k) {
                        J(0, 0) += dN_dxi_gp(k, 0) * x_element[k];
                        J(0, 1) += dN_dxi_gp(k, 0) * y_element[k];
                        J(1, 0) += dN_dxi_gp(k, 1) * x_element[k];
                        J(1, 1) += dN_dxi_gp(k, 1) * y_element[k];
                    }

                    double detJ = J.determinant();
                    Eigen::MatrixXd invJ = J.inverse();

                    // Compute the derivative of shape functions wrt physical coordinates
                    Eigen::MatrixXd dN_dx_gp = invJ * dN_dxi_gp.transpose();

                    // Compute the differential volume
                    double dV = wgp_buf(i) * wgp_buf(j) * detJ;

                    // Local diffusion matrix contribution
                    K_element += dV * (dN_dx_gp.transpose() * dN_dx_gp);
                }
            }

            // Assemble the local diffusion matrix into the global sparse matrix
            for (int a = 0; a < n_local_nodes; ++a) {
                int global_a_x = element_nodes[a];
                int global_a_y = element_nodes[a] + n_nodes;

                for (int b = 0; b < n_local_nodes; ++b) {
                    int global_b_x = element_nodes[b];
                    int global_b_y = element_nodes[b] + n_nodes;

                    double value = K_element(a, b);

                    // For u_x component
                    tripletList_private.emplace_back(global_a_x, global_b_x, value);
                    // For u_y component
                    tripletList_private.emplace_back(global_a_y, global_b_y, value);
                }
            }
        }

        #pragma omp critical
        tripletList.insert(tripletList.end(), tripletList_private.begin(), tripletList_private.end());
    }

    // Convert triplet list to sparse matrix
    Eigen::SparseMatrix<double> K_global(2 * n_nodes, 2 * n_nodes);
    K_global.setFromTriplets(tripletList.begin(), tripletList.end());

    return K_global;  // Return sparse matrix directly
}

// Binding the C++ function to Python using Pybind11
PYBIND11_MODULE(diffusion_matrix_parallel, m) {
    m.def("assemble_diffusion_matrix", &assemble_diffusion_matrix, "Assemble sparse diffusion matrix");
}

