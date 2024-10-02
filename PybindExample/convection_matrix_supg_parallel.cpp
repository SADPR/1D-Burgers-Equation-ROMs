#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Sparse>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <omp.h>  // Include OpenMP header

namespace py = pybind11;

// Function to assemble the convection matrix using Eigen's SparseMatrix and return it in a sparse format
Eigen::SparseMatrix<double> assemble_convection_matrix_supg(py::array_t<double> X, py::array_t<double> Y, py::array_t<int> T,
                                                           py::array_t<double> N_values, py::array_t<double> dN_values, 
                                                           py::array_t<double> U_n, py::array_t<double> wgp, py::array_t<double> zgp,
                                                           int n_nodes, int n_elements, int n_local_nodes) {
    // Map the input data
    auto X_buf = X.unchecked<1>();
    auto Y_buf = Y.unchecked<1>();
    auto T_buf = T.unchecked<2>();
    auto N_buf = N_values.unchecked<3>();
    auto dN_buf = dN_values.unchecked<4>();  // This is now a 4D array
    auto U_n_buf = U_n.unchecked<2>();
    auto wgp_buf = wgp.unchecked<1>();
    auto zgp_buf = zgp.unchecked<1>();

    // Initialize the sparse matrix in COO format using triplets
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(2 * n_elements * n_local_nodes * n_local_nodes);  // For u_x and u_y components

    // Parallelize the outer loop using OpenMP
    #pragma omp parallel
    {
        std::vector<Eigen::Triplet<double>> local_tripletList;  // Each thread will have its own local triplet list
        local_tripletList.reserve(2 * n_elements * n_local_nodes * n_local_nodes / omp_get_num_threads());

        // Loop over elements in parallel
        #pragma omp for
        for (int elem = 0; elem < n_elements; ++elem) {
            // Extract the nodes for the current element
            std::array<int, 4> element_nodes;
            for (int i = 0; i < n_local_nodes; ++i) {
                element_nodes[i] = T_buf(elem, i) - 1;  // Convert to zero-based indexing
            }

            // Extract the coordinates of the element's nodes and velocity field values
            std::array<double, 4> x_element, y_element, u_element, v_element;
            for (int i = 0; i < n_local_nodes; ++i) {
                x_element[i] = X_buf(element_nodes[i]);
                y_element[i] = Y_buf(element_nodes[i]);
                u_element[i] = U_n_buf(element_nodes[i], 0);  // u_x component
                v_element[i] = U_n_buf(element_nodes[i], 1);  // u_y component
            }

            // Convert arrays to Eigen vectors
            Eigen::VectorXd u_element_eigen = Eigen::Map<Eigen::VectorXd>(u_element.data(), u_element.size());
            Eigen::VectorXd v_element_eigen = Eigen::Map<Eigen::VectorXd>(v_element.data(), v_element.size());

            // Initialize the local convection and supg matrices
            Eigen::MatrixXd C_element = Eigen::MatrixXd::Zero(2 * n_local_nodes, 2 * n_local_nodes);
            Eigen::MatrixXd C_supg_element = Eigen::MatrixXd::Zero(2 * n_local_nodes, 2 * n_local_nodes);

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

                    // Shape function derivatives at the Gauss point
                    Eigen::MatrixXd dN_dxi_gp(4, 2);
                    for (int k = 0; k < n_local_nodes; ++k) {
                        dN_dxi_gp(k, 0) = dN_buf(i, j, k, 0);  // Corrected to use 4D indexing
                        dN_dxi_gp(k, 1) = dN_buf(i, j, k, 1);
                    }

                    // Compute Jacobian matrix
                    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2, 2);
                    for (int k = 0; k < n_local_nodes; ++k) {
                        J(0, 0) += dN_dxi_gp(k, 0) * x_element[k];
                        J(0, 1) += dN_dxi_gp(k, 0) * y_element[k];
                        J(1, 0) += dN_dxi_gp(k, 1) * x_element[k];
                        J(1, 1) += dN_dxi_gp(k, 1) * y_element[k];
                    }

                    double detJ = J.determinant();
                    Eigen::MatrixXd invJ = J.inverse();
                    Eigen::MatrixXd dN_dx_gp = invJ * dN_dxi_gp.transpose();

                    // Compute velocities at Gauss point
                    double u_x_gp = N_gp.dot(u_element_eigen);
                    double u_y_gp = N_gp.dot(v_element_eigen);
                    Eigen::Vector2d u_gp(u_x_gp, u_y_gp);

                    // Compute element size and stabilization parameter tau_e
                    double h_e = std::sqrt(2.0 * detJ);
                    double u_mag = u_gp.norm();
                    double tau_e = h_e / (2.0 * u_mag + 1e-10);  // Small number to avoid division by zero

                    double dV = wgp_buf(i) * wgp_buf(j) * detJ;

                    // Compute convection matrix entries
                    for (int a = 0; a < n_local_nodes; ++a) {
                        for (int b = 0; b < n_local_nodes; ++b) {
                            int ia_u = a;
                            int ia_v = a + n_local_nodes;
                            int ib_u = b;
                            int ib_v = b + n_local_nodes;

                            // Standard Galerkin terms for u_x and u_y equations
                            C_element(ia_u, ib_u) += N_gp[a] * (u_x_gp * dN_dx_gp(0, b) + u_y_gp * dN_dx_gp(1, b)) * dV;
                            C_element(ia_v, ib_v) += N_gp[a] * (u_x_gp * dN_dx_gp(0, b) + u_y_gp * dN_dx_gp(1, b)) * dV;

                            // supg stabilization terms
                            Eigen::Vector2d grad_N_a = dN_dx_gp.col(a);
                            double streamline_derivative = u_gp.dot(grad_N_a);

                            C_supg_element(ia_u, ib_u) += tau_e * streamline_derivative * u_gp.dot(dN_dx_gp.col(b)) * dV;
                            C_supg_element(ia_v, ib_v) += tau_e * streamline_derivative * u_gp.dot(dN_dx_gp.col(b)) * dV;
                        }
                    }
                }
            }

            // Assemble the local matrix into the global sparse matrix
            for (int a = 0; a < n_local_nodes; ++a) {
                int global_a_x = element_nodes[a];
                int global_a_y = element_nodes[a] + n_nodes;

                for (int b = 0; b < n_local_nodes; ++b) {
                    int global_b_x = element_nodes[b];
                    int global_b_y = element_nodes[b] + n_nodes;

                    double value = C_element(a, b) + C_supg_element(a, b);

                    // For u_x component
                    local_tripletList.emplace_back(global_a_x, global_b_x, value);
                    // For u_y component
                    local_tripletList.emplace_back(global_a_y, global_b_y, value);
                }
            }
        }

        // Merge local triplet lists into the global list
        #pragma omp critical
        {
            tripletList.insert(tripletList.end(), local_tripletList.begin(), local_tripletList.end());
        }
    }

    // Convert triplet list to sparse matrix
    Eigen::SparseMatrix<double> C_global(2 * n_nodes, 2 * n_nodes);
    C_global.setFromTriplets(tripletList.begin(), tripletList.end());

    return C_global;  // Return sparse matrix directly
}

// Binding the C++ function to Python using Pybind11
PYBIND11_MODULE(convection_matrix_supg_parallel, m) {
    m.def("assemble_convection_matrix_supg", &assemble_convection_matrix_supg, "Assemble sparse convection matrix with supg");
}



