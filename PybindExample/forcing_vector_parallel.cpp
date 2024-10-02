#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Sparse>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <omp.h>  // Include OpenMP for parallelization

namespace py = pybind11;

// Function to assemble the forcing vector using Eigen's matrix
py::array_t<double> assemble_forcing_vector(py::array_t<double> X, py::array_t<double> Y, py::array_t<int> T,
                                            py::array_t<double> N_values, py::array_t<double> wgp, py::array_t<double> zgp,
                                            double mu2, int n_nodes, int n_elements, int n_local_nodes) {
    // Map the input data
    auto X_buf = X.unchecked<1>();
    auto Y_buf = Y.unchecked<1>();
    auto T_buf = T.unchecked<2>();
    auto N_buf = N_values.unchecked<3>();
    auto wgp_buf = wgp.unchecked<1>();
    auto zgp_buf = zgp.unchecked<1>();

    // Initialize the global forcing vector
    Eigen::VectorXd F_global = Eigen::VectorXd::Zero(2 * n_nodes);

    // Parallelize the outer loop using OpenMP
    #pragma omp parallel
    {
        Eigen::VectorXd F_local = Eigen::VectorXd::Zero(2 * n_nodes);  // Local forcing vector for each thread

        #pragma omp for
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

            // Initialize the local forcing vector
            Eigen::MatrixXd F_element = Eigen::MatrixXd::Zero(n_local_nodes, 2);

            // Loop over Gauss points
            for (int i = 0; i < 2; ++i) {  // ngaus = 2
                for (int j = 0; j < 2; ++j) {  // ngaus = 2
                    // Shape functions at the Gauss point
                    Eigen::VectorXd N_gp(n_local_nodes);
                    for (int k = 0; k < n_local_nodes; ++k) {
                        N_gp(k) = N_buf(i, j, k);
                    }

                    // Compute Jacobian and its determinant
                    Eigen::MatrixXd dN_dxi_gp(4, 2);
                    dN_dxi_gp(0, 0) = -(1 - zgp_buf(j)) / 4.0; dN_dxi_gp(0, 1) = -(1 - zgp_buf(i)) / 4.0;
                    dN_dxi_gp(1, 0) = (1 - zgp_buf(j)) / 4.0; dN_dxi_gp(1, 1) = -(1 + zgp_buf(i)) / 4.0;
                    dN_dxi_gp(2, 0) = (1 + zgp_buf(j)) / 4.0; dN_dxi_gp(2, 1) = (1 + zgp_buf(i)) / 4.0;
                    dN_dxi_gp(3, 0) = -(1 + zgp_buf(j)) / 4.0; dN_dxi_gp(3, 1) = (1 - zgp_buf(i)) / 4.0;

                    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2, 2);
                    for (int k = 0; k < n_local_nodes; ++k) {
                        J(0, 0) += dN_dxi_gp(k, 0) * x_element[k];
                        J(0, 1) += dN_dxi_gp(k, 0) * y_element[k];
                        J(1, 0) += dN_dxi_gp(k, 1) * x_element[k];
                        J(1, 1) += dN_dxi_gp(k, 1) * y_element[k];
                    }

                    double detJ = J.determinant();
                    double dV = wgp_buf(i) * wgp_buf(j) * detJ;

                    // Compute physical coordinates at the Gauss point
                    double x_gp = N_gp.dot(Eigen::Map<Eigen::Vector4d>(x_element.data()));

                    // Compute the forcing term at the Gauss point
                    double f_x_gp = 0.02 * std::exp(mu2 * x_gp);
                    double f_y_gp = 0.0;  // Forcing term for u_y component is zero

                    // Update local forcing vector
                    F_element.col(0) += f_x_gp * N_gp * dV;
                    F_element.col(1) += f_y_gp * N_gp * dV;
                }
            }

            // Assemble the local forcing vector into the global vector
            for (int a = 0; a < n_local_nodes; ++a) {
                F_local[element_nodes[a]] += F_element(a, 0);
                F_local[element_nodes[a] + n_nodes] += F_element(a, 1);
            }
        }

        // Critical section to safely update the global vector
        #pragma omp critical
        {
            F_global += F_local;
        }
    }

    // Convert Eigen vector to a NumPy array and return it
    py::array_t<double> result = py::array_t<double>({2 * n_nodes});
    Eigen::Map<Eigen::VectorXd>(result.mutable_data(), 2 * n_nodes) = F_global;
    return result;
}

// Binding the C++ function to Python using Pybind11
PYBIND11_MODULE(forcing_vector_parallel, m) {
    m.def("assemble_forcing_vector", &assemble_forcing_vector, "Assemble forcing vector");
}





