// File: randomized_svd_solver.cpp
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <random>

namespace py = pybind11;

// Function to perform randomized SVD using Eigen
std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd> randomized_svd(const Eigen::MatrixXd& matrix, int n_components, int n_iter = 2) {
    int m = matrix.rows();
    int n = matrix.cols();

    // Step 1: Create a random matrix of size (n, n_components)
    Eigen::MatrixXd omega = Eigen::MatrixXd::Random(n, n_components);

    // Step 2: Compute Y = A * omega
    Eigen::MatrixXd Y = matrix * omega;

    // Step 3: Perform QR decomposition on Y
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Y);
    Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(m, n_components);

    // Step 4: Project A onto Q to form B = Q^T * A
    Eigen::MatrixXd B = Q.transpose() * matrix;

    // Step 5: Compute the SVD of B (smaller problem)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd U_tilde = svd.matrixU();
    Eigen::VectorXd S = svd.singularValues();
    Eigen::MatrixXd V = svd.matrixV();

    // Step 6: Form the approximate U = Q * U_tilde
    Eigen::MatrixXd U = Q * U_tilde;

    return std::make_tuple(U, S, V);
}

// Pybind11 module definition
PYBIND11_MODULE(svd_solver, m) {
    m.doc() = "Module for computing randomized SVD using Eigen";  // Optional module docstring
    m.def("randomized_svd", &randomized_svd, "Performs randomized SVD on a given matrix",
          py::arg("matrix"), py::arg("n_components"), py::arg("n_iter") = 2);
}
