#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <omp.h>  // Include OpenMP
#include <chrono> // For timing
#include <iostream> // For printing

namespace py = pybind11;

// Function to perform sparse-dense matrix multiplication and then compute Ar and br
std::tuple<Eigen::MatrixXd, Eigen::VectorXd> compute_Ar_br(
    const Eigen::SparseMatrix<double>& A, 
    const Eigen::MatrixXd& Phi, 
    const Eigen::VectorXd& R
) {
    // Step 1: Compute J_Phi = A * Phi
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd J_Phi = A * Phi;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Step 1 (J_Phi computation) time: " << elapsed.count() << " seconds" << std::endl;

    // Step 2: Compute Ar = J_Phi^T * J_Phi in parallel
    start = std::chrono::high_resolution_clock::now();
    int num_cols = J_Phi.cols();
    Eigen::MatrixXd Ar(num_cols, num_cols);
    #pragma omp parallel for
    for (int i = 0; i < num_cols; ++i) {
        for (int j = 0; j <= i; ++j) {
            Ar(i, j) = J_Phi.col(i).dot(J_Phi.col(j));
            Ar(j, i) = Ar(i, j); // Exploit symmetry
        }
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Step 2 (Ar computation) time: " << elapsed.count() << " seconds" << std::endl;

    // Step 3: Compute br = J_Phi^T * R in parallel
    start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd br(num_cols);
    #pragma omp parallel for
    for (int i = 0; i < num_cols; ++i) {
        br(i) = J_Phi.col(i).dot(R);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Step 3 (br computation) time: " << elapsed.count() << " seconds" << std::endl;

    // Step 4: Create the tuple for returning the results
    start = std::chrono::high_resolution_clock::now();
    auto result = std::make_tuple(Ar, br);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Step 4 (tuple creation) time: " << elapsed.count() << " seconds" << std::endl;

    // Return both Ar and br
    return result;
}

PYBIND11_MODULE(eigen_sparse_dense_operations, m) {
    m.doc() = "Module for performing optimized sparse-dense matrix operations using Eigen";
    m.def("compute_Ar_br", &compute_Ar_br, "Computes Ar = J_Phi^T * J_Phi and br = J_Phi^T * R",
          py::arg("A"), py::arg("Phi"), py::arg("R"));
}


