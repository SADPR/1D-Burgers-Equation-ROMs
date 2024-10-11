#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <mkl.h>    // Include MKL header
#include <chrono>   // For timing
#include <iostream> // For printing

namespace py = pybind11;

// Function to compute Ar and br using MKL
std::tuple<Eigen::MatrixXd, Eigen::VectorXd> compute_Ar_br(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::MatrixXd& Phi,
    const Eigen::VectorXd& R
) {
    int num_rows = A.rows();
    int num_cols = Phi.cols();

    // Step 1: Compute J_Phi = A * Phi using MKL

    // Convert A to CSR format (row-major)
    Eigen::SparseMatrix<double, Eigen::RowMajor> A_csr = A;

    // auto start = std::chrono::high_resolution_clock::now();

    // Create MKL sparse matrix
    sparse_matrix_t mkl_A;
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    sparse_status_t status = mkl_sparse_d_create_csr(
        &mkl_A, SPARSE_INDEX_BASE_ZERO, A_csr.rows(), A_csr.cols(),
        const_cast<int*>(A_csr.outerIndexPtr()),
        const_cast<int*>(A_csr.outerIndexPtr() + 1),
        const_cast<int*>(A_csr.innerIndexPtr()),
        const_cast<double*>(A_csr.valuePtr())
    );
    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Error: MKL sparse matrix creation failed." << std::endl;
        return std::make_tuple(Eigen::MatrixXd(), Eigen::VectorXd());
    }

    Eigen::MatrixXd J_Phi = Eigen::MatrixXd::Zero(num_rows, num_cols);

    // Use SPARSE_LAYOUT_COLUMN_MAJOR because Eigen's matrices are column-major by default
    status = mkl_sparse_d_mm(
        SPARSE_OPERATION_NON_TRANSPOSE, 1.0, mkl_A, descr, SPARSE_LAYOUT_COLUMN_MAJOR,
        Phi.data(), num_cols, Phi.rows(), 0.0, J_Phi.data(), J_Phi.rows()
    );
    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Error: MKL sparse-dense multiplication failed." << std::endl;
        mkl_sparse_destroy(mkl_A);
        return std::make_tuple(Eigen::MatrixXd(), Eigen::VectorXd());
    }

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "MKL Step 1 (J_Phi computation) time: " << elapsed.count() << " seconds" << std::endl;

    // Step 2: Compute Ar = J_Phi^T * J_Phi using MKL's cblas_dgemm

    // start = std::chrono::high_resolution_clock::now();

    int m = J_Phi.rows(); // Number of rows in J_Phi
    int n = J_Phi.cols(); // Number of columns in J_Phi

    Eigen::MatrixXd Ar = Eigen::MatrixXd::Zero(n, n);

    // Perform Ar = J_Phi^T * J_Phi
    cblas_dgemm(
        CblasColMajor, CblasTrans, CblasNoTrans,
        n, n, m,
        1.0,
        J_Phi.data(), m, // A = J_Phi, lda = m (number of rows)
        J_Phi.data(), m, // B = J_Phi, ldb = m
        0.0,
        Ar.data(), n     // C = Ar, ldc = n (number of rows of C)
    );

    // end = std::chrono::high_resolution_clock::now();
    // elapsed = end - start;
    // std::cout << "MKL Step 2 (Ar computation) time: " << elapsed.count() << " seconds" << std::endl;

    // Step 3: Compute br = J_Phi^T * R using MKL's cblas_dgemv

    // start = std::chrono::high_resolution_clock::now();

    Eigen::VectorXd br = Eigen::VectorXd::Zero(n);

    // Perform br = J_Phi^T * R
    cblas_dgemv(
        CblasColMajor, CblasTrans,
        m, n,
        1.0,
        J_Phi.data(), m, // A = J_Phi, lda = m
        R.data(), 1,     // x = R, incx = 1
        0.0,
        br.data(), 1     // y = br, incy = 1
    );

    // end = std::chrono::high_resolution_clock::now();
    // elapsed = end - start;
    // std::cout << "MKL Step 3 (br computation) time: " << elapsed.count() << " seconds" << std::endl;

    // Clean up MKL resources
    mkl_sparse_destroy(mkl_A);

    // Return Ar and br computed using MKL
    return std::make_tuple(Ar, br);
}

PYBIND11_MODULE(mkl_sparse_dense_operations, m) {
    m.doc() = "Module for performing optimized sparse-dense matrix operations using MKL";
    m.def("compute_Ar_br", &compute_Ar_br,
          "Computes Ar = J_Phi^T * J_Phi and br = J_Phi^T * R using MKL",
          py::arg("A"), py::arg("Phi"), py::arg("R"));
}
