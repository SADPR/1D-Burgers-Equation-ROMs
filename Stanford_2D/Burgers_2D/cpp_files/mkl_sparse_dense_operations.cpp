#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <mkl.h>    // Include MKL header
#include <chrono>   // For timing
#include <iostream> // For printing
#include <Eigen/Core>
#include <omp.h> // Include OpenMP header

namespace py = pybind11;

void compute_Ar_br(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::Ref<const Eigen::MatrixXd>& Phi,
    const Eigen::Ref<const Eigen::VectorXd>& R,
    Eigen::Ref<Eigen::MatrixXd> Ar,
    Eigen::Ref<Eigen::VectorXd> br,
    bool echo_level
) {
    auto starts = std::chrono::high_resolution_clock::now();
    int num_rows = A.rows();
    int num_cols = Phi.cols();

    // Convert A to CSR format (row-major)
    Eigen::SparseMatrix<double, Eigen::RowMajor> A_csr = A;

    // Step 1: Compute J_Phi = A * Phi using MKL
    auto start = std::chrono::high_resolution_clock::now();

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
        throw std::runtime_error("MKL sparse matrix creation failed.");
    }

    Eigen::MatrixXd J_Phi = Eigen::MatrixXd::Zero(num_rows, num_cols);

    // Perform matrix multiplication J_Phi = A * Phi using MKL
    status = mkl_sparse_d_mm(
        SPARSE_OPERATION_NON_TRANSPOSE, 1.0, mkl_A, descr, SPARSE_LAYOUT_COLUMN_MAJOR,
        Phi.data(), num_cols, Phi.rows(), 0.0, J_Phi.data(), J_Phi.rows()
    );
    if (status != SPARSE_STATUS_SUCCESS) {
        mkl_sparse_destroy(mkl_A);
        std::cerr << "Error: MKL sparse-dense multiplication failed." << std::endl;
        throw std::runtime_error("MKL sparse-dense multiplication failed.");
    }

    auto end = std::chrono::high_resolution_clock::now();
    if (echo_level) {
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "MKL Step 1 (J_Phi computation) time: " << elapsed.count() << " seconds" << std::endl;
    }

    // Step 2: Compute Ar = J_Phi^T * J_Phi using MKL's cblas_dgemm
    start = std::chrono::high_resolution_clock::now();

    int m = J_Phi.rows(); // Number of rows in J_Phi
    int n = J_Phi.cols(); // Number of columns in J_Phi

    cblas_dgemm(
        CblasColMajor, CblasTrans, CblasNoTrans,
        n, n, m,
        1.0,
        J_Phi.data(), m, // A = J_Phi, lda = m
        J_Phi.data(), m, // B = J_Phi, ldb = m
        0.0,
        Ar.data(), n     // C = Ar, ldc = n
    );

    end = std::chrono::high_resolution_clock::now();
    if (echo_level) {
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "MKL Step 2 (Ar computation) time: " << elapsed.count() << " seconds" << std::endl;
    }

    // Step 3: Compute br = J_Phi^T * R using MKL's cblas_dgemv
    start = std::chrono::high_resolution_clock::now();

    cblas_dgemv(
        CblasColMajor, CblasTrans,
        m, n,
        1.0,
        J_Phi.data(), m, // A = J_Phi, lda = m
        R.data(), 1,     // x = R, incx = 1
        0.0,
        br.data(), 1     // y = br, incy = 1
    );

    end = std::chrono::high_resolution_clock::now();
    if (echo_level) {
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "MKL Step 3 (br computation) time: " << elapsed.count() << " seconds" << std::endl;
    }

    // Clean up MKL resources
    mkl_sparse_destroy(mkl_A);

    auto endd = std::chrono::high_resolution_clock::now();
    if (echo_level) {
        std::chrono::duration<double> elapsedd = endd - starts;
        std::cout << "Total time: " << elapsedd.count() << " seconds" << std::endl;
    }
}

// Function to perform dense matrix multiplication using MKL (U_s * rbf_jacobian)
void multiply_dense_matrices_mkl(const Eigen::Ref<const Eigen::MatrixXd>& U_s, 
                                 const Eigen::Ref<const Eigen::MatrixXd>& rbf_jacobian, 
                                 Eigen::Ref<Eigen::MatrixXd> result, 
                                 bool echo_level) {
    // Get the shape of the input matrices
    int m = U_s.rows(); // Number of rows in U_s
    int k = U_s.cols(); // Number of columns in U_s
    int n = rbf_jacobian.cols(); // Number of columns in rbf_jacobian

    // Ensure the dimensions are compatible for multiplication
    if (k != rbf_jacobian.rows()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    // Start the timing
    auto start = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication using MKL
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                1.0,
                U_s.data(), m, // A = U_s, lda = m
                rbf_jacobian.data(), k, // B = rbf_jacobian, ldb = k
                0.0,
                result.data(), m // C = result, ldc = m
    );

    // End the timing
    auto end = std::chrono::high_resolution_clock::now();
    if (echo_level) {
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "MKL Dense Matrix Multiplication time: " << elapsed.count() << " seconds" << std::endl;
    }
}



// Function to perform dense matrix addition in parallel using MKL
void sum_dense_matrices_mkl(
    const Eigen::Ref<const Eigen::MatrixXd>& U_p,
    const Eigen::Ref<const Eigen::MatrixXd>& dD_rbf_dq,
    Eigen::Ref<Eigen::MatrixXd> result,
    bool echo_level
) {
    int m = U_p.rows();
    int n = U_p.cols();

    auto start = std::chrono::high_resolution_clock::now();

    // Perform the addition in parallel using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < m * n; ++i) {
        result.data()[i] = U_p.data()[i] + dD_rbf_dq.data()[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    if (echo_level) {
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "MKL Parallel Dense Matrix Addition time: " << elapsed.count() << " seconds" << std::endl;
    }
}


PYBIND11_MODULE(mkl_sparse_dense_operations, m) {
    m.doc() = "Module for performing optimized sparse-dense matrix operations using MKL";
    m.def("compute_Ar_br", &compute_Ar_br,
      "Computes Ar and br in-place using MKL",
      py::arg("A"), py::arg("Phi"), py::arg("R"), py::arg("Ar"), py::arg("br"), py::arg("echo_level"));
    m.def("sum_dense_matrices_mkl", &sum_dense_matrices_mkl,
          "Adds two dense matrices in parallel using MKL",
          py::arg("U_p"), py::arg("dD_rbf_dq"), py::arg("result"), py::arg("echo_level"));
    m.def("multiply_dense_matrices_mkl", &multiply_dense_matrices_mkl,
      "Multiplies two dense matrices using MKL and modifies the result in place",
      py::arg("U_s"), py::arg("rbf_jacobian"), py::arg("result"), py::arg("echo_level"));
}

