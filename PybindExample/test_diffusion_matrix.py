import numpy as np
import scipy.sparse as sp
import time
import diffusion_matrix_parallel
from numba import njit, prange
import os
os.environ['NUMBA_THREADING_LAYER'] = 'omp'

# Numba-compatible functions for shape functions and their derivatives
@njit
def dN_dxi_numba(xi, eta):
    return 0.25 * np.array([[-(1 - eta), -(1 - xi)], 
                            [(1 - eta), -(1 + xi)], 
                            [(1 + eta), (1 + xi)], 
                            [-(1 + eta), (1 - xi)]])

@njit
def N_numba(xi, eta):
    return 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta)
    ])

class FEMBurgers2D:
    def __init__(self, X, Y, T):
        self.X = X
        self.Y = Y
        self.T = T
        self.n_local_nodes = self.T.shape[1]  # Number of local nodes per element
        self.ngaus = 2
        self.zgp = np.array([-np.sqrt(3) / 3, np.sqrt(3) / 3])
        self.wgp = np.array([1, 1])

    def compute_diffusion_matrix_python(self):
        n_nodes = len(self.X)
        n_elements, n_local_nodes = self.T.shape
        K_global = sp.lil_matrix((2 * n_nodes, 2 * n_nodes))

        for elem in range(n_elements):
            element_nodes = self.T[elem, :] - 1  # Adjusted for 0-based indexing
            x_element = self.X[element_nodes]
            y_element = self.Y[element_nodes]

            K_element = np.zeros((n_local_nodes, n_local_nodes))

            for i in range(self.ngaus):
                for j in range(self.ngaus):
                    xi = self.zgp[i]
                    eta = self.zgp[j]
                    N_gp = N_numba(xi, eta)

                    dN_dxi_gp = dN_dxi_numba(xi, eta)
                    J = dN_dxi_gp.T @ np.vstack((x_element, y_element)).T
                    detJ = np.linalg.det(J)
                    invJ = np.linalg.inv(J)
                    dN_dx_gp = invJ @ dN_dxi_gp.T
                    dV = self.wgp[i] * self.wgp[j] * detJ

                    K_element += dV * (dN_dx_gp.T @ dN_dx_gp)

            for a in range(n_local_nodes):
                for b in range(n_local_nodes):
                    K_global[element_nodes[a], element_nodes[b]] += K_element[a, b]
                    K_global[element_nodes[a] + n_nodes, element_nodes[b] + n_nodes] += K_element[a, b]

        return K_global.tocsc()

    def compute_diffusion_matrix_numba(self):
        n_nodes = len(self.X)
        n_elements = self.T.shape[0]
        n_local_nodes = self.n_local_nodes
        total_entries = 2 * n_elements * n_local_nodes * n_local_nodes

        I_data = np.zeros(total_entries, dtype=np.int32)
        J_data = np.zeros(total_entries, dtype=np.int32)
        V_data = np.zeros(total_entries)

        X = self.X
        Y = self.Y
        T = self.T - 1  # Adjusted for 0-based indexing
        ngaus = self.ngaus
        zgp = self.zgp
        wgp = self.wgp

        N_values = np.zeros((ngaus, ngaus, n_local_nodes))
        dN_dxi_values = np.zeros((ngaus, ngaus, n_local_nodes, 2))  # Derivatives of shape functions wrt xi and eta
        for i in range(ngaus):
            for j in range(ngaus):
                xi = zgp[i]
                eta = zgp[j]
                N_gp = N_numba(xi, eta)
                dN_dxi_gp = dN_dxi_numba(xi, eta)
                N_values[i, j, :] = N_gp
                dN_dxi_values[i, j, :, :] = dN_dxi_gp

        @njit(parallel=True)
        def assemble_diffusion_matrix(I_data, J_data, V_data, X, Y, T, n_elements, n_local_nodes, N_values, dN_dxi_values, zgp, wgp, n_nodes):
            for elem in prange(n_elements):
                element_nodes = T[elem, :]
                x_element = X[element_nodes]
                y_element = Y[element_nodes]
                K_element = np.zeros((n_local_nodes, n_local_nodes))
                for i in range(ngaus):
                    for j in range(ngaus):
                        xi = zgp[i]
                        eta = zgp[j]
                        N_gp = N_values[i, j, :]
                        dN_dxi_gp = dN_dxi_values[i, j, :, :]  # Shape (n_local_nodes, 2)
                        J = np.zeros((2, 2))
                        for a in range(n_local_nodes):
                            J[0, 0] += dN_dxi_gp[a, 0] * x_element[a]
                            J[0, 1] += dN_dxi_gp[a, 0] * y_element[a]
                            J[1, 0] += dN_dxi_gp[a, 1] * x_element[a]
                            J[1, 1] += dN_dxi_gp[a, 1] * y_element[a]
                        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                        invJ = np.linalg.inv(J)
                        dN_dx_gp = np.dot(invJ, dN_dxi_gp.T)

                        dV = wgp[i] * wgp[j] * detJ
                        K_element += dV * (dN_dx_gp.T @ dN_dx_gp)

                idx_elem = elem * n_local_nodes * n_local_nodes * 2
                idx = idx_elem
                for a in range(n_local_nodes):
                    global_a_x = element_nodes[a]
                    global_a_y = element_nodes[a] + n_nodes
                    for b in range(n_local_nodes):
                        global_b_x = element_nodes[b]
                        global_b_y = element_nodes[b] + n_nodes
                        value = K_element[a, b]
                        I_data[idx] = global_a_x
                        J_data[idx] = global_b_x
                        V_data[idx] = value
                        idx += 1
                        I_data[idx] = global_a_y
                        J_data[idx] = global_b_y
                        V_data[idx] = value
                        idx += 1

        assemble_diffusion_matrix(I_data, J_data, V_data, X, Y, T, n_elements, n_local_nodes, N_values, dN_dxi_values, zgp, wgp, n_nodes)
        K_global = sp.coo_matrix((V_data, (I_data, J_data)), shape=(2 * n_nodes, 2 * n_nodes)).tocsc()
        return K_global

# Initialize problem parameters
a, b = 0, 100
nx, ny = 250, 250
x = np.linspace(a, b, nx + 1)
y = np.linspace(a, b, ny + 1)
X_grid, Y_grid = np.meshgrid(x, y)
X, Y = X_grid.flatten(), Y_grid.flatten()
node_indices = np.arange((nx + 1) * (ny + 1)).reshape((ny + 1, nx + 1))

T = []
for i in range(ny):
    for j in range(nx):
        n1 = node_indices[i, j]
        n2 = node_indices[i, j + 1]
        n3 = node_indices[i + 1, j + 1]
        n4 = node_indices[i + 1, j]
        T.append([n1 + 1, n2 + 1, n3 + 1, n4 + 1])  # +1 for 1-based indexing
T = np.array(T)

# FEMBurgers2D setup
fem_burgers_2d = FEMBurgers2D(X, Y, T)

# Measure time for Python diffusion matrix assembly
start_time_python = time.time()
K_python = fem_burgers_2d.compute_diffusion_matrix_python()
end_time_python = time.time()
time_python = end_time_python - start_time_python
print(f"Time for Python Diffusion Matrix: {time_python:.6f} seconds")

# Measure time for Numba diffusion matrix assembly
start_time_numba = time.time()
K_numba = fem_burgers_2d.compute_diffusion_matrix_numba()
end_time_numba = time.time()
time_numba = end_time_numba - start_time_numba
print(f"Time for Numba Diffusion Matrix: {time_numba:.6f} seconds")

# Call the C++ function to assemble the diffusion matrix
start_time_cpp = time.time()
n_nodes = len(X)
n_elements = T.shape[0]

# Precompute N_values and dN_dxi_values
N_values = np.zeros((fem_burgers_2d.ngaus, fem_burgers_2d.ngaus, fem_burgers_2d.n_local_nodes))
dN_dxi_values = np.zeros((fem_burgers_2d.ngaus, fem_burgers_2d.ngaus, fem_burgers_2d.n_local_nodes, 2))
for i in range(fem_burgers_2d.ngaus):
    for j in range(fem_burgers_2d.ngaus):
        xi = fem_burgers_2d.zgp[i]
        eta = fem_burgers_2d.zgp[j]
        N_values[i, j, :] = N_numba(xi, eta)
        dN_dxi_values[i, j, :, :] = dN_dxi_numba(xi, eta)

K_cpp = diffusion_matrix_parallel.assemble_diffusion_matrix(
    X, Y, T, N_values, dN_dxi_values, fem_burgers_2d.wgp, fem_burgers_2d.zgp, n_nodes, n_elements, fem_burgers_2d.n_local_nodes)
end_time_cpp = time.time()
time_cpp = end_time_cpp - start_time_cpp
print(f"Time for C++ Diffusion Matrix: {time_cpp:.6f} seconds")

# Function to compute the Frobenius norm for sparse matrices
def frobenius_norm_sparse(A, B):
    """Compute the Frobenius norm of the difference between two sparse matrices."""
    diff = A - B
    return np.linalg.norm(diff.data)

# Compute the Frobenius norm differences between the matrices
difference_python_numba = frobenius_norm_sparse(K_python, K_numba)
difference_python_cpp = frobenius_norm_sparse(K_python, K_cpp)

# Print out the differences
print(f"Difference between Python and Numba (Diffusion Matrix): {difference_python_numba:.2e}")
print(f"Difference between Python and C++ (Diffusion Matrix): {difference_python_cpp:.2e}")




