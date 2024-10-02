import numpy as np
import scipy.sparse as sp
import time
import convection_matrix_supg_parallel
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
        self.N = lambda xi, eta: 0.25 * np.array([(1 - xi) * (1 - eta),
                                                  (1 + xi) * (1 - eta),
                                                  (1 + xi) * (1 + eta),
                                                  (1 - xi) * (1 + eta)])
        self.dN_dxi = lambda xi, eta: 0.25 * np.array([[-(1 - eta), -(1 - xi)],
                                                       [(1 - eta), -(1 + xi)],
                                                       [(1 + eta), (1 + xi)],
                                                       [-(1 + eta), (1 - xi)]])
    
    def compute_convection_matrix_SUPG_python(self, U_n):
        n_nodes = len(self.X)
        n_elements, n_local_nodes = self.T.shape
        C_global = sp.lil_matrix((2 * n_nodes, 2 * n_nodes))

        for elem in range(n_elements):
            element_nodes = self.T[elem, :] - 1
            x_element = self.X[element_nodes]
            y_element = self.Y[element_nodes]
            u_element = U_n[element_nodes, 0]
            v_element = U_n[element_nodes, 1]

            C_element = np.zeros((2 * n_local_nodes, 2 * n_local_nodes))
            C_SUPG_element = np.zeros((2 * n_local_nodes, 2 * n_local_nodes))

            for i in range(self.ngaus):
                for j in range(self.ngaus):
                    xi = self.zgp[i]
                    eta = self.zgp[j]
                    N_gp = self.N(xi, eta)
                    dN_dxi_gp = self.dN_dxi(xi, eta)

                    J = dN_dxi_gp.T @ np.vstack((x_element, y_element)).T
                    detJ = np.linalg.det(J)
                    invJ = np.linalg.inv(J)
                    dN_dx_gp = invJ @ dN_dxi_gp.T

                    u_x_gp = N_gp @ u_element
                    u_y_gp = N_gp @ v_element
                    u_gp = np.array([u_x_gp, u_y_gp])

                    u_mag = np.linalg.norm(u_gp)
                    h_e = np.sqrt(2 * detJ)

                    tau_e = h_e / (2 * u_mag + 1e-10)

                    dV = self.wgp[i] * self.wgp[j] * detJ

                    for a in range(n_local_nodes):
                        for b in range(n_local_nodes):
                            ia_u = a
                            ia_v = a + n_local_nodes
                            ib_u = b
                            ib_v = b + n_local_nodes

                            C_element[ia_u, ib_u] += N_gp[a] * (u_x_gp * dN_dx_gp[0, b] + u_y_gp * dN_dx_gp[1, b]) * dV
                            C_element[ia_v, ib_v] += N_gp[a] * (u_x_gp * dN_dx_gp[0, b] + u_y_gp * dN_dx_gp[1, b]) * dV

                            grad_N_a = dN_dx_gp[:, a]
                            streamline_derivative = u_gp @ grad_N_a

                            C_SUPG_element[ia_u, ib_u] += tau_e * streamline_derivative * (u_gp @ dN_dx_gp[:, b]) * dV
                            C_SUPG_element[ia_v, ib_v] += tau_e * streamline_derivative * (u_gp @ dN_dx_gp[:, b]) * dV

            global_indices = np.concatenate([element_nodes, element_nodes + n_nodes])
            for local_row, global_row in enumerate(global_indices):
                for local_col, global_col in enumerate(global_indices):
                    C_global[global_row, global_col] += C_element[local_row, local_col] + C_SUPG_element[local_row, local_col]

        return C_global.tocsc()

    def compute_convection_matrix_SUPG_numba(self, U_n):
        n_nodes = len(self.X)
        n_elements = self.T.shape[0]
        n_local_nodes = self.n_local_nodes

        total_entries = 2 * n_elements * n_local_nodes * n_local_nodes
        I_data = np.zeros(total_entries, dtype=np.int32)
        J_data = np.zeros(total_entries, dtype=np.int32)
        V_data = np.zeros(total_entries)

        X = self.X
        Y = self.Y
        T = self.T - 1
        ngaus = self.ngaus
        zgp = self.zgp
        wgp = self.wgp

        N_values = np.zeros((ngaus, ngaus, n_local_nodes))
        dN_values = np.zeros((ngaus, ngaus, n_local_nodes, 2))
        for i in range(ngaus):
            for j in range(ngaus):
                xi = zgp[i]
                eta = zgp[j]
                N_gp = N_numba(xi, eta)
                dN_dxi_gp = dN_dxi_numba(xi, eta)
                N_values[i, j, :] = N_gp
                dN_values[i, j, :, :] = dN_dxi_gp

        @njit(parallel=True)
        def assemble_convection_SUPG(I_data, J_data, V_data, X, Y, U_n, T, n_elements, n_local_nodes, N_values, dN_values, zgp, wgp, n_nodes):
            for elem in prange(n_elements):
                element_nodes = T[elem, :]
                x_element = X[element_nodes]
                y_element = Y[element_nodes]
                u_element = U_n[element_nodes, 0]
                v_element = U_n[element_nodes, 1]

                C_element = np.zeros((2 * n_local_nodes, 2 * n_local_nodes))
                C_SUPG_element = np.zeros((2 * n_local_nodes, 2 * n_local_nodes))

                for i in range(ngaus):
                    for j in range(ngaus):
                        N_gp = N_values[i, j, :]
                        dN_dxi_gp = dN_values[i, j, :, :]

                        J = np.zeros((2, 2))
                        for a in range(n_local_nodes):
                            J[0, 0] += dN_dxi_gp[a, 0] * x_element[a]
                            J[0, 1] += dN_dxi_gp[a, 0] * y_element[a]
                            J[1, 0] += dN_dxi_gp[a, 1] * x_element[a]
                            J[1, 1] += dN_dxi_gp[a, 1] * y_element[a]

                        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                        invJ = np.linalg.inv(J)
                        dN_dx_gp = invJ @ dN_dxi_gp.T

                        u_x_gp = np.dot(N_gp, u_element)
                        u_y_gp = np.dot(N_gp, v_element)
                        u_gp = np.array([u_x_gp, u_y_gp])

                        u_mag = np.linalg.norm(u_gp)
                        h_e = np.sqrt(2 * detJ)
                        tau_e = h_e / (2 * u_mag + 1e-10)

                        dV = wgp[i] * wgp[j] * detJ

                        for a in range(n_local_nodes):
                            for b in range(n_local_nodes):
                                ia_u = a
                                ia_v = a + n_local_nodes
                                ib_u = b
                                ib_v = b + n_local_nodes

                                C_element[ia_u, ib_u] += N_gp[a] * (u_x_gp * dN_dx_gp[0, b] + u_y_gp * dN_dx_gp[1, b]) * dV
                                C_element[ia_v, ib_v] += N_gp[a] * (u_x_gp * dN_dx_gp[0, b] + u_y_gp * dN_dx_gp[1, b]) * dV

                                grad_N_a = dN_dx_gp[:, a]
                                streamline_derivative = np.dot(u_gp, grad_N_a)

                                C_SUPG_element[ia_u, ib_u] += tau_e * streamline_derivative * np.dot(u_gp, dN_dx_gp[:, b]) * dV
                                C_SUPG_element[ia_v, ib_v] += tau_e * streamline_derivative * np.dot(u_gp, dN_dx_gp[:, b]) * dV

                idx_elem = elem * n_local_nodes * n_local_nodes * 2
                idx = idx_elem
                for a in range(n_local_nodes):
                    global_a_x = element_nodes[a]
                    global_a_y = element_nodes[a] + n_nodes
                    for b in range(n_local_nodes):
                        global_b_x = element_nodes[b]
                        global_b_y = element_nodes[b] + n_nodes
                        value = C_element[a, b] + C_SUPG_element[a, b]
                        I_data[idx] = global_a_x
                        J_data[idx] = global_b_x
                        V_data[idx] = value
                        idx += 1
                        I_data[idx] = global_a_y
                        J_data[idx] = global_b_y
                        V_data[idx] = value
                        idx += 1

        assemble_convection_SUPG(I_data, J_data, V_data, X, Y, U_n, T, n_elements,
                                 n_local_nodes, N_values, dN_values, zgp, wgp, n_nodes)
        C_global = sp.coo_matrix((V_data, (I_data, J_data)), shape=(2 * n_nodes, 2 * n_nodes)).tocsc()
        return C_global

# Initialize problem parameters
a, b = 0, 100
nx, ny = 50, 50  # Mesh size
x = np.linspace(a, b, nx + 1)
y = np.linspace(a, b, ny + 1)
X_grid, Y_grid = np.meshgrid(x, y)
X, Y = X_grid.flatten(), Y_grid.flatten()
node_indices = np.arange((nx + 1) * (ny + 1)).reshape((ny + 1, nx + 1))

# Generate T (connectivity matrix)
T = []
for i in range(ny):
    for j in range(nx):
        n1 = node_indices[i, j]
        n2 = node_indices[i, j + 1]
        n3 = node_indices[i + 1, j + 1]
        n4 = node_indices[i + 1, j]
        T.append([n1 + 1, n2 + 1, n3 + 1, n4 + 1])  # +1 for 1-based indexing
T = np.array(T)

# Initialize velocity field U_n
U_n = np.random.rand(len(X), 2)  # Random velocity field

# FEMBurgers2D setup
fem_burgers_2d = FEMBurgers2D(X, Y, T)

# Measure time for Python assembly
start_time_python = time.time()
C_python = fem_burgers_2d.compute_convection_matrix_SUPG_python(U_n)
end_time_python = time.time()
time_python = end_time_python - start_time_python
print(f"Time for Python: {time_python:.6f} seconds")

# Measure time for Numba assembly
start_time_numba = time.time()
C_numba = fem_burgers_2d.compute_convection_matrix_SUPG_numba(U_n)
end_time_numba = time.time()
time_numba = end_time_numba - start_time_numba
print(f"Time for Numba: {time_numba:.6f} seconds")

# Call the C++ function to assemble the convection matrix with SUPG (sparse version)
start_time_cpp = time.time()
n_nodes = len(X)
n_elements = T.shape[0]
N_values = np.zeros((fem_burgers_2d.ngaus, fem_burgers_2d.ngaus, fem_burgers_2d.n_local_nodes))
dN_values = np.zeros((fem_burgers_2d.ngaus, fem_burgers_2d.ngaus, fem_burgers_2d.n_local_nodes, 2))
for i in range(fem_burgers_2d.ngaus):
    for j in range(fem_burgers_2d.ngaus):
        xi = fem_burgers_2d.zgp[i]
        eta = fem_burgers_2d.zgp[j]
        N_values[i, j, :] = N_numba(xi, eta)
        dN_values[i, j, :, :] = dN_dxi_numba(xi, eta)

C_cpp = convection_matrix_supg_parallel.assemble_convection_matrix_supg(
    X, Y, T, N_values, dN_values, U_n, fem_burgers_2d.wgp, fem_burgers_2d.zgp, n_nodes, n_elements, fem_burgers_2d.n_local_nodes)
end_time_cpp = time.time()
time_cpp = end_time_cpp - start_time_cpp
print(f"Time for C++: {time_cpp:.6f} seconds")

# Function to compute the Frobenius norm for sparse matrices
def frobenius_norm_sparse(A, B):
    """Compute the Frobenius norm of the difference between two sparse matrices."""
    diff = A - B
    return np.linalg.norm(diff.data)

# Compute the Frobenius norm differences between the matrices
difference_python_numba = frobenius_norm_sparse(C_python, C_numba)
difference_python_cpp = frobenius_norm_sparse(C_python, C_cpp)

# Print out the differences
print(f"Difference between Python and Numba: {difference_python_numba:.2e}")
print(f"Difference between Python and C++: {difference_python_cpp:.2e}")



