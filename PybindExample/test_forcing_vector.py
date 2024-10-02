import numpy as np
import time
import scipy.sparse as sp
import forcing_vector_parallel
from numba import njit, prange
import os
os.environ['NUMBA_THREADING_LAYER'] = 'omp'

# Define the shape functions and their derivatives for Numba
@njit
def N_numba(xi, eta):
    return 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta)
    ])

@njit
def dN_dxi_numba(xi, eta):
    return 0.25 * np.array([[-(1 - eta), -(1 - xi)], 
                            [(1 - eta), -(1 + xi)], 
                            [(1 + eta), (1 + xi)], 
                            [-(1 + eta), (1 - xi)]])

class FEMBurgers2D:
    def __init__(self, X, Y, T):
        self.X = X
        self.Y = Y
        self.T = T
        self.n_local_nodes = self.T.shape[1]
        self.ngaus = 2
        self.zgp = np.array([-np.sqrt(3) / 3, np.sqrt(3) / 3])
        self.wgp = np.array([1, 1])

    def compute_forcing_vector_python(self, mu2):
        n_nodes = len(self.X)
        n_elements, n_local_nodes = self.T.shape
        F_global = np.zeros(2 * n_nodes)

        for elem in range(n_elements):
            element_nodes = self.T[elem, :] - 1  # Adjust for 0-based indexing
            x_element = self.X[element_nodes]
            y_element = self.Y[element_nodes]

            F_element = np.zeros((n_local_nodes, 2))

            for i in range(self.ngaus):
                for j in range(self.ngaus):
                    xi = self.zgp[i]
                    eta = self.zgp[j]
                    N_gp = N_numba(xi, eta)

                    dN_dxi_gp = dN_dxi_numba(xi, eta)
                    J = dN_dxi_gp.T @ np.vstack((x_element, y_element)).T
                    detJ = np.linalg.det(J)

                    x_gp = N_gp @ x_element
                    y_gp = N_gp @ y_element

                    dV = self.wgp[i] * self.wgp[j] * detJ

                    f_x_gp = 0.02 * np.exp(mu2 * x_gp)
                    f_y_gp = 0.0

                    F_element[:, 0] += f_x_gp * N_gp * dV
                    F_element[:, 1] += f_y_gp * N_gp * dV

            for a in range(n_local_nodes):
                F_global[element_nodes[a]] += F_element[a, 0]
                F_global[element_nodes[a] + n_nodes] += F_element[a, 1]

        return F_global

    def compute_forcing_vector_numba(self, mu2):
        n_nodes = len(self.X)
        n_elements = self.T.shape[0]
        n_local_nodes = self.n_local_nodes
        F_global = np.zeros(2 * n_nodes)

        X = self.X
        Y = self.Y
        T = self.T - 1
        zgp = self.zgp
        wgp = self.wgp

        N_values = np.zeros((self.ngaus, self.ngaus, n_local_nodes))
        for i in range(self.ngaus):
            for j in range(self.ngaus):
                xi = zgp[i]
                eta = zgp[j]
                N_gp = N_numba(xi, eta)
                N_values[i, j, :] = N_gp

        @njit(parallel=True)
        def assemble_forcing_vector(F_global, X, Y, T, n_elements, n_local_nodes, N_values, zgp, wgp, mu2, n_nodes):
            for elem in prange(n_elements):
                element_nodes = T[elem, :]
                x_element = X[element_nodes]
                y_element = Y[element_nodes]
                F_element = np.zeros((n_local_nodes, 2))

                for i in range(2):
                    for j in range(2):
                        N_gp = N_values[i, j, :]
                        dN_dxi_gp = dN_dxi_numba(zgp[i], zgp[j])

                        J = np.zeros((2, 2))
                        for a in range(n_local_nodes):
                            J[0, 0] += dN_dxi_gp[a, 0] * x_element[a]
                            J[0, 1] += dN_dxi_gp[a, 0] * y_element[a]
                            J[1, 0] += dN_dxi_gp[a, 1] * x_element[a]
                            J[1, 1] += dN_dxi_gp[a, 1] * y_element[a]

                        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                        dV = wgp[i] * wgp[j] * detJ

                        x_gp = N_gp @ x_element

                        f_x_gp = 0.02 * np.exp(mu2 * x_gp)

                        F_element[:, 0] += f_x_gp * N_gp * dV
                        F_element[:, 1] += 0.0 * N_gp * dV

                for a in range(n_local_nodes):
                    F_global[element_nodes[a]] += F_element[a, 0]
                    F_global[element_nodes[a] + n_nodes] += F_element[a, 1]

        assemble_forcing_vector(F_global, X, Y, T, n_elements, n_local_nodes, N_values, zgp, wgp, mu2, n_nodes)
        return F_global

# Initialize problem parameters
a, b = 0, 100
nx, ny = 50, 50
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
        T.append([n1 + 1, n2 + 1, n3 + 1, n4 + 1])
T = np.array(T)

# FEMBurgers2D setup
fem_burgers_2d = FEMBurgers2D(X, Y, T)

# Forcing function parameter
mu2 = 0.02

# Measure time for Python assembly
start_time_python = time.time()
F_python = fem_burgers_2d.compute_forcing_vector_python(mu2)
end_time_python = time.time()
time_python = end_time_python - start_time_python
print(f"Time for Python: {time_python:.6f} seconds")

# Measure time for Numba assembly
start_time_numba = time.time()
F_numba = fem_burgers_2d.compute_forcing_vector_numba(mu2)
end_time_numba = time.time()
time_numba = end_time_numba - start_time_numba
print(f"Time for Numba: {time_numba:.6f} seconds")

# Measure time for C++ parallel assembly
start_time_cpp = time.time()
n_nodes = len(X)
n_elements = T.shape[0]
N_values = np.zeros((fem_burgers_2d.ngaus, fem_burgers_2d.ngaus, fem_burgers_2d.n_local_nodes))
for i in range(fem_burgers_2d.ngaus):
    for j in range(fem_burgers_2d.ngaus):
        xi = fem_burgers_2d.zgp[i]
        eta = fem_burgers_2d.zgp[j]
        N_values[i, j, :] = N_numba(xi, eta)

F_cpp = forcing_vector_parallel.assemble_forcing_vector(
    X, Y, T, N_values, fem_burgers_2d.wgp, fem_burgers_2d.zgp, mu2, n_nodes, n_elements, fem_burgers_2d.n_local_nodes)
end_time_cpp = time.time()
time_cpp = end_time_cpp - start_time_cpp
print(f"Time for C++: {time_cpp:.6f} seconds")

# Function to compare results
def compare_forcing_vectors(F_python, F_numba, F_cpp):
    diff_python_numba = np.linalg.norm(F_python - F_numba)
    diff_python_cpp = np.linalg.norm(F_python - F_cpp)
    print(f"Difference between Python and Numba: {diff_python_numba:.2e}")
    print(f"Difference between Python and C++: {diff_python_cpp:.2e}")

# Compare results
compare_forcing_vectors(F_python, F_numba, F_cpp)




