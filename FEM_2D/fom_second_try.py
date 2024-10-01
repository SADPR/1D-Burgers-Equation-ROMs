import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from numba import njit, prange

def plot_2d_cuts(U, X, Y, line='x', fixed_value=50.0):
    """
    Plot 2D cuts of the current converged solution along a line.

    Parameters:
    U : np.ndarray
        The current converged solution matrix with shape (n_nodes, 2).
    X, Y : np.ndarray
        The x and y coordinates of the nodes.
    line : str
        The line along which to cut ('x' or 'y').
    fixed_value : float
        The value of y (for line='x') or x (for line='y') to plot the cut.
    """
    if line == 'x':
        # Select the nodes along the line y = fixed_value
        indices = np.where(np.isclose(Y, fixed_value))[0]
        x_values = X[indices]
        u_x_values = U[indices, 0]  # u_x component

        plt.figure()
        plt.plot(x_values, u_x_values, label=f'$u_x(x, y={fixed_value})$')
        plt.xlabel('$x$')
        plt.ylabel(f'$u_x(x, y={fixed_value})$')
        plt.title(f'Solution along y = {fixed_value}')
        plt.legend()
        plt.grid()
        plt.show()

    elif line == 'y':
        # Select the nodes along the line x = fixed_value
        indices = np.where(np.isclose(X, fixed_value))[0]
        y_values = Y[indices]
        u_x_values = U[indices, 0]  # u_x component

        plt.figure()
        plt.plot(y_values, u_x_values, label=f'$u_x(x={fixed_value}, y)$')
        plt.xlabel('$y$')
        plt.ylabel(f'$u_x(x={fixed_value}, y)$')
        plt.title(f'Solution along x = {fixed_value}')
        plt.legend()
        plt.grid()
        plt.show()
        
def plot_3d_surface(U, X, Y):
    """
    Plot a 3D surface plot of the current converged solution.

    Parameters:
    U : np.ndarray
        The current converged solution matrix with shape (n_nodes, 2).
    X, Y : np.ndarray
        The x and y coordinates of the nodes.
    """
    # Create a grid of points
    x_values = np.unique(X)
    y_values = np.unique(Y)
    X_grid, Y_grid = np.meshgrid(x_values, y_values)

    # Reshape U_x values to match the grid
    u_x_values = U[:, 0].reshape(len(y_values), len(x_values))

    # Create a 3D surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_grid, Y_grid, u_x_values, cmap='viridis', edgecolor='none')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$u_x(x, y)$')
    ax.set_title('Current Converged Solution')
    fig.colorbar(surf)
    plt.show()

@njit
def dN_dxi_numba(xi, eta):
    # Define the derivative of the shape functions here
    return 0.25 * np.array([[-(1 - eta), -(1 - xi)], 
                            [(1 - eta), -(1 + xi)], 
                            [(1 + eta), (1 + xi)], 
                            [-(1 + eta), (1 - xi)]])

# Numba-compatible shape functions
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
        self.ngaus = 2
        # 2D Gauss points and weights for quadrature on [-1, 1]
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
        self.n_local_nodes = self.T.shape[1]  # Number of local nodes per element

    def compute_mass_matrix_(self):
        n_nodes = len(self.X)  # Total number of nodes
        n_elements, n_local_nodes = self.T.shape  # Number of elements and local nodes per element
        M_global = sp.lil_matrix((2 * n_nodes, 2 * n_nodes))  # Initialize global mass matrix

        # Print the total number of elements for reference
        print(f"Total number of elements: {n_elements}")

        for elem in range(n_elements):
            element_nodes = self.T[elem, :] - 1  # Adjusted for 0-based indexing
            x_element = self.X[element_nodes]
            y_element = self.Y[element_nodes]

            M_element = np.zeros((n_local_nodes, n_local_nodes))  # Initialize local mass matrix

            for i in range(self.ngaus):
                for j in range(self.ngaus):
                    xi = self.zgp[i]
                    eta = self.zgp[j]
                    N_gp = self.N(xi, eta)  # Shape functions at the Gauss point

                    dN_dxi_gp = self.dN_dxi(xi, eta)  # Shape function derivatives wrt reference coordinates
                    J = dN_dxi_gp.T @ np.vstack((x_element, y_element)).T  # Jacobian matrix
                    detJ = np.linalg.det(J)

                    dV = self.wgp[i] * self.wgp[j] * detJ  # Differential volume

                    M_element += np.outer(N_gp, N_gp) * dV

            # Assemble the local matrix into the global mass matrix for both u_x and u_y components
            for a in range(n_local_nodes):
                for b in range(n_local_nodes):
                    # For u_x component
                    M_global[element_nodes[a], element_nodes[b]] += M_element[a, b]
                    # For u_y component
                    M_global[element_nodes[a] + n_nodes, element_nodes[b] + n_nodes] += M_element[a, b]

            # Print the progress of assembly
            if (elem + 1) % 1000 == 0 or elem == n_elements - 1:
                print(f"Mass Assembled element {elem + 1}/{n_elements}")

        return M_global.tocsc()  # Convert to sparse format for efficiency

    def compute_mass_matrix(self):
        n_nodes = len(self.X)  # Total number of nodes
        n_elements, n_local_nodes = self.T.shape  # Number of elements and local nodes per element

        # Initialize a dense array to accumulate local matrices
        M_global_x = np.zeros((n_nodes, n_nodes))  # For u_x component
        M_global_y = np.zeros((n_nodes, n_nodes))  # For u_y component

        for elem in range(n_elements):
            element_nodes = self.T[elem, :] - 1  # Adjusted for 0-based indexing
            x_element = self.X[element_nodes]
            y_element = self.Y[element_nodes]

            M_element = np.zeros((n_local_nodes, n_local_nodes))  # Initialize local mass matrix

            # Precompute shape functions and volume for each Gauss point
            for i in range(self.ngaus):
                for j in range(self.ngaus):
                    xi = self.zgp[i]
                    eta = self.zgp[j]
                    N_gp = self.N(xi, eta)  # Shape functions at the Gauss point

                    dN_dxi_gp = self.dN_dxi(xi, eta)  # Shape function derivatives wrt reference coordinates
                    J = dN_dxi_gp.T @ np.vstack((x_element, y_element)).T  # Jacobian matrix
                    detJ = np.linalg.det(J)

                    dV = self.wgp[i] * self.wgp[j] * detJ  # Differential volume

                    M_element += np.outer(N_gp, N_gp) * dV

            # Assemble the local matrix into the global mass matrix for both u_x and u_y components
            for a in range(n_local_nodes):
                for b in range(n_local_nodes):
                    M_global_x[element_nodes[a], element_nodes[b]] += M_element[a, b]
                    M_global_y[element_nodes[a], element_nodes[b]] += M_element[a, b]

        # Combine u_x and u_y into the final block sparse matrix
        M_global = sp.bmat([[sp.csc_matrix(M_global_x), None], [None, sp.csc_matrix(M_global_y)]])

        return M_global.tocsc()  # Convert to sparse format for efficiency

    def compute_mass_matrix(self):
        n_nodes = len(self.X)
        n_elements = self.T.shape[0]
        n_local_nodes = self.n_local_nodes
        total_entries = 2 * n_elements * n_local_nodes * n_local_nodes  # For u_x and u_y components

        I_data = np.zeros(total_entries, dtype=np.int32)
        J_data = np.zeros(total_entries, dtype=np.int32)
        V_data = np.zeros(total_entries)

        X = self.X
        Y = self.Y
        T = self.T - 1  # Adjusted for 0-based indexing
        ngaus = self.ngaus
        zgp = self.zgp
        wgp = self.wgp

        # Precompute shape functions and their products at Gauss points
        N_values = np.zeros((ngaus, ngaus, n_local_nodes))
        N_outer = np.zeros((ngaus, ngaus, n_local_nodes, n_local_nodes))
        for i in range(ngaus):
            for j in range(ngaus):
                xi = zgp[i]
                eta = zgp[j]
                N_gp = N_numba(xi, eta)
                N_values[i, j, :] = N_gp
                N_outer[i, j, :, :] = np.outer(N_gp, N_gp)

        # Numba-compiled function for assembly
        @njit(parallel=True)
        def assemble_mass_matrix(I_data, J_data, V_data, X, Y, T, n_elements, n_local_nodes, N_values, N_outer, zgp, wgp, n_nodes):
            for elem in prange(n_elements):
                element_nodes = T[elem, :]
                x_element = X[element_nodes]
                y_element = Y[element_nodes]
                M_element = np.zeros((n_local_nodes, n_local_nodes))
                for i in range(ngaus):
                    for j in range(ngaus):
                        xi = zgp[i]
                        eta = zgp[j]
                        N_gp = N_values[i, j, :]
                        dN_dxi_gp = dN_dxi_numba(xi, eta)  # Shape (n_local_nodes, 2)
                        J = np.zeros((2, 2))
                        for a in range(n_local_nodes):
                            J[0, 0] += dN_dxi_gp[a, 0] * x_element[a]
                            J[0, 1] += dN_dxi_gp[a, 0] * y_element[a]
                            J[1, 0] += dN_dxi_gp[a, 1] * x_element[a]
                            J[1, 1] += dN_dxi_gp[a, 1] * y_element[a]
                        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                        dV = wgp[i] * wgp[j] * detJ
                        M_element += N_outer[i, j, :, :] * dV
                # Assemble M_element into global matrix entries
                idx_elem = elem * n_local_nodes * n_local_nodes * 2
                idx = idx_elem
                for a in range(n_local_nodes):
                    global_a_x = element_nodes[a]
                    global_a_y = element_nodes[a] + n_nodes
                    for b in range(n_local_nodes):
                        global_b_x = element_nodes[b]
                        global_b_y = element_nodes[b] + n_nodes
                        value = M_element[a, b]
                        # For u_x component
                        I_data[idx] = global_a_x
                        J_data[idx] = global_b_x
                        V_data[idx] = value
                        idx += 1
                        # For u_y component
                        I_data[idx] = global_a_y
                        J_data[idx] = global_b_y
                        V_data[idx] = value
                        idx += 1

        # Call the Numba-compiled assembly function
        assemble_mass_matrix(
            I_data, J_data, V_data, X, Y, T, n_elements,
            n_local_nodes, N_values, N_outer, zgp, wgp, n_nodes
        )

        # Create the sparse global mass matrix
        M_global = sp.coo_matrix((V_data, (I_data, J_data)), shape=(2 * n_nodes, 2 * n_nodes)).tocsc()
        return M_global
    
    def compute_diffusion_matrix_(self):
        n_nodes = len(self.X)  # Total number of nodes
        n_elements, n_local_nodes = self.T.shape  # Number of elements and local nodes per element
        K_global = sp.lil_matrix((2 * n_nodes, 2 * n_nodes))  # Initialize global diffusion matrix

        # Print the total number of elements for reference
        print(f"Total number of elements: {n_elements}")

        for elem in range(n_elements):
            element_nodes = self.T[elem, :] - 1  # Adjusted for 0-based indexing
            x_element = self.X[element_nodes]
            y_element = self.Y[element_nodes]

            K_element = np.zeros((n_local_nodes, n_local_nodes))  # Initialize local diffusion matrix

            for i in range(self.ngaus):
                for j in range(self.ngaus):
                    xi = self.zgp[i]
                    eta = self.zgp[j]
                    N_gp = self.N(xi, eta)  # Shape functions at the Gauss point
                    dN_dxi_gp = self.dN_dxi(xi, eta)  # Derivatives of shape functions wrt xi and eta

                    # Compute the Jacobian matrix
                    J = dN_dxi_gp.T @ np.vstack((x_element, y_element)).T  # [2x2] matrix

                    # Compute the determinant and inverse of the Jacobian
                    detJ = np.linalg.det(J)
                    invJ = np.linalg.inv(J)

                    # Compute the derivative of shape functions wrt physical coordinates
                    dN_dx_gp = invJ @ dN_dxi_gp.T  # [2x4] matrix: [dN/dx, dN/dy]

                    # Compute the differential volume
                    dV = self.wgp[i] * self.wgp[j] * detJ  # Differential volume for this Gauss point

                    # Compute local diffusion matrix
                    K_element += dV * (dN_dx_gp.T @ dN_dx_gp)

            # Assemble the local matrix into the global diffusion matrix for both u_x and u_y components
            for a in range(n_local_nodes):
                for b in range(n_local_nodes):
                    # For u_x component
                    K_global[element_nodes[a], element_nodes[b]] += K_element[a, b]
                    # For u_y component
                    K_global[element_nodes[a] + n_nodes, element_nodes[b] + n_nodes] += K_element[a, b]

            # Print the progress of assembly
            if (elem + 1) % 1000 == 0 or elem == n_elements - 1:
                print(f"Diffusion Assembled element {elem + 1}/{n_elements}")

        return K_global.tocsc()  # Convert to sparse format for efficiency
    
    def compute_diffusion_matrix(self):
        n_nodes = len(self.X)
        n_elements = self.T.shape[0]
        n_local_nodes = self.n_local_nodes
        total_entries = 2 * n_elements * n_local_nodes * n_local_nodes  # For u_x and u_y components

        I_data = np.zeros(total_entries, dtype=np.int32)
        J_data = np.zeros(total_entries, dtype=np.int32)
        V_data = np.zeros(total_entries)

        X = self.X
        Y = self.Y
        T = self.T - 1  # Adjusted for 0-based indexing
        ngaus = self.ngaus
        zgp = self.zgp
        wgp = self.wgp

        # Precompute shape functions and derivatives at Gauss points
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

        # Numba-compiled function for assembly
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

                # Assemble K_element into global matrix entries
                idx_elem = elem * n_local_nodes * n_local_nodes * 2
                idx = idx_elem
                for a in range(n_local_nodes):
                    global_a_x = element_nodes[a]
                    global_a_y = element_nodes[a] + n_nodes
                    for b in range(n_local_nodes):
                        global_b_x = element_nodes[b]
                        global_b_y = element_nodes[b] + n_nodes
                        value = K_element[a, b]
                        # For u_x component
                        I_data[idx] = global_a_x
                        J_data[idx] = global_b_x
                        V_data[idx] = value
                        idx += 1
                        # For u_y component
                        I_data[idx] = global_a_y
                        J_data[idx] = global_b_y
                        V_data[idx] = value
                        idx += 1

        # Call the Numba-compiled assembly function
        assemble_diffusion_matrix(
            I_data, J_data, V_data, X, Y, T, n_elements,
            n_local_nodes, N_values, dN_dxi_values, zgp, wgp, n_nodes
        )

        # Create the sparse global diffusion matrix
        K_global = sp.coo_matrix((V_data, (I_data, J_data)), shape=(2 * n_nodes, 2 * n_nodes)).tocsc()
        return K_global


    def compute_convection_matrix(self, U_n):
        n_nodes = len(self.X)
        n_elements, n_local_nodes = self.T.shape

        # Initialize global convection matrix
        C_global = sp.lil_matrix((2 * n_nodes, 2 * n_nodes))

        for elem in range(n_elements):
            element_nodes = self.T[elem, :] - 1  # Adjusted for 0-based indexing
            x_element = self.X[element_nodes]
            y_element = self.Y[element_nodes]
            u_element = U_n[element_nodes, 0]  # u_x at nodes
            v_element = U_n[element_nodes, 1]  # u_y at nodes

            # Initialize local convection matrix
            C_element = np.zeros((2 * n_local_nodes, 2 * n_local_nodes))

            for i in range(self.ngaus):
                for j in range(self.ngaus):
                    xi = self.zgp[i]
                    eta = self.zgp[j]
                    N_gp = self.N(xi, eta)  # Shape functions at Gauss point
                    dN_dxi_gp = self.dN_dxi(xi, eta)  # Derivatives w.r.t xi, eta

                    # Compute Jacobian and its inverse
                    J = dN_dxi_gp.T @ np.vstack((x_element, y_element)).T
                    detJ = np.linalg.det(J)
                    invJ = np.linalg.inv(J)
                    dN_dx_gp = invJ @ dN_dxi_gp.T  # Derivatives w.r.t x, y

                    # Compute velocities at Gauss point
                    u_x_gp = N_gp @ u_element
                    u_y_gp = N_gp @ v_element

                    # Compute differential volume
                    dV = self.wgp[i] * self.wgp[j] * detJ

                    # Compute local convection matrix entries
                    for a in range(n_local_nodes):
                        for b in range(n_local_nodes):
                            # Indices in local convection matrix
                            ia_u = a
                            ia_v = a + n_local_nodes
                            ib_u = b
                            ib_v = b + n_local_nodes

                            # For u_x equation
                            C_element[ia_u, ib_u] -= dN_dx_gp[0, a] * (u_x_gp ** 2 / 2) * N_gp[b] * dV
                            C_element[ia_u, ib_u] -= dN_dx_gp[1, a] * u_x_gp * u_y_gp * N_gp[b] * dV

                            # For u_y equation
                            C_element[ia_v, ib_u] -= dN_dx_gp[0, a] * u_x_gp * u_y_gp * N_gp[b] * dV
                            C_element[ia_v, ib_v] -= dN_dx_gp[1, a] * (u_y_gp ** 2 / 2) * N_gp[b] * dV

            # Assemble into global convection matrix
            global_indices = np.concatenate([element_nodes, element_nodes + n_nodes])
            for local_row, global_row in enumerate(global_indices):
                for local_col, global_col in enumerate(global_indices):
                    C_global[global_row, global_col] += C_element[local_row, local_col]

            # Print the progress of assembly
            if (elem + 1) % 1000 == 0 or elem == n_elements - 1:
                print(f"Convection Assembled element {elem + 1}/{n_elements}")

        return C_global.tocsc()
    
    def compute_convection_matrix_SUPG_(self, U_n):
        n_nodes = len(self.X)
        n_elements, n_local_nodes = self.T.shape

        # Initialize global convection matrix
        C_global = sp.lil_matrix((2 * n_nodes, 2 * n_nodes))

        for elem in range(n_elements):
            element_nodes = self.T[elem, :] - 1  # Adjusted for 0-based indexing
            x_element = self.X[element_nodes]
            y_element = self.Y[element_nodes]
            u_element = U_n[element_nodes, 0]  # u_x at nodes
            v_element = U_n[element_nodes, 1]  # u_y at nodes

            # Initialize local convection matrix and SUPG terms
            C_element = np.zeros((2 * n_local_nodes, 2 * n_local_nodes))
            C_SUPG_element = np.zeros((2 * n_local_nodes, 2 * n_local_nodes))

            for i in range(self.ngaus):
                for j in range(self.ngaus):
                    xi = self.zgp[i]
                    eta = self.zgp[j]
                    N_gp = self.N(xi, eta)  # Shape functions at Gauss point
                    dN_dxi_gp = self.dN_dxi(xi, eta)  # Derivatives w.r.t xi, eta

                    # Compute Jacobian and its inverse
                    J = dN_dxi_gp.T @ np.vstack((x_element, y_element)).T
                    detJ = np.linalg.det(J)
                    invJ = np.linalg.inv(J)
                    dN_dx_gp = invJ @ dN_dxi_gp.T  # Derivatives w.r.t x, y

                    # Compute velocities at Gauss point
                    u_x_gp = N_gp @ u_element
                    u_y_gp = N_gp @ v_element
                    u_gp = np.array([u_x_gp, u_y_gp])

                    # Compute average velocity magnitude and element length
                    u_mag = np.linalg.norm(u_gp)
                    h_e = np.sqrt(2 * detJ)  # Approximate element size

                    # Compute stabilization parameter tau_e
                    tau_e = h_e / (2 * u_mag + 1e-10)  # Add small number to avoid division by zero

                    # Compute residual of momentum equation at Gauss point (simplified here)
                    # For SUPG, we need to compute the terms involving the test function derivatives
                    # Along the streamline direction

                    # Compute differential volume
                    dV = self.wgp[i] * self.wgp[j] * detJ

                    # Compute local convection matrix entries
                    for a in range(n_local_nodes):
                        for b in range(n_local_nodes):
                            # Indices in local convection matrix
                            ia_u = a
                            ia_v = a + n_local_nodes
                            ib_u = b
                            ib_v = b + n_local_nodes

                            # Standard Galerkin terms
                            # For u_x equation
                            C_element[ia_u, ib_u] += N_gp[a] * (u_x_gp * dN_dx_gp[0, b] + u_y_gp * dN_dx_gp[1, b]) * dV
                            # For u_y equation
                            C_element[ia_v, ib_v] += N_gp[a] * (u_x_gp * dN_dx_gp[0, b] + u_y_gp * dN_dx_gp[1, b]) * dV

                            # SUPG stabilization terms
                            # Compute the streamline derivative of the test function
                            grad_N_a = dN_dx_gp[:, a]
                            streamline_derivative = u_gp @ grad_N_a

                            # For u_x equation
                            C_SUPG_element[ia_u, ib_u] += tau_e * streamline_derivative * (u_gp @ dN_dx_gp[:, b]) * dV
                            # For u_y equation
                            C_SUPG_element[ia_v, ib_v] += tau_e * streamline_derivative * (u_gp @ dN_dx_gp[:, b]) * dV

            # Assemble into global convection matrix
            global_indices = np.concatenate([element_nodes, element_nodes + n_nodes])
            for local_row, global_row in enumerate(global_indices):
                for local_col, global_col in enumerate(global_indices):
                    C_global[global_row, global_col] += C_element[local_row, local_col] + C_SUPG_element[local_row, local_col]
            
            # Print the progress of assembly
            if (elem + 1) % 1000 == 0 or elem == n_elements - 1:
                print(f"Convection Assembled element {elem + 1}/{n_elements}")

        return C_global.tocsc()
    
    def compute_convection_matrix_SUPG(self, U_n):
        n_nodes = len(self.X)
        n_elements = self.T.shape[0]
        n_local_nodes = self.n_local_nodes

        total_entries = 2 * n_elements * n_local_nodes * n_local_nodes  # For u_x and u_y components
        I_data = np.zeros(total_entries, dtype=np.int32)
        J_data = np.zeros(total_entries, dtype=np.int32)
        V_data = np.zeros(total_entries)

        X = self.X
        Y = self.Y
        T = self.T - 1  # Adjusted for 0-based indexing
        ngaus = self.ngaus
        zgp = self.zgp
        wgp = self.wgp

        # Precompute shape functions at Gauss points
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

        # Numba-compiled function for assembly
        @njit(parallel=True)
        def assemble_convection_SUPG(I_data, J_data, V_data, X, Y, U_n, T, n_elements, n_local_nodes, N_values, dN_values, zgp, wgp, n_nodes):
            for elem in prange(n_elements):
                element_nodes = T[elem, :]
                x_element = X[element_nodes]
                y_element = Y[element_nodes]
                u_element = U_n[element_nodes, 0]  # u_x at nodes
                v_element = U_n[element_nodes, 1]  # u_y at nodes

                # Initialize local convection matrix and SUPG terms
                C_element = np.zeros((2 * n_local_nodes, 2 * n_local_nodes))
                C_SUPG_element = np.zeros((2 * n_local_nodes, 2 * n_local_nodes))

                for i in range(ngaus):
                    for j in range(ngaus):
                        N_gp = N_values[i, j, :]
                        dN_dxi_gp = dN_values[i, j, :, :]

                        # Compute Jacobian and its inverse
                        J = np.zeros((2, 2))
                        for a in range(n_local_nodes):
                            J[0, 0] += dN_dxi_gp[a, 0] * x_element[a]
                            J[0, 1] += dN_dxi_gp[a, 0] * y_element[a]
                            J[1, 0] += dN_dxi_gp[a, 1] * x_element[a]
                            J[1, 1] += dN_dxi_gp[a, 1] * y_element[a]

                        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                        invJ = np.linalg.inv(J)
                        dN_dx_gp = invJ @ dN_dxi_gp.T  # Derivatives w.r.t x, y

                        # Compute velocities at Gauss point
                        u_x_gp = np.dot(N_gp, u_element)
                        u_y_gp = np.dot(N_gp, v_element)
                        u_gp = np.array([u_x_gp, u_y_gp])

                        # Compute average velocity magnitude and element length
                        u_mag = np.linalg.norm(u_gp)
                        h_e = np.sqrt(2 * detJ)  # Approximate element size

                        # Compute stabilization parameter tau_e
                        tau_e = h_e / (2 * u_mag + 1e-10)  # Small number added to avoid division by zero

                        # Compute differential volume
                        dV = wgp[i] * wgp[j] * detJ

                        # Compute local convection matrix entries
                        for a in range(n_local_nodes):
                            for b in range(n_local_nodes):
                                # Indices in local convection matrix
                                ia_u = a
                                ia_v = a + n_local_nodes
                                ib_u = b
                                ib_v = b + n_local_nodes

                                # Standard Galerkin terms
                                # For u_x equation
                                C_element[ia_u, ib_u] += N_gp[a] * (u_x_gp * dN_dx_gp[0, b] + u_y_gp * dN_dx_gp[1, b]) * dV
                                # For u_y equation
                                C_element[ia_v, ib_v] += N_gp[a] * (u_x_gp * dN_dx_gp[0, b] + u_y_gp * dN_dx_gp[1, b]) * dV

                                # SUPG stabilization terms
                                # Compute the streamline derivative of the test function
                                grad_N_a = dN_dx_gp[:, a]
                                streamline_derivative = np.dot(u_gp, grad_N_a)

                                # For u_x equation
                                C_SUPG_element[ia_u, ib_u] += tau_e * streamline_derivative * np.dot(u_gp, dN_dx_gp[:, b]) * dV
                                # For u_y equation
                                C_SUPG_element[ia_v, ib_v] += tau_e * streamline_derivative * np.dot(u_gp, dN_dx_gp[:, b]) * dV

                # Assemble into global convection matrix
                idx_elem = elem * n_local_nodes * n_local_nodes * 2
                idx = idx_elem
                for a in range(n_local_nodes):
                    global_a_x = element_nodes[a]
                    global_a_y = element_nodes[a] + n_nodes
                    for b in range(n_local_nodes):
                        global_b_x = element_nodes[b]
                        global_b_y = element_nodes[b] + n_nodes
                        value = C_element[a, b] + C_SUPG_element[a, b]
                        # For u_x component
                        I_data[idx] = global_a_x
                        J_data[idx] = global_b_x
                        V_data[idx] = value
                        idx += 1
                        # For u_y component
                        I_data[idx] = global_a_y
                        J_data[idx] = global_b_y
                        V_data[idx] = value
                        idx += 1

        # Call the Numba-compiled assembly function
        assemble_convection_SUPG(
            I_data, J_data, V_data, X, Y, U_n, T, n_elements,
            n_local_nodes, N_values, dN_values, zgp, wgp, n_nodes
        )

        # Create the sparse global convection matrix
        C_global = sp.coo_matrix((V_data, (I_data, J_data)), shape=(2 * n_nodes, 2 * n_nodes)).tocsc()
        return C_global



    def compute_forcing_vector_(self, mu2):
        n_nodes = len(self.X)  # Total number of nodes
        n_elements, n_local_nodes = self.T.shape  # Number of elements and local nodes per element
        F_global = np.zeros(2 * n_nodes)  # Initialize global forcing vector

        # Print the total number of elements for reference
        print(f"Total number of elements: {n_elements}")

        for elem in range(n_elements):
            element_nodes = self.T[elem, :] - 1  # Adjusted for 0-based indexing
            x_element = self.X[element_nodes]
            y_element = self.Y[element_nodes]

            F_element = np.zeros((n_local_nodes, 2))  # Initialize local forcing vector for both u_x and u_y

            for i in range(self.ngaus):
                for j in range(self.ngaus):
                    xi = self.zgp[i]
                    eta = self.zgp[j]
                    N_gp = self.N(xi, eta)  # Shape functions at the Gauss point

                    dN_dxi_gp = self.dN_dxi(xi, eta)  # Derivatives of shape functions wrt xi and eta
                    J = dN_dxi_gp.T @ np.vstack((x_element, y_element)).T  # [2x2] matrix
                    detJ = np.linalg.det(J)

                    # Compute the physical coordinates at the Gauss point
                    x_gp = N_gp @ x_element
                    y_gp = N_gp @ y_element

                    # Compute the differential volume
                    dV = self.wgp[i] * self.wgp[j] * detJ  # Differential volume for this Gauss point

                    # Compute the forcing function at the Gauss point
                    f_x_gp = 0.02 * np.exp(mu2 * x_gp)  # Forcing term for u_x component
                    f_y_gp = 0.0  # Forcing term for u_y component (zero in this case)

                    # Update the local forcing vector
                    F_element[:, 0] += f_x_gp.flatten() * N_gp * dV  # x-component contribution
                    F_element[:, 1] += f_y_gp * N_gp * dV  # y-component contribution

            # Assemble the local vector into the global forcing vector
            for a in range(n_local_nodes):
                # For u_x component
                F_global[element_nodes[a]] += F_element[a, 0]
                # For u_y component
                F_global[element_nodes[a] + n_nodes] += F_element[a, 1]

            # Print the progress of assembly
            if (elem + 1) % 1000 == 0 or elem == n_elements - 1:
                print(f"Forcing Assembled element {elem + 1}/{n_elements}")

        return F_global  # Return the global forcing vector of size (2 * n_nodes)
    
    def compute_forcing_vector(self, mu2):
        n_nodes = len(self.X)
        n_elements = self.T.shape[0]
        n_local_nodes = self.n_local_nodes

        F_global = np.zeros(2 * n_nodes)  # Global forcing vector for u_x and u_y

        X = self.X
        Y = self.Y
        T = self.T - 1  # Adjusted for 0-based indexing
        ngaus = self.ngaus
        zgp = self.zgp
        wgp = self.wgp

        # Precompute shape functions at Gauss points
        N_values = np.zeros((ngaus, ngaus, n_local_nodes))
        for i in range(ngaus):
            for j in range(ngaus):
                xi = zgp[i]
                eta = zgp[j]
                N_gp = N_numba(xi, eta)
                N_values[i, j, :] = N_gp

        # Numba-compiled function for assembly
        @njit(parallel=True)
        def assemble_forcing_vector(F_global, X, Y, T, n_elements, n_local_nodes, N_values, zgp, wgp, mu2, n_nodes):
            for elem in prange(n_elements):
                element_nodes = T[elem, :]
                x_element = X[element_nodes]
                y_element = Y[element_nodes]
                F_element = np.zeros((n_local_nodes, 2))  # Local forcing vector for u_x and u_y

                for i in range(ngaus):
                    for j in range(ngaus):
                        N_gp = N_values[i, j, :]
                        dN_dxi_gp = dN_dxi_numba(zgp[i], zgp[j])

                        # Compute Jacobian and its determinant
                        J = np.zeros((2, 2))
                        for a in range(n_local_nodes):
                            J[0, 0] += dN_dxi_gp[a, 0] * x_element[a]
                            J[0, 1] += dN_dxi_gp[a, 0] * y_element[a]
                            J[1, 0] += dN_dxi_gp[a, 1] * x_element[a]
                            J[1, 1] += dN_dxi_gp[a, 1] * y_element[a]

                        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                        dV = wgp[i] * wgp[j] * detJ

                        # Compute the physical coordinates at the Gauss point
                        x_gp = np.dot(N_gp, x_element)

                        # Compute the forcing function at the Gauss point
                        f_x_gp = 0.02 * np.exp(mu2 * x_gp)  # Forcing term for u_x component

                        # Update local forcing vector
                        F_element[:, 0] += f_x_gp * N_gp * dV  # Forcing in u_x direction (non-zero)
                        F_element[:, 1] += 0.0 * N_gp * dV     # Forcing in u_y direction (zero)

                # Assemble local forcing into global vector
                for a in range(n_local_nodes):
                    F_global[element_nodes[a]] += F_element[a, 0]            # For u_x component
                    F_global[element_nodes[a] + n_nodes] += F_element[a, 1]  # For u_y component

        # Call the Numba-compiled assembly function
        assemble_forcing_vector(F_global, X, Y, T, n_elements, n_local_nodes, N_values, zgp, wgp, mu2, n_nodes)

        return F_global  # Return the global forcing vector

    
    def compute_residual(self, U_new_flat, U_n_flat, At, M, E, K, F):
        n_nodes = len(self.X)
        # Reshape U_new_flat to get U_new
        U_new = np.zeros((n_nodes, 2))
        U_new[:, 0] = U_new_flat[:n_nodes]       # u_x components
        U_new[:, 1] = U_new_flat[n_nodes:]       # u_y components

        # Compute convection matrix at current U_new
        start = time.time()
        C = self.compute_convection_matrix_SUPG(U_new)
        end = time.time()
        print(f"Time for new mass matrix assembling: {end-start}")

        # Assemble global system matrix A
        A = M + At * (C + E * K)

        # Assemble right-hand side vector b
        b = M @ U_n_flat + At * F

        # Compute residual R = A * U_new_flat - b
        R = A @ U_new_flat - b

        return R, A


    def fom_burgers_2d(self, At, nTimeSteps, u0, mu1, E, mu2):
        n_nodes = len(self.X)

        # Allocate memory for the solution matrix
        U = np.zeros((n_nodes, nTimeSteps + 1, 2))  # 2 components: u_x and u_y
        U[:, 0, :] = u0  # Set initial condition

        # Compute the mass and diffusion matrices (these are constant)
        start = time.time()
        M = self.compute_mass_matrix()
        end = time.time()
        print(f"Time for new mass matrix assembling: {end-start}")

        start = time.time()
        K = self.compute_diffusion_matrix()
        end = time.time()
        print(f"Time for new diff matrix assembling: {end-start}")

        # Identify the nodes at the left boundary (x = 0)
        tolerance = 1e-8
        left_boundary_nodes = np.where(np.isclose(self.X, 0.0, atol=tolerance))[0]
        boundary_dofs_u_x = left_boundary_nodes  # DOFs for u_x at x = 0

        # Precompute the forcing vector (if it's constant)
        start = time.time()
        F = self.compute_forcing_vector(mu2)  # Returns a flat vector of size 2n_nodes
        end = time.time()
        print(f"Time for new forcing vector assembling: {end-start}")

        # Time-stepping loop
        for n in range(nTimeSteps):
            print(f"Time Step: {n+1}/{nTimeSteps}. Time: {(n+1) * At}")
            U_n = U[:, n, :]  # Current solution at time step n
            U_n_flat = np.concatenate([U_n[:, 0], U_n[:, 1]])  # Flattened solution vector [u_x; u_y]
            U_new_flat = U_n_flat.copy()  # Initial guess for U_new_flat

            error_U = 1
            k = 0
            max_iter = 15  # Increase maximum iterations
            tol = 1e-8

            while (error_U > tol) and (k < max_iter):  # Nonlinear solver loop
                print(f"  Iteration: {k}, Error: {error_U:.2e}")
                
                # Compute residual and system matrix
                R, A = self.compute_residual(U_new_flat, U_n_flat, At, M, E, K, F)
                
                # Apply boundary conditions to R and A
                A = A.tolil()
                for node in boundary_dofs_u_x:
                    dof_u_x = node  # u_x DOF
                    A[dof_u_x, :] = 0
                    A[dof_u_x, dof_u_x] = 1.0
                    R[dof_u_x] = 0.0  # Enforce residual zero at Dirichlet node
                A = A.tocsc()
                
                # Solve for the update delta_U
                delta_U = spla.spsolve(A, -R)
                
                # Update the solution
                U_new_flat += delta_U  # U_new_flat = U_new_flat + delta_U
                
                # Enforce boundary conditions
                U_new_flat[boundary_dofs_u_x] = mu1  # u_x = mu1 at x = 0
                
                # Compute the error
                error_U = np.linalg.norm(delta_U) / np.linalg.norm(U_new_flat)
                
                k += 1

            # Reshape U_new_flat to get U_new
            U_new = np.zeros((n_nodes, 2))
            U_new[:, 0] = U_new_flat[:n_nodes]       # u_x components
            U_new[:, 1] = U_new_flat[n_nodes:]       # u_y components

            if k == max_iter:
                print("  Warning: Nonlinear solver did not converge within maximum iterations")

            # Store the converged solution for this time step
            U[:, n + 1, :] = U_new

            # Save the solution up to the current time step
            np.save('U_FOM.npy', U[:, :n + 2, :])

        return U





if __name__ == "__main__":
    # Domain and mesh
    a, b = 0, 100
    nx, ny = 250, 250  # Adjusted for quicker testing; increase for finer mesh
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

    # Save X and Y for plotting later
    np.save('X.npy', X)
    np.save('Y.npy', Y)

    # Initial conditions
    u0 = np.ones((len(X), 2))  # Initialize for both u_x and u_y

    # Time discretization
    Tf, At = 35.0, 0.05  # Adjusted for quicker testing; increase Tf for longer simulation
    nTimeSteps = int(Tf / At)
    E = 1.0

    # Parameters
    mu1, mu2 = 4.77, 0.02

    # Create an instance of the FEMBurgers2D class
    fem_burgers_2d = FEMBurgers2D(X, Y, T)

    # Run the simulation
    U_FOM = fem_burgers_2d.fom_burgers_2d(At, nTimeSteps, u0, mu1, E, mu2)

    # Optionally, generate animations after simulation
    # plot_3d_surface_animation(U_FOM, X, Y, filename='surface_animation.gif')
    # plot_2d_cuts_animation(U_FOM, X, Y, line='x', fixed_value=50.0, filename='cuts_animation.gif')


