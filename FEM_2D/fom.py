import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt

def plot_2d_cuts(U, X, Y, line='x', fixed_value=50.1):
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

from mpl_toolkits.mplot3d import Axes3D

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


class FEMBurgers2D:
    def __init__(self, X, Y, T):
        self.X = X
        self.Y = Y
        self.T = T
        self.ngaus = 2
        # 2D Gauss points and weights for quadrature on [-1, 1]
        self.zgp = np.array([-np.sqrt(3) / 3, np.sqrt(3) / 3])
        self.wgp = np.array([1, 1])
        self.N = lambda xi, eta: 0.25 * np.array([(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)])
        self.dN_dxi = lambda xi, eta: 0.25 * np.array([[-(1-eta), -(1-xi)], [(1-eta), -(1+xi)], [(1+eta), (1+xi)], [-(1+eta), (1-xi)]])

    def compute_mass_matrix(self):
        n_nodes = len(self.X)  # Number of nodes
        n_elements, n_local_nodes = self.T.shape  # Number of elements and nodes per element
        M_global = sp.lil_matrix((n_nodes, n_nodes))  # Initialize global mass matrix

        # Print the total number of elements for reference
        print(f"Total number of elements: {n_elements}")

        for elem in range(n_elements):
            element_nodes = self.T[elem, :] - 1  # Adjusted for 0-based indexing
            x_element = self.X[element_nodes].reshape(-1, 1)
            y_element = self.Y[element_nodes].reshape(-1, 1)
            
            M_element = np.zeros((n_local_nodes, n_local_nodes))  # Initialize local mass matrix

            for i in range(self.ngaus):
                for j in range(self.ngaus):
                    xi = self.zgp[i]
                    eta = self.zgp[j]
                    N_gp = self.N(xi, eta)  # Shape functions at the Gauss point

                    dN_dxi_gp = self.dN_dxi(xi, eta)  # Shape function derivatives wrt reference coordinates
                    J = np.dot(dN_dxi_gp.T, np.c_[x_element, y_element])  # Jacobian

                    dV = self.wgp[i] * self.wgp[j] * np.linalg.det(J)  # Differential volume
                    
                    M_element += np.outer(N_gp, N_gp) * dV

            # Assemble the local matrix into the global mass matrix
            for i in range(n_local_nodes):
                for j in range(n_local_nodes):
                    M_global[element_nodes[i], element_nodes[j]] += M_element[i, j]

            # Print the progress of assembly
            if (elem + 1) % 1000 == 0 or elem == n_elements - 1:
                print(f"Mass Assembled element {elem + 1}/{n_elements}")

        return M_global.tocsc()  # Convert to sparse format for efficiency

    
    def compute_diffusion_matrix(self):
        n_nodes = len(self.X)  # Total number of nodes
        n_elements, n_local_nodes = self.T.shape  # Number of elements and local nodes per element
        K_global = sp.lil_matrix((n_nodes, n_nodes))  # Initialize global diffusion matrix

        # Print the total number of elements for reference
        print(f"Total number of elements: {n_elements}")

        for elem in range(n_elements):
            element_nodes = self.T[elem, :] - 1  # Adjusted for 0-based indexing
            x_element = self.X[element_nodes].reshape(-1, 1)  # x-coordinates of the element's nodes
            y_element = self.Y[element_nodes].reshape(-1, 1)  # y-coordinates of the element's nodes

            K_element = np.zeros((n_local_nodes, n_local_nodes))  # Initialize local diffusion matrix

            for i in range(self.ngaus):
                for j in range(self.ngaus):
                    xi = self.zgp[i]
                    eta = self.zgp[j]
                    N_gp = self.N(xi, eta)  # Shape functions at the Gauss point
                    dN_dxi_gp = self.dN_dxi(xi, eta)  # Derivatives of shape functions wrt xi and eta

                    # Compute the Jacobian matrix
                    J = np.dot(dN_dxi_gp.T, np.c_[x_element, y_element])  # [2x2] matrix

                    # Compute the determinant of the Jacobian
                    detJ = np.linalg.det(J)

                    # Compute the inverse of the Jacobian matrix
                    invJ = np.linalg.inv(J)

                    # Compute the derivative of shape functions wrt physical coordinates
                    dN_dx_gp = np.dot(invJ, dN_dxi_gp.T)  # [2x4] matrix: [dN/dx, dN/dy]

                    # Compute the differential volume
                    dV = self.wgp[i] * self.wgp[j] * detJ  # Differential volume for this Gauss point

                    # Compute local diffusion matrix
                    K_element += dV * (dN_dx_gp.T @ dN_dx_gp)

            # Assemble the local matrix into the global diffusion matrix
            for i in range(n_local_nodes):
                for j in range(n_local_nodes):
                    K_global[element_nodes[i], element_nodes[j]] += K_element[i, j]

            # Print the progress of assembly
            if (elem + 1) % 1000 == 0 or elem == n_elements - 1:
                print(f"Diffusion Assembled element {elem + 1}/{n_elements}")

        return K_global.tocsc()  # Convert to sparse format for efficiency


    def compute_convection_matrix(self, U_n):
        n_nodes = len(self.X)  # Total number of nodes
        n_elements, n_local_nodes = self.T.shape  # Number of elements and local nodes per element

        # Initialize global convection matrices for u_x and u_y components
        C_global_x = sp.lil_matrix((n_nodes, n_nodes))  # Convection matrix for u_x component
        C_global_y = sp.lil_matrix((n_nodes, n_nodes))  # Convection matrix for u_y component

        # Print the total number of elements for reference
        print(f"Total number of elements: {n_elements}")

        for elem in range(n_elements):
            element_nodes = self.T[elem, :] - 1  # Adjusted for 0-based indexing
            x_element = self.X[element_nodes].reshape(-1, 1)  # x-coordinates of the element's nodes
            y_element = self.Y[element_nodes].reshape(-1, 1)  # y-coordinates of the element's nodes
            u_element = U_n[element_nodes, 0]  # x-component of velocity at the element's nodes
            v_element = U_n[element_nodes, 1]  # y-component of velocity at the element's nodes

            # Initialize local convection matrices for u_x and u_y components
            C_element_x = np.zeros((n_local_nodes, n_local_nodes))  # Local convection matrix for u_x
            C_element_y = np.zeros((n_local_nodes, n_local_nodes))  # Local convection matrix for u_y

            for i in range(self.ngaus):
                for j in range(self.ngaus):
                    xi = self.zgp[i]
                    eta = self.zgp[j]
                    N_gp = self.N(xi, eta).reshape(-1, 1)  # Shape functions at the Gauss point (column vector)
                    dN_dxi_gp = self.dN_dxi(xi, eta)  # Derivatives of shape functions wrt xi and eta

                    # Compute the Jacobian matrix
                    J = np.dot(dN_dxi_gp.T, np.c_[x_element, y_element])  # [2x2] matrix

                    # Compute the determinant of the Jacobian
                    detJ = np.linalg.det(J)

                    # Compute the inverse of the Jacobian matrix
                    invJ = np.linalg.inv(J)

                    # Compute the derivative of shape functions wrt physical coordinates
                    dN_dx_gp = np.dot(invJ, dN_dxi_gp.T)  # [2x4] matrix: [dN/dx, dN/dy]

                    # Compute the velocity at the Gauss point
                    u_gp = np.dot(N_gp.T, u_element)  # u_x velocity at the Gauss point
                    v_gp = np.dot(N_gp.T, v_element)  # u_y velocity at the Gauss point

                    # Compute the differential volume
                    dV = self.wgp[i] * self.wgp[j] * detJ  # Differential volume for this Gauss point

                    # Compute local convection matrix for u_x component
                    # The term (u * du/dx + v * du/dy) is a scalar, and N_gp * dV is a (4, 1) vector
                    C_element_x += (u_gp * dN_dx_gp[0, :] + v_gp * dN_dx_gp[1, :])[:, None] @ N_gp.T * dV

                    # Compute local convection matrix for u_y component
                    # The term (u * dv/dx + v * dv/dy) is a scalar, and N_gp * dV is a (4, 1) vector
                    C_element_y += (u_gp * dN_dx_gp[1, :] + v_gp * dN_dx_gp[0, :])[:, None] @ N_gp.T * dV

            # Assemble the local matrices into the global convection matrices
            for i in range(n_local_nodes):
                for j in range(n_local_nodes):
                    C_global_x[element_nodes[i], element_nodes[j]] += C_element_x[i, j]
                    C_global_y[element_nodes[i], element_nodes[j]] += C_element_y[i, j]

            # Print the progress of assembly
            if (elem + 1) % 1000 == 0 or elem == n_elements - 1:
                print(f"Convection Assembled element {elem + 1}/{n_elements}")

        # Convert to sparse format for efficiency
        return C_global_x.tocsc(), C_global_y.tocsc()


    def compute_forcing_vector(self, mu2):
        n_nodes = len(self.X)  # Total number of nodes
        n_elements, n_local_nodes = self.T.shape  # Number of elements and local nodes per element
        F_global = np.zeros((n_nodes, 2))  # Initialize global forcing vector for both u_x and u_y

        # Print the total number of elements for reference
        print(f"Total number of elements: {n_elements}")

        for elem in range(n_elements):
            element_nodes = self.T[elem, :] - 1  # Adjusted for 0-based indexing
            x_element = self.X[element_nodes].reshape(-1, 1)  # x-coordinates of the element's nodes
            y_element = self.Y[element_nodes].reshape(-1, 1)  # y-coordinates of the element's nodes

            F_element = np.zeros((n_local_nodes, 2))  # Initialize local forcing vector for both u_x and u_y

            for i in range(self.ngaus):
                for j in range(self.ngaus):
                    xi = self.zgp[i]
                    eta = self.zgp[j]
                    N_gp = self.N(xi, eta)  # Shape functions at the Gauss point
                    dN_dxi_gp = self.dN_dxi(xi, eta)  # Derivatives of shape functions wrt xi and eta

                    # Compute the Jacobian matrix
                    J = np.dot(dN_dxi_gp.T, np.c_[x_element, y_element])  # [2x2] matrix

                    # Compute the determinant of the Jacobian
                    detJ = np.linalg.det(J)

                    # Compute the physical coordinates at the Gauss point
                    x_gp = np.dot(N_gp, x_element)
                    y_gp = np.dot(N_gp, y_element)

                    # Compute the differential volume
                    dV = self.wgp[i] * self.wgp[j] * detJ  # Differential volume for this Gauss point

                    # Compute the forcing function at the Gauss point
                    f_x_gp = 0.02 * np.exp(mu2 * x_gp)  # Forcing term for u_x component
                    f_y_gp = 0.0  # Forcing term for u_y component (zero in this case)

                    # Update the local forcing vector
                    F_element[:, 0] += f_x_gp * N_gp * dV  # x-component contribution
                    F_element[:, 1] += f_y_gp * N_gp * dV  # y-component contribution

            # Assemble the local vector into the global forcing vector
            for i in range(n_local_nodes):
                F_global[element_nodes[i], 0] += F_element[i, 0]  # Assemble for u_x component
                F_global[element_nodes[i], 1] += F_element[i, 1]  # Assemble for u_y component

            # Print the progress of assembly
            if (elem + 1) % 1000 == 0 or elem == n_elements - 1:
                print(f"Forcing Assembled element {elem + 1}/{n_elements}")

        return F_global


    def fom_burgers_2d(self, At, nTimeSteps, u0, mu1, E, mu2):
        n_nodes = len(self.X)

        # Allocate memory for the solution matrix
        U = np.zeros((n_nodes, nTimeSteps + 1, 2))  # 2 components: u_x and u_y
        U[:, 0, :] = u0  # Set initial condition

        # Compute the mass and diffusion matrices (these are constant)
        M = self.compute_mass_matrix()
        K = self.compute_diffusion_matrix()

        # Identify the nodes at the left boundary (x = 0)
        left_boundary_nodes = np.where(self.X == 0)[0]

        # Time-stepping loop
        for n in range(nTimeSteps):
            print(f"Time Step: {n}. Time: {n * At}")
            U0 = U[:, n, :].copy()  # Current solution at time step n
            error_U = 1
            k = 0

            while (error_U > 1e-6) and (k < 20):  # Iterative solver loop
                print(f"Iteration: {k}, Error: {error_U}")

                # Compute convection matrix based on current solution guess
                C_x, C_y = self.compute_convection_matrix(U0)  # Extract u_x and u_y convection matrices

                # Compute forcing vector
                F = self.compute_forcing_vector(mu2)

                # Form the system matrix A and right-hand side b for the u_x component
                A_x = M + At * (C_x + E * K)
                # Form the system matrix A and right-hand side b for the u_y component
                A_y = M + At * (C_y + E * K)

                # Apply Dirichlet boundary condition at x = 0 for u_x component
                for node in left_boundary_nodes:
                    A_x[node, :] = 0
                    A_x[node, node] = 1

                # Right-hand side vector for u_x component
                b_x = M @ U[:, n, 0] + At * F[:, 0]
                b_x[left_boundary_nodes] = mu1  # Apply Dirichlet condition for u_x

                # Solve the linear system for u_x component
                Ux_new = spla.spsolve(A_x, b_x)

                # Apply Dirichlet boundary condition for u_y component
                # for node in left_boundary_nodes:
                #     A_y[node, :] = 0
                #     A_y[node, node] = 1

                # Right-hand side vector for u_y component
                b_y = M @ U[:, n, 1] + At * F[:, 1]
                b_y[left_boundary_nodes] = mu1  # Apply Dirichlet condition for u_y

                # Solve the linear system for u_y component
                Uy_new = spla.spsolve(A_y, b_y)

                # Update the solution vector
                U1 = np.vstack((Ux_new, Uy_new)).T  # Combine u_x and u_y components

                # Compute the error to check for convergence
                error_U = np.linalg.norm(U1 - U0) / np.linalg.norm(U1)

                # Update the guess for the next iteration
                U0 = U1
                k += 1

            # Plot the current converged solution
            plot_3d_surface(U1, self.X, self.Y)
            plot_2d_cuts(U1, self.X, self.Y, line='x', fixed_value=50.0)

            # Store the converged solution for this time step
            U[:, n + 1, :] = U1

        return U



if __name__ == "__main__":
    # Domain and mesh
    a, b = 0, 100
    nx, ny = 64, 64#256, 256
    x = np.linspace(a, b, nx + 1)
    y = np.linspace(a, b, ny + 1)
    X, Y = np.meshgrid(x, y)
    X, Y = X.flatten(), Y.flatten()
    T = np.array([[(i + j * (nx + 1)) + np.array([0, 1, nx + 2, nx + 1])] for j in range(ny) for i in range(nx)])
    T = T.reshape(-1, 4)

    # Initial conditions
    u0 = np.ones((len(X), 2))  # Initialize for both u_x and u_y

    # Time discretization
    Tf, At = 35, 0.07
    nTimeSteps = int(Tf / At)
    E = 0.01

    # Parameters
    mu1, mu2 = 4.77, 0.0200

    # Create an instance of the FEMBurgers2D class
    fem_burgers_2d = FEMBurgers2D(X, Y, T)

    # Run the simulation
    U_FOM = fem_burgers_2d.fom_burgers_2d(At, nTimeSteps, u0, mu1, E, mu2)

    # Save the results and create an animation as required
