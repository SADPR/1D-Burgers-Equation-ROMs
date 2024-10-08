import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import forcing_vector_parallel  # C++ bindings for forcing vector
import mass_matrix_parallel  # C++ bindings for mass matrix
import diffusion_matrix_parallel  # C++ bindings for diffusion matrix
import convection_matrix_supg_parallel  # C++ bindings for convection matrix
import boundary_conditions_parallel
import sparse_solver_parallel

class FEMBurgers2D:
    def __init__(self, X, Y, T):
        self.X = X
        self.Y = Y
        self.T = T
        self.n_local_nodes = self.T.shape[1]  # Number of local nodes per element
        self.ngaus = 2
        self.zgp = np.array([-np.sqrt(3) / 3, np.sqrt(3) / 3])  # Gauss points
        self.wgp = np.array([1, 1])  # Weights

        # Precompute shape functions and their derivatives at Gauss points (N_values and dN_dxi_values)
        self.N_values, self.dN_dxi_values = self._precompute_shape_functions()

    def _precompute_shape_functions(self):
        """ Precompute shape functions and their derivatives at Gauss points """
        n_local_nodes = self.n_local_nodes

        # Define shape functions (standard Python version)
        def N_python(xi, eta):
            return 0.25 * np.array([
                (1 - xi) * (1 - eta),
                (1 + xi) * (1 - eta),
                (1 + xi) * (1 + eta),
                (1 - xi) * (1 + eta)
            ])

        # Define derivatives of shape functions wrt xi and eta
        def dN_dxi_python(xi, eta):
            return 0.25 * np.array([[-(1 - eta), -(1 - xi)], 
                                    [(1 - eta), -(1 + xi)], 
                                    [(1 + eta), (1 + xi)], 
                                    [-(1 + eta), (1 - xi)]])

        # Precompute N_values and dN_dxi_values at Gauss points
        N_values = np.zeros((self.ngaus, self.ngaus, n_local_nodes))
        dN_dxi_values = np.zeros((self.ngaus, self.ngaus, n_local_nodes, 2))  # Shape (gauss points, local nodes, xi/eta derivatives)

        for i in range(self.ngaus):
            for j in range(self.ngaus):
                xi = self.zgp[i]
                eta = self.zgp[j]
                N_values[i, j, :] = N_python(xi, eta)
                dN_dxi_values[i, j, :, :] = dN_dxi_python(xi, eta)

        return N_values, dN_dxi_values

    def compute_mass_matrix(self):
        n_nodes = len(self.X)
        n_elements = self.T.shape[0]
        n_local_nodes = self.n_local_nodes

        # Call C++ function to compute the mass matrix, passing precomputed N_values
        M_global = mass_matrix_parallel.assemble_mass_matrix(self.X, self.Y, self.T, self.N_values, self.wgp, self.zgp, n_nodes, n_elements, n_local_nodes)
        return sp.csc_matrix(M_global)

    def compute_diffusion_matrix(self):
        n_nodes = len(self.X)
        n_elements = self.T.shape[0]
        n_local_nodes = self.n_local_nodes

        # Call C++ function to compute the diffusion matrix, passing precomputed N_values and dN_dxi_values
        K_global = diffusion_matrix_parallel.assemble_diffusion_matrix(self.X, self.Y, self.T, self.N_values, self.dN_dxi_values, self.wgp, self.zgp, n_nodes, n_elements, n_local_nodes)
        return sp.csc_matrix(K_global)

    def compute_convection_matrix_SUPG(self, U_n):
        n_nodes = len(self.X)
        n_elements = self.T.shape[0]
        n_local_nodes = self.n_local_nodes

        # Call C++ function to compute the convection matrix with SUPG
        C_global = convection_matrix_supg_parallel.assemble_convection_matrix_supg(self.X, self.Y, self.T, self.N_values, self.dN_dxi_values, U_n, self.wgp, self.zgp, n_nodes, n_elements, n_local_nodes)
        return sp.csc_matrix(C_global)

    def compute_forcing_vector(self, mu2):
        n_nodes = len(self.X)
        n_elements = self.T.shape[0]
        n_local_nodes = self.n_local_nodes

        # Call C++ function to compute the forcing vector
        F_global = forcing_vector_parallel.assemble_forcing_vector(self.X, self.Y, self.T, self.N_values, self.wgp, self.zgp, mu2, n_nodes, n_elements, n_local_nodes)
        return F_global

    def compute_residual(self, U_new_flat, U_n_flat, At, M, E, K, F):
        n_nodes = len(self.X)
        # Reshape U_new_flat to get U_new
        U_new = np.zeros((n_nodes, 2))
        U_new[:, 0] = U_new_flat[:n_nodes]       # u_x components
        U_new[:, 1] = U_new_flat[n_nodes:]       # u_y components

        # Compute convection matrix at current U_new
        C = self.compute_convection_matrix_SUPG(U_new)

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

        # Measure time for mass matrix assembly
        start_time = time.time()
        M = self.compute_mass_matrix()
        print(f"Mass matrix assembly time: {time.time() - start_time:.6f} seconds")

        # Measure time for diffusion matrix assembly
        start_time = time.time()
        K = self.compute_diffusion_matrix()
        print(f"Diffusion matrix assembly time: {time.time() - start_time:.6f} seconds")

        # Identify the nodes at the left boundary (x = 0)
        tolerance = 1e-8
        left_boundary_nodes = np.where(np.isclose(self.X, 0.0, atol=tolerance))[0]
        boundary_dofs_u_x = left_boundary_nodes  # DOFs for u_x at x = 0

        # Precompute the forcing vector (if it's constant)
        start_time = time.time()
        F = self.compute_forcing_vector(mu2)
        print(f"Forcing vector assembly time: {time.time() - start_time:.6f} seconds")

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

                # Measure time for residual and system matrix computation
                start_time = time.time()
                R, A = self.compute_residual(U_new_flat, U_n_flat, At, M, E, K, F)
                print(f"    Residual and system matrix computation time: {time.time() - start_time:.6f} seconds")

                # Apply boundary conditions to R and A
                start_time = time.time()
                # Before calling apply_boundary_conditions
                U_current_flat = U_new_flat.copy()  # This is U_current in the current Newton iteration

                # Call the C++ function with the current solution and boundary value mu1
                A, R = boundary_conditions_parallel.apply_boundary_conditions(A, R, U_current_flat, boundary_dofs_u_x, mu1)

                # # Call the C++ function
                # A, R = boundary_conditions_parallel.apply_boundary_conditions(A, R, boundary_dofs_u_x)
                print(f"    Boundary conditions application time: {time.time() - start_time:.6f} seconds")

                # Measure time for solving the system
                start_time = time.time()
                # delta_U = spla.spsolve(A, -R)
                # Solve the linear system using the Eigen-based C++ solver
                delta_U = sparse_solver_parallel.solve_sparse_system(A, R)
                print(f"    Solver time: {time.time() - start_time:.6f} seconds")

                # Update the solution
                U_new_flat += delta_U  # U_new_flat = U_new_flat + delta_U

                # Enforce boundary conditions
                # U_new_flat[boundary_dofs_u_x] = mu1  # u_x = mu1 at x = 0

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
    nx, ny = 250, 250  # Adjusted for quicker testing
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
    Tf, At = 1.0, 0.05  # Adjusted for quicker testing
    nTimeSteps = int(Tf / At)
    E = 0.2

    # Parameters
    mu1, mu2 = 4.3, 0.021

    # Create an instance of the FEMBurgers2D class
    fem_burgers_2d = FEMBurgers2D(X, Y, T)

    # Run the simulation
    U_FOM = fem_burgers_2d.fom_burgers_2d(At, nTimeSteps, u0, mu1, E, mu2)
