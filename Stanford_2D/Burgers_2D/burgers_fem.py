import numpy as np
import scipy.sparse as sp
import time
import forcing_vector_parallel
import mass_matrix_parallel
import diffusion_matrix_parallel
import convection_matrix_supg_parallel
import boundary_conditions_parallel
import sparse_solver_parallel
import eigen_sparse_dense_operations
import mkl_sparse_dense_operations
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

class FEMBurgers2D:
    def __init__(self, X, Y, T):
        self.X = X
        self.Y = Y
        self.T = T
        self.n_local_nodes = self.T.shape[1]
        self.ngaus = 2
        self.zgp = np.array([-np.sqrt(3) / 3, np.sqrt(3) / 3]) 
        self.wgp = np.array([1, 1])

        self.N_values, self.dN_dxi_values = self._precompute_shape_functions()

    def _precompute_shape_functions(self):
        n_local_nodes = self.n_local_nodes

        def N_python(xi, eta):
            return 0.25 * np.array([
                (1 - xi) * (1 - eta),
                (1 + xi) * (1 - eta),
                (1 + xi) * (1 + eta),
                (1 - xi) * (1 + eta)
            ])

        def dN_dxi_python(xi, eta):
            return 0.25 * np.array([[-(1 - eta), -(1 - xi)], 
                                    [(1 - eta), -(1 + xi)], 
                                    [(1 + eta), (1 + xi)], 
                                    [-(1 + eta), (1 - xi)]])
        
        N_values = np.zeros((self.ngaus, self.ngaus, n_local_nodes))
        dN_dxi_values = np.zeros((self.ngaus, self.ngaus, n_local_nodes, 2))

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
        return sp.csr_matrix(M_global)

    def compute_diffusion_matrix(self):
        n_nodes = len(self.X)
        n_elements = self.T.shape[0]
        n_local_nodes = self.n_local_nodes

        # Call C++ function to compute the diffusion matrix, passing precomputed N_values and dN_dxi_values
        K_global = diffusion_matrix_parallel.assemble_diffusion_matrix(self.X, self.Y, self.T, self.N_values, self.dN_dxi_values, self.wgp, self.zgp, n_nodes, n_elements, n_local_nodes)
        return sp.csr_matrix(K_global)

    def compute_convection_matrix_SUPG(self, U_n):
        n_nodes = len(self.X)
        n_elements = self.T.shape[0]
        n_local_nodes = self.n_local_nodes

        # Call C++ function to compute the convection matrix with SUPG
        C_global = convection_matrix_supg_parallel.assemble_convection_matrix_supg(self.X, self.Y, self.T, self.N_values, self.dN_dxi_values, U_n, self.wgp, self.zgp, n_nodes, n_elements, n_local_nodes)
        return sp.csr_matrix(C_global)

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
        total_dofs = n_nodes * 2  # Total degrees of freedom (u_x and u_y components for each node)

        # Flatten the initial condition
        u0_flat = np.concatenate([u0[:, 0], u0[:, 1]])  # Shape: (total_dofs,)

        # Allocate memory for the flattened solution matrix
        U_flat = np.zeros((nTimeSteps + 1, total_dofs))  # Each row is the solution at a time step
        U_flat[0, :] = u0_flat  # Set initial condition

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

        # Map boundary node indices to flattened DOF indices for u_x component
        boundary_dof_indices = boundary_dofs_u_x  # Since u_x components are from index 0 to n_nodes - 1

        # Precompute the forcing vector (if it's constant)
        start_time = time.time()
        F = self.compute_forcing_vector(mu2)
        print(f"Forcing vector assembly time: {time.time() - start_time:.6f} seconds")

        # Time-stepping loop
        for n in range(nTimeSteps):
            print(f"Time Step: {n+1}/{nTimeSteps}. Time: {(n+1) * At}")

            U_n_flat = U_flat[n, :]  # Current solution at time step n
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
                U_current_flat = U_new_flat.copy()  # This is U_current in the current Newton iteration

                # Apply boundary conditions (assuming boundary_conditions_parallel is updated for flattened vectors)
                A, R = boundary_conditions_parallel.apply_boundary_conditions(A, R, U_current_flat, boundary_dof_indices, mu1)
                print(f"    Boundary conditions application time: {time.time() - start_time:.6f} seconds")

                # Measure time for solving the system
                start_time = time.time()
                # Solve the linear system using your preferred solver
                delta_U = sparse_solver_parallel.solve_sparse_system(A, R)
                print(f"    Solver time: {time.time() - start_time:.6f} seconds")

                # Update the solution
                U_new_flat += delta_U  # U_new_flat = U_new_flat + delta_U

                # Enforce boundary conditions explicitly in the solution vector
                U_new_flat[boundary_dof_indices] = mu1  # u_x = mu1 at x = 0

                # Compute the error
                error_U = np.linalg.norm(delta_U) / (np.linalg.norm(U_new_flat) + 1e-12)

                k += 1

            if k == max_iter:
                print("  Warning: Nonlinear solver did not converge within maximum iterations")

            # Store the converged solution for this time step
            U_flat[n + 1, :] = U_new_flat

            # Optional: Save the solution up to the current time step
            np.save('U_FOM.npy', U_flat[:n + 2, :])

        # Return the flattened solution matrix
        return U_flat


    def pod_prom_burgers_(self, At, nTimeSteps, u0, mu1, E, mu2, Phi, X, Y, projection="LSPG", plot_interval=1):
        n_nodes = len(self.X)
        total_dofs = n_nodes * 2  # Total degrees of freedom (u_x and u_y components for each node)

        # Flatten the initial condition
        u0_flat = np.concatenate([u0[:, 0], u0[:, 1]])  # Shape: (total_dofs,)

        # Allocate memory for the flattened solution matrix
        U_PROM_flat = np.zeros((nTimeSteps + 1, total_dofs))
        U_PROM_flat[0, :] = u0_flat  # Set initial condition

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

            U_n_flat = U_PROM_flat[n, :]  # Current solution at time step n
            U_new_flat = U_n_flat.copy()  # Initial guess for U_new_flat

            error_U = 1
            k = 0
            max_iter = 15  # Maximum iterations for the nonlinear solver
            tol = 1e-8

            while (error_U > tol) and (k < max_iter):  # Nonlinear solver loop
                print(f"  Iteration: {k}, Error: {error_U:.2e}")

                # Measure time for residual and system matrix computation
                start_time = time.time()
                R, A = self.compute_residual(U_new_flat, U_n_flat, At, M, E, K, F)
                print(f"    Residual and system matrix computation time: {time.time() - start_time:.6f} seconds")

                # Measure time for boundary conditions application
                start_time = time.time()
                U_current_flat = U_new_flat.copy()  # This is U_current in the current Newton iteration
                A, R = boundary_conditions_parallel.apply_boundary_conditions(A, R, U_current_flat, boundary_dofs_u_x, mu1)
                print(f"    Boundary conditions application time: {time.time() - start_time:.6f} seconds")

                # Measure time for projecting the system into the reduced-order basis
                start_time = time.time()
                if projection == "Galerkin":
                    # Galerkin projection
                    Ar = Phi.T @ A @ Phi
                    br = Phi.T @ R
                elif projection == "LSPG":
                    # # LSPG projection Python 
                    # J_Phi = A @ Phi
                    # Ar = J_Phi.T @ J_Phi
                    # br = J_Phi.T @ R

                    # C++ implementation
                    start_time = time.time()

                    # Using the C++ function to compute Ar_cpp and br_cpp
                    Ar, br = mkl_sparse_dense_operations.compute_Ar_br(A, Phi, R)
                else:
                    raise ValueError(f"Projection method '{projection}' is not available. Please use 'Galerkin' or 'LSPG'.")
                print(f"    Projection time: {time.time() - start_time:.6f} seconds")

                # Measure time for solving the reduced system
                start_time = time.time()
                delta_q = np.linalg.solve(Ar, -br)
                print(f"    Reduced system solver time: {time.time() - start_time:.6f} seconds")

                # Update the reduced coordinates q
                q = Phi.T @ U_new_flat + delta_q

                # Measure time for updating the full-order solution
                start_time = time.time()
                U_new_flat = Phi @ q
                print(f"    Update solution time: {time.time() - start_time:.6f} seconds")

                # Enforce boundary conditions explicitly in the solution vector
                # U_new_flat[boundary_dof_indices] = mu1  # Enforce u_x = mu1 at x = 0

                # Compute the error
                error_U = np.linalg.norm(delta_q) / (np.linalg.norm(q) + 1e-12)

                k += 1

            if k == max_iter:
                print("  Warning: Nonlinear solver did not converge within maximum iterations")

            # Store the converged solution for this time step
            U_PROM_flat[n + 1, :] = U_new_flat

            # Optional: Plot the solution at every 'plot_interval' time step
            # if (n % plot_interval == 0):
            #     self.plot_solution(U_new_flat, X, Y, n+1, k)

        # Return the flattened solution matrix
        return U_PROM_flat
    
    def pod_prom_burgers(self, At, nTimeSteps, u0, mu1, E, mu2, Phi, X, Y, projection="LSPG", plot_interval=10):
        n_nodes = len(self.X)
        total_dofs = n_nodes * 2  # Total degrees of freedom (u_x and u_y components for each node)

        # Flatten the initial condition
        u0_flat = np.concatenate([u0[:, 0], u0[:, 1]])  # Shape: (total_dofs,)

        # Allocate memory for the flattened solution matrix
        U_PROM_flat = np.zeros((nTimeSteps + 1, total_dofs))
        U_PROM_flat[0, :] = u0_flat  # Set initial condition

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

            U_n_flat = U_PROM_flat[n, :]  # Current solution at time step n
            U_new_flat = U_n_flat.copy()  # Initial guess for U_new_flat

            error_U = 1
            k = 0
            max_iter = 15  # Maximum iterations for the nonlinear solver
            tol = 1e-8

            while (error_U > tol) and (k < max_iter):  # Nonlinear solver loop
                print(f"  Iteration: {k}, Error: {error_U:.2e}")

                # Measure time for residual and system matrix computation
                start_time = time.time()
                R, A = self.compute_residual(U_new_flat, U_n_flat, At, M, E, K, F)
                print(f"    Residual and system matrix computation time: {time.time() - start_time:.6f} seconds")

                # Measure time for boundary conditions application
                start_time = time.time()
                U_current_flat = U_new_flat.copy()  # This is U_current in the current Newton iteration
                A, R = boundary_conditions_parallel.apply_boundary_conditions(A, R, U_current_flat, boundary_dofs_u_x, mu1)
                print(f"    Boundary conditions application time: {time.time() - start_time:.6f} seconds")

                # Measure time for projecting the system into the reduced-order basis
                start_time = time.time()
                if projection == "Galerkin":
                    # Galerkin projection
                    Ar = Phi.T @ A @ Phi
                    br = Phi.T @ R
                elif projection == "LSPG":
                    # # LSPG projection Python 
                    # J_Phi = A @ Phi
                    # Ar_ = J_Phi.T @ J_Phi
                    # br_ = J_Phi.T @ R

                    # Allocate memory for Ar and br
                    Phi = np.asfortranarray(Phi)
                    R = R.reshape(-1, 1)
                    Ar = np.zeros((Phi.shape[1], Phi.shape[1]))
                    br = np.zeros(Phi.shape[1])
                    Ar = np.asfortranarray(Ar)
                    br = np.asfortranarray(br)

                    # Call the function
                    mkl_sparse_dense_operations.compute_Ar_br(A, Phi, R, Ar, br)
                    print(f"    Projection time cpp: {time.time() - start_time:.6f} seconds")
                else:
                    raise ValueError(f"Projection method '{projection}' is not available. Please use 'Galerkin' or 'LSPG'.")
                print(f"    Projection time: {time.time() - start_time:.6f} seconds")

                # Measure time for solving the reduced system
                start_time = time.time()
                delta_q = np.linalg.solve(Ar, -br)
                print(f"    Reduced system solver time: {time.time() - start_time:.6f} seconds")

                # Update the reduced coordinates q
                q = Phi.T @ U_new_flat + delta_q

                ####FOM####
                # # Measure time for solving the system
                # start_time = time.time()
                # # Solve the linear system using your preferred solver
                # delta_U = sparse_solver_parallel.solve_sparse_system(A, R)
                # print(f"    Solver time: {time.time() - start_time:.6f} seconds")

                # # Update the solution
                # U_new_flat_fom = U_new_flat + delta_U  # U_new_flat = U_new_flat + delta_U
                # U_new_flat_fom_approx = Phi@(Phi.T @ U_new_flat_fom)

                ####FOM####

                # Measure time for updating the full-order solution
                start_time = time.time()
                U_new_flat = Phi @ q
                print(f"    Update solution time: {time.time() - start_time:.6f} seconds")

                # Enforce boundary conditions explicitly in the solution vector
                # U_new_flat[boundary_dof_indices] = mu1  # Enforce u_x = mu1 at x = 0

                # Compute the error
                error_U = np.linalg.norm(delta_q) / (np.linalg.norm(q) + 1e-12)
                # error_U = np.linalg.norm(delta_U) / (np.linalg.norm(U_new_flat) + 1e-12)

                k += 1

            if k == max_iter:
                print("  Warning: Nonlinear solver did not converge within maximum iterations")

            # Store the converged solution for this time step
            U_PROM_flat[n + 1, :] = U_new_flat

            # Optional: Plot the solution at every 'plot_interval' time step
            # if (n % plot_interval == 0):
            #     self.plot_solution(U_new_flat, X, Y, n+1, k)
            #     self.plot_solution(U_new_flat_fom_approx, X, Y, n+1, k)

        # Return the flattened solution matrix
        return U_PROM_flat

    def gaussian_rbf(self, r, epsilon):
        """Gaussian RBF kernel function."""
        return np.exp(-(epsilon * r) ** 2)
    
    def pod_rbf_prom_nearest_neighbours_dynamic_2d(self, At, nTimeSteps, u0, mu1, E, mu2, U_p, U_s, q_p_train, q_s_train, kdtree, epsilon, neighbors=100, projection="LSPG"):
        """
        POD-RBF based PROM for 2D case using nearest neighbors dynamically.

        Parameters:
        - At: Time step size.
        - nTimeSteps: Number of time steps.
        - u0: Initial condition vector (shape: (n_nodes, 2) for u_x and u_y).
        - mu1: Boundary condition value for u_x.
        - E: Diffusion coefficient.
        - mu2: Parameter mu2 for the forcing term.
        - U_p: Primary POD basis (shape: (2*n_nodes, r)).
        - U_s: Secondary POD basis (shape: (2*n_nodes, s)).
        - q_p_train: Training data for principal modes (shape: (num_train_samples, r)).
        - q_s_train: Training data for secondary modes (shape: (num_train_samples, s)).
        - kdtree: Precomputed KDTree for finding nearest neighbors.
        - epsilon: The width parameter for the RBF kernel.
        - neighbors: Number of nearest neighbors to use for interpolation.
        - projection: Type of projection ("Galerkin" or "LSPG").

        Returns:
        - U_PROM_flat: Solution matrix over time (shape: (nTimeSteps+1, 2*n_nodes)).
        """
        n_nodes = len(self.X)
        total_dofs = n_nodes * 2  # Total degrees of freedom (u_x and u_y components for each node)
        U_s = np.asfortranarray(U_s)

        # Flatten the initial condition
        u0_flat = np.concatenate([u0[:, 0], u0[:, 1]])  # Shape: (total_dofs,)

        # Allocate memory for the solution matrix
        U_PROM_flat = np.zeros((nTimeSteps + 1, total_dofs))
        U_PROM_flat[0, :] = u0_flat  # Set initial condition

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

            U_n_flat = U_PROM_flat[n, :]  # Current solution at time step n
            U_new_flat = U_n_flat.copy()  # Initial guess for U_new_flat

            # Project the current state onto the primary POD basis
            # Measure time for projecting the current state onto the primary POD basis
            start_time = time.time()
            q_p = U_p.T @ U_n_flat
            print(f"Projection onto primary POD basis time: {time.time() - start_time:.6f} seconds")

            error_U = 1
            k = 0
            max_iter = 15  # Maximum iterations for the nonlinear solver
            tol = 1e-8

            while (error_U > tol) and (k < max_iter):  # Nonlinear solver loop
                print(f"  Iteration: {k}, Error: {error_U:.2e}")

                # Measure time for residual and system matrix computation
                start_time = time.time()
                R, A = self.compute_residual(U_new_flat, U_n_flat, At, M, E, K, F)
                print(f"    Residual and system matrix computation time: {time.time() - start_time:.6f} seconds")

                # Measure time for boundary conditions application
                start_time = time.time()
                U_current_flat = U_new_flat.copy()  # This is U_current in the current Newton iteration
                A, R = boundary_conditions_parallel.apply_boundary_conditions(A, R, U_current_flat, boundary_dofs_u_x, mu1)
                print(f"    Boundary conditions application time: {time.time() - start_time:.6f} seconds")

                # Measure time for RBF Jacobian computation
                start_time = time.time()
                rbf_jacobian = self.compute_rbf_jacobian_nearest_neighbours_dynamic(kdtree, q_p_train, q_s_train, q_p, epsilon, neighbors)
                print(f"    RBF Jacobian computation time: {time.time() - start_time:.6f} seconds")

                # projected_rbf_jacobian = U_s @ rbf_jacobian
                
                start_time = time.time()
                rbf_jacobian = np.asfortranarray(rbf_jacobian)
                # Allocate memory for projected_rbf_jacobian directly in Fortran order
                projected_rbf_jacobian = np.zeros((U_s.shape[0], rbf_jacobian.shape[1]), order='F')
                mkl_sparse_dense_operations.multiply_dense_matrices_mkl(U_s, rbf_jacobian, projected_rbf_jacobian, False)
                dD_u_dq = U_p + projected_rbf_jacobian
                print(f"    Project and sum RBF Jacobian time: {time.time() - start_time:.6f} seconds")
                
                start_time = time.time()
                if projection == "Galerkin":
                    # Galerkin projection
                    Ar = dD_u_dq.T @ A @ dD_u_dq
                    br = dD_u_dq.T @ R
                elif projection == "LSPG":
                    # LSPG projection
                    # J_dD_u_dq = A @ dD_u_dq
                    # Ar_ = J_dD_u_dq.T @ J_dD_u_dq
                    # br_ = J_dD_u_dq.T @ R

                    # Call the C++ function
                    dD_u_dq = np.asfortranarray(dD_u_dq)

                    # Allocate memory for Ar and br directly in Fortran order
                    Ar = np.zeros((dD_u_dq.shape[1], dD_u_dq.shape[1]), order='F')
                    br = np.zeros(dD_u_dq.shape[1], order='F')

                    # Call the function
                    mkl_sparse_dense_operations.compute_Ar_br(A, dD_u_dq, R, Ar, br, False)
                print(f"    Projection time: {time.time() - start_time:.6f} seconds")
                # Measure time for solving the reduced system
                start_time = time.time()
                delta_q_p = np.linalg.solve(Ar, -br)
                print(f"    Reduced system solver time: {time.time() - start_time:.6f} seconds")

                # Measure time for updating reduced coordinates
                start_time = time.time()
                q_p += delta_q_p
                print(f"    Update reduced coordinates time: {time.time() - start_time:.6f} seconds")

                # Measure time for interpolating q_s
                start_time = time.time()
                q_s = self.interpolate_with_rbf_nearest_neighbours_dynamic(kdtree, q_p_train, q_s_train, q_p, epsilon, neighbors)
                print(f"    Interpolation time for q_s: {time.time() - start_time:.6f} seconds")

                # Measure time for reconstructing the solution
                start_time = time.time()
                U1 = U_p @ q_p + U_s @ q_s
                print(f"    Solution reconstruction time: {time.time() - start_time:.6f} seconds")

                # Compute the error and update the solution
                error_U = np.linalg.norm(U1 - U_new_flat) / (np.linalg.norm(U1) + 1e-12)
                U_new_flat = U1
                k += 1

            if k == max_iter:
                print("  Warning: Nonlinear solver did not converge within maximum iterations")

            # Store the converged solution for this time step
            U_PROM_flat[n + 1, :] = U_new_flat

        return U_PROM_flat



    def compute_rbf_jacobian_nearest_neighbours_dynamic(self, kdtree, q_p_train, q_s_train, q_p_sample, epsilon, neighbors):
        """
        Compute the Jacobian of the RBF interpolation with respect to q_p using nearest neighbors dynamically.

        Parameters:
        - kdtree: KDTree to find nearest neighbors.
        - q_p_train: Training data for principal modes.
        - q_s_train: Training data for secondary modes.
        - q_p_sample: The input sample point (reduced coordinates, q_p).
        - epsilon: The width parameter for the RBF kernel.
        - neighbors: Number of nearest neighbors to use.

        Returns:
        - jacobian: The Jacobian matrix of the RBF's output with respect to q_p.
        """
        # Find the nearest neighbors in q_p_train
        dist, idx = kdtree.query(q_p_sample.reshape(1, -1), k=neighbors)

        # Extract the neighbor points and corresponding secondary modes
        q_p_neighbors = q_p_train[idx].reshape(neighbors, -1)
        q_s_neighbors = q_s_train[idx].reshape(neighbors, -1)

        # Compute pairwise distances between neighbors using pdist
        dists_neighbors = squareform(pdist(q_p_neighbors))

        # Compute the RBF matrix for the neighbors
        Phi_neighbors = self.gaussian_rbf(dists_neighbors, epsilon)

        # Regularization for numerical stability
        Phi_neighbors += np.eye(neighbors) * 1e-8

        # Solve for the RBF weights (W_neighbors)
        W_neighbors = np.linalg.solve(Phi_neighbors, q_s_neighbors)

        # Compute RBF kernel values between q_p_sample and its neighbors
        rbf_values = self.gaussian_rbf(dist.flatten(), epsilon)

        # Compute the Jacobian by multiplying weights and RBF kernel derivatives
        jacobian = np.zeros((q_s_neighbors.shape[1], q_p_neighbors.shape[1]))
        for i in range(neighbors):
            dphi_dq_p = -2 * epsilon**2 * (q_p_sample - q_p_neighbors[i]) * rbf_values[i]
            jacobian += np.outer(W_neighbors[i], dphi_dq_p)

        return jacobian

    def interpolate_with_rbf_nearest_neighbours_dynamic(self, kdtree, q_p_train, q_s_train, q_p_sample, epsilon, neighbors):
        """
        Interpolate the secondary modes q_s using nearest neighbors and RBF interpolation dynamically.

        Parameters:
        - kdtree: KDTree to find nearest neighbors.
        - q_p_train: Training data for principal modes.
        - q_s_train: Training data for secondary modes.
        - q_p_sample: The input sample point (reduced coordinates, q_p).
        - epsilon: The width parameter for the RBF kernel.
        - neighbors: Number of nearest neighbors to use.

        Returns:
        - q_s_pred: The predicted secondary modes for the given q_p_sample.
        """
        # Find the nearest neighbors in q_p_train
        dist, idx = kdtree.query(q_p_sample.reshape(1, -1), k=neighbors)

        # Extract the neighbor points and corresponding secondary modes
        q_p_neighbors = q_p_train[idx].reshape(neighbors, -1)
        q_s_neighbors = q_s_train[idx].reshape(neighbors, -1)

        # Compute pairwise distances between the neighbors
        dists_neighbors = squareform(pdist(q_p_neighbors))

        # Compute the RBF matrix for the neighbors
        Phi_neighbors = self.gaussian_rbf(dists_neighbors, epsilon)

        # Regularization for numerical stability
        Phi_neighbors += np.eye(neighbors) * 1e-8

        # Solve for the RBF weights (W_neighbors)
        W_neighbors = np.linalg.solve(Phi_neighbors, q_s_neighbors)

        # Compute RBF kernel values between q_p_sample and its neighbors
        rbf_values = self.gaussian_rbf(dist.flatten(), epsilon)

        # Interpolate q_s using the precomputed weights and RBF kernel values
        q_s_pred = rbf_values @ W_neighbors

        return q_s_pred

    def plot_solution(self, U_new_flat, X, Y, time_step, iteration):
        """ Plot the 3D surface of the solution at a given nonlinear iteration and time step """
        n_nodes = len(X)

        # Reshape u_x for plotting (assuming U_new_flat is in flattened form)
        u_x_values = U_new_flat[:n_nodes].reshape(len(np.unique(Y)), len(np.unique(X)))

        X_grid, Y_grid = np.meshgrid(np.unique(X), np.unique(Y))

        # Plot the 3D surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X_grid, Y_grid, u_x_values, cmap='viridis', edgecolor='none')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$u_x(x, y)$')
        ax.set_title(f'3D Surface Plot at Time Step {time_step}, Iteration {iteration}', fontsize=12)
        ax.view_init(30, -60)
        plt.show()



