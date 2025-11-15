import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch.autograd import grad
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def get_sym(qi):
    ''' Auxiliary function to get the symmetric part of q kron q '''
    size = qi.shape[0]
    vec = []

    for i in range(size):
        for j in range(size):
            if j >= i:
                vec.append(qi[i]*qi[j])

    return np.array(vec)

def get_single_Q(n: int, q: np.ndarray) -> np.ndarray:
    """
    Return the unique quadratic monomials of a single vector  q ∈ ℝⁿ.

    Parameters
    ----------
    n : int          # expected length of q (sanity check)
    q : (n,) array

    Returns
    -------
    Qq : (k,) array  with k = n(n+1)/2
    """
    assert q.size == n, "q length mismatch"
    return get_sym(q)


def get_dQ_dq(n: int, q: np.ndarray) -> np.ndarray:
    """
    Derivative  d(get_single_Q)/dq  ∈ ℝ^{k×n}   (k = n(n+1)/2)

    Row ordering matches get_sym:
        [ q1²,  q1 q2,  q1 q3, …, q2²,  q2 q3, …, qn² ]
    """
    k = n * (n + 1) // 2
    dQ = np.zeros((k, n), dtype=q.dtype)

    row = 0
    for i in range(n):
        # diagonal term  q_i²
        dQ[row, i] = 2.0 * q[i]
        row += 1
        # off-diagonal terms  q_i q_j  (j>i)
        for j in range(i + 1, n):
            dQ[row, i] = q[j]
            dQ[row, j] = q[i]
            row += 1
    return dQ

class FEMBurgers:
    def __init__(self, X, T):
        self.X = X
        self.T = T
        self.ngaus = 2
        self.zgp = np.array([-np.sqrt(3) / 3, np.sqrt(3) / 3])
        self.wgp = np.array([1, 1])
        self.N = np.array([(1 - self.zgp) / 2, (1 + self.zgp) / 2]).T
        self.Nxi = np.array([[-1 / 2, 1 / 2], [-1 / 2, 1 / 2]])

    def compute_mass_matrix(self):
        n_nodes = len(self.X)  # Number of nodes
        n_elements, n_local_nodes = self.T.shape  # Number of elements and nodes per element
        M_global = sp.lil_matrix((n_nodes, n_nodes))  # Initialize global mass matrix

        for elem in range(n_elements):
            element_nodes = self.T[elem, :] - 1  # Nodes of the current element (adjusted for 0-based indexing)
            x_element = self.X[element_nodes].reshape(-1, 1)  # Physical coordinates of the element's nodes

            M_element = np.zeros((n_local_nodes, n_local_nodes))  # Initialize local mass matrix

            for gauss_point in range(self.ngaus):
                N_gp = self.N[gauss_point, :]  # Shape functions at the Gauss point
                dN_dxi_gp = self.Nxi[gauss_point, :]  # Shape function derivatives wrt reference coordinate at the Gauss point

                # Compute the Jacobian
                J = dN_dxi_gp @ x_element

                # Compute the differential volume
                dV = self.wgp[gauss_point] * np.abs(J)

                # Update the local mass matrix
                M_element += np.outer(N_gp, N_gp) * dV

            # Assemble the local matrix into the global mass matrix
            for i in range(n_local_nodes):
                for j in range(n_local_nodes):
                    M_global[element_nodes[i], element_nodes[j]] += M_element[i, j]

        return M_global.tocsc()  # Convert to compressed sparse column format for efficiency

    def compute_diffusion_matrix(self):
        n_nodes = len(self.X)  # Number of nodes
        n_elements, n_local_nodes = self.T.shape  # Number of elements and nodes per element
        K_global = sp.lil_matrix((n_nodes, n_nodes))  # Initialize global diffusion matrix

        for elem in range(n_elements):
            element_nodes = self.T[elem, :] - 1  # Nodes of the current element (adjusted for 0-based indexing)
            x_element = self.X[element_nodes].reshape(-1, 1)  # Physical coordinates of the element's nodes

            K_element = np.zeros((n_local_nodes, n_local_nodes))  # Initialize local diffusion matrix

            for gauss_point in range(self.ngaus):
                N_gp = self.N[gauss_point, :]  # Shape functions at the Gauss point
                dN_dxi_gp = self.Nxi[gauss_point, :]  # Shape function derivatives wrt reference coordinate at the Gauss point

                # Compute the Jacobian
                J = dN_dxi_gp @ x_element

                # Compute the differential volume
                dV = self.wgp[gauss_point] * np.abs(J)

                # Compute the derivative of shape functions with respect to the physical coordinate
                dN_dx_gp = dN_dxi_gp / J

                # Update the local diffusion matrix
                K_element += np.outer(dN_dx_gp, dN_dx_gp) * dV

            # Assemble the local matrix into the global diffusion matrix
            for i in range(n_local_nodes):
                for j in range(n_local_nodes):
                    K_global[element_nodes[i], element_nodes[j]] += K_element[i, j]

        return K_global.tocsc()  # Convert to compressed sparse column format for efficiency

    def compute_convection_matrix(self, U_n):
        n_nodes = len(self.X)  # Number of nodes
        n_elements, n_local_nodes = self.T.shape  # Number of elements and nodes per element
        C_global = sp.lil_matrix((n_nodes, n_nodes))  # Initialize global convection matrix

        for elem in range(n_elements):
            element_nodes = self.T[elem, :] - 1  # Nodes of the current element (adjusted for 0-based indexing)
            x_element = self.X[element_nodes].reshape(-1, 1)  # Physical coordinates of the element's nodes

            u_element = U_n[element_nodes]  # Solution values at the element's nodes
            C_element = np.zeros((n_local_nodes, n_local_nodes))  # Initialize local convection matrix

            for gauss_point in range(self.ngaus):
                N_gp = self.N[gauss_point, :]  # Shape functions at the Gauss point
                dN_dxi_gp = self.Nxi[gauss_point, :]  # Shape function derivatives wrt reference coordinate at the Gauss point

                # Compute the Jacobian
                J = dN_dxi_gp @ x_element

                # Compute the differential volume
                dV = self.wgp[gauss_point] * np.abs(J)

                # Compute the derivative of shape functions with respect to the physical coordinate
                dN_dx_gp = dN_dxi_gp / J

                # Compute the solution value at the Gauss point
                u_gp = N_gp @ u_element

                # Update the local convection matrix
                C_element += np.outer(N_gp, u_gp * dN_dx_gp) * dV

            # Assemble the local matrix into the global convection matrix
            for i in range(n_local_nodes):
                for j in range(n_local_nodes):
                    C_global[element_nodes[i], element_nodes[j]] += C_element[i, j]

        return C_global.tocsc()  # Convert to compressed sparse column format for efficiency

    def compute_forcing_vector(self, mu2):
        n_nodes = len(self.X)  # Number of nodes
        n_elements, n_local_nodes = self.T.shape  # Number of elements and nodes per element
        F_global = np.zeros(n_nodes)  # Initialize global forcing vector

        for elem in range(n_elements):
            element_nodes = self.T[elem, :] - 1  # Nodes of the current element (adjusted for 0-based indexing)
            x_element = self.X[element_nodes].reshape(-1, 1)  # Physical coordinates of the element's nodes

            F_element = np.zeros(n_local_nodes)  # Initialize local forcing vector

            for gauss_point in range(self.ngaus):
                N_gp = self.N[gauss_point, :]  # Shape functions at the Gauss point
                dN_dxi_gp = self.Nxi[gauss_point, :]  # Shape function derivatives wrt reference coordinate at the Gauss point

                # Compute the Jacobian
                J = dN_dxi_gp @ x_element

                # Compute the differential volume
                dV = self.wgp[gauss_point] * np.abs(J)

                # Compute the physical coordinate at the Gauss point
                x_gp = N_gp @ x_element

                # Compute the forcing function at the Gauss point
                f_gp = 0.02 * np.exp(mu2 * x_gp)

                # Update the local forcing vector
                F_element += f_gp * N_gp * dV

            # Assemble the local vector into the global forcing vector
            for i in range(n_local_nodes):
                F_global[element_nodes[i]] += F_element[i]

        return F_global

    def compute_convection_matrix_derivative(self, U_n):
        n_nodes = len(self.X)  # Number of nodes
        n_elements, n_local_nodes = self.T.shape  # Number of elements and nodes per element
        dC_dU_global = sp.lil_matrix((n_nodes, n_nodes))  # Initialize global derivative matrix

        for elem in range(n_elements):
            element_nodes = self.T[elem, :] - 1  # Nodes of the current element
            x_element = self.X[element_nodes].reshape(-1, 1)  # Physical coordinates of the element's nodes

            u_element = U_n[element_nodes]  # Solution values at the element's nodes
            dC_dU_element = np.zeros((n_local_nodes, n_local_nodes))  # Initialize local derivative matrix

            for gauss_point in range(self.ngaus):
                N_gp = self.N[gauss_point, :]  # Shape functions at the Gauss point
                dN_dxi_gp = self.Nxi[gauss_point, :]  # Shape function derivatives wrt reference coordinate at the Gauss point

                # Compute the Jacobian
                J = dN_dxi_gp @ x_element

                # Compute the differential volume
                dV = self.wgp[gauss_point] * np.abs(J)

                # Compute the derivative of shape functions with respect to the physical coordinate
                dN_dx_gp = dN_dxi_gp / J

                # Compute the derivative of the convection matrix with respect to U
                for i in range(n_local_nodes):
                    for j in range(n_local_nodes):
                        dC_dU_element[i, j] += N_gp[i] * dN_dx_gp[j] * dV

            # Assemble the local matrix into the global convection matrix derivative
            for i in range(n_local_nodes):
                for j in range(n_local_nodes):
                    dC_dU_global[element_nodes[i], element_nodes[j]] += dC_dU_element[i, j]

        return dC_dU_global.tocsc()  # Convert to compressed sparse column format for efficiency
    
    def compute_supg_term(self, U_n, mu2):
        """
        Computes the SUPG stabilization term for the inviscid Burgers' equation:
            R(u) = (u * d(u)/dx) - f(x)

        The code integrates:
            tau_e * R(u) * (dN/dx)
        over each element, where tau_e is the local stabilization parameter.

        Parameters
        ----------
        U_n   : ndarray
            The current nodal solution vector (size = n_nodes).
        mu2   : float
            Parameter for the forcing term f(x) = 0.02 * exp(mu2 * x).

        Returns
        -------
        S_global : ndarray
            Nodal vector of the SUPG stabilization contribution.
        """
        n_nodes = len(self.X)
        n_elements, n_local_nodes = self.T.shape
        S_global = np.zeros(n_nodes)

        # User-chosen factor for tau:
        alpha = 0.5
        # Small offset to avoid division-by-zero or extremely large tau:
        eps_vel = 1.0e-10

        for e in range(n_elements):
            element_nodes = self.T[e, :] - 1
            x_element = self.X[element_nodes]        # e.g., [x_left, x_right]
            u_element = U_n[element_nodes]           # local solution at the 2 nodes
            h_e = x_element[1] - x_element[0]        # element length

            # Average velocity in this element:
            u_e = np.mean(u_element)

            # Stabilization param tau_e ~ alpha * (h_e / (2 * |u_e|)) 
            # (commonly used in 1D for convection-dominated problems)
            vel_scale = abs(u_e) if abs(u_e) > eps_vel else eps_vel
            tau_e = alpha * h_e / (2.0 * vel_scale)

            # Approx. derivative du/dx in the element:
            du_dx = (u_element[1] - u_element[0]) / h_e

            # Prepare local array for accumulation
            S_element = np.zeros(n_local_nodes)

            # Gauss integration in the reference element [-1,+1]
            for gp in range(self.ngaus):
                N_gp = self.N[gp, :]
                dN_dxi_gp = self.Nxi[gp, :]

                # Jacobian
                J = dN_dxi_gp @ x_element
                dN_dx_gp = dN_dxi_gp / J

                # Physical coordinate at this Gauss point
                x_gp = N_gp @ x_element

                # Field values at the Gauss point
                u_gp = N_gp @ u_element
                # Forcing term at x_gp
                f_gp = 0.02 * np.exp(mu2 * x_gp)

                # PDE residual (inviscid Burgers):
                # R(u) = (u * du/dx) - f(x)
                R_gp = (u_gp * du_dx) - f_gp

                # Weighting (Jacobian, Gauss weight)
                dV = self.wgp[gp] * abs(J)

                # Add the contribution:
                S_element += tau_e * R_gp * dN_dx_gp * dV

            # Assemble local SUPG contributions into the global vector
            for i in range(n_local_nodes):
                S_global[element_nodes[i]] += S_element[i]

        return S_global


    def fom_burgers_newton(self, At, nTimeSteps, u0, mu1, E, mu2):
        m = len(self.X) - 1

        # Allocate memory for the solution matrix
        U = np.zeros((m + 1, nTimeSteps + 1))

        # Initial condition
        U[:, 0] = u0

        M = self.compute_mass_matrix()
        K = self.compute_diffusion_matrix()

        # Initial lambda value and scaling factor
        lambda_ = 0.1
        tolerance = 1e-6

        for n in range(nTimeSteps):
            print(f"Time Step: {n}. Time: {n * At}")
            U0 = U[:, n]
            error_U = 1
            k = 0

            while (error_U > tolerance) and (k < 100):
                print(f"Error: {error_U}, Iteration: {k}")

                # Compute convection matrix
                C = self.compute_convection_matrix(U0)

                # Compute derivative of convection matrix
                dC_dU = self.compute_convection_matrix_derivative(U0)

                # Construct the Jacobian matrix
                J = M + At * E * K + At * C + At * dC_dU @ U0  # Matrix multiplication for dC_dU * U0

                # Apply boundary condition to Jacobian matrix
                J[0, :] = 0
                J[0, 0] = 1

                # Compute the residual
                R = (M + At * C + At * E * K) @ U0 - (M @ U[:, n] + At * self.compute_forcing_vector(mu2))

                # Apply boundary condition to the residual
                R[0] = U0[0] - mu1

                # Solve for the update delta_U
                delta_U = spla.spsolve(J, -R)

                # Update the solution with the damping factor
                U1 = U0 + lambda_ * delta_U

                # Compute the error for convergence check
                error_new = np.linalg.norm(U1 - U0) / np.linalg.norm(U1)

                error_U = error_new
                U0 = U1
                k += 1

            U[:, n + 1] = U1

        return U


    def fom_burgers(self, At, nTimeSteps, u0, mu1, E, mu2):
        m = len(self.X) - 1

        # Allocate memory for the solution matrix
        U = np.zeros((m + 1, nTimeSteps + 1))

        # Initial condition
        U[:, 0] = u0

        M = self.compute_mass_matrix()
        K = self.compute_diffusion_matrix()

        for n in range(nTimeSteps):
            print(f"Time Step: {n}. Time: {n * At}")
            U0 = U[:, n]
            error_U = 1
            k = 0
            while (error_U > 1e-6) and (k < 20):
                print(f"Iteration: {k}, Error: {error_U}")

                # Compute convection matrix using the current solution guess
                C = self.compute_convection_matrix(U0)

                # Compute SUPG term
                S = self.compute_supg_term(U0, mu2)

                # Compute forcing vector
                F = self.compute_forcing_vector(mu2)

                # Form the system matrix A
                A = M + At * C + At * E * K

                # Modify A for boundary conditions
                A[0, :] = 0
                A[0, 0] = 1

                # Compute the right-hand side vector b
                b = M @ U[:, n] + At * F - At * S

                # Modify b for boundary conditions
                b[0] = mu1

                # Compute the residual R
                R = A @ U0 - b

                # Solve the linear system J * δU = -R
                delta_U = spla.spsolve(A, -R)

                # Update the solution using the correction term
                U1 = U0 + delta_U

                # Compute the error to check for convergence
                error_U = np.linalg.norm(delta_U) / np.linalg.norm(U1)

                # Update the guess for the next iteration
                U0 = U1
                k += 1

            # Store the converged solution for this time step
            U[:, n + 1] = U1

        return U
    
    def pod_prom_burgers(self, At, nTimeSteps, u0, mu1, E, mu2, Phi, projection="Galerkin"):
        m = len(self.X) - 1

        # Allocate memory for the solution matrix
        U = np.zeros((m + 1, nTimeSteps + 1))

        # Initial condition
        U[:, 0] = u0

        M = self.compute_mass_matrix()
        K = self.compute_diffusion_matrix()

        for n in range(nTimeSteps):
            print(f"Time Step: {n}. Time: {n * At}")
            U0 = U[:, n]
            error_U = 1
            k = 0
            while (error_U > 1e-6) and (k < 20):
                print(f"Iteration: {k}, Error: {error_U}")

                # Compute convection matrix using the current solution guess
                C = self.compute_convection_matrix(U0)

                # Compute SUPG term
                S = self.compute_supg_term(U0, mu2)

                # Compute forcing vector
                F = self.compute_forcing_vector(mu2)

                # Form the system matrix A
                A = M + At * C + At * E * K

                # Modify A for boundary conditions
                A[0, :] = 0
                A[0, 0] = 1

                # Compute the right-hand side vector b
                b = M @ U[:, n] + At * F - At * S

                # Modify b for boundary conditions
                b[0] = mu1

                # Compute the residual R
                R = A @ U0 - b

                if projection == "Galerkin":
                    # Galerkin projection
                    Ar = Phi.T @ A @ Phi
                    br = Phi.T @ R
                elif projection == "LSPG":
                    # LSPG projection
                    J_Phi = A @ Phi
                    Ar = J_Phi.T @ J_Phi
                    br = J_Phi.T @ R
                else:
                    raise ValueError(f"Projection method '{projection}' is not available. Please use 'Galerkin' or 'LSPG'.")

                # Solve the reduced-order system for the correction δq
                delta_q = np.linalg.solve(Ar, -br)

                # Update the reduced coordinates q
                q = Phi.T @ U0 + delta_q

                # Compute the updated solution in the full-order space
                U1 = Phi @ q

                # Compute the error to check for convergence
                error_U = np.linalg.norm(delta_q) / np.linalg.norm(q)

                # Update the guess for the next iteration
                U0 = U1
                k += 1

            # Store the converged solution for this time step
            U[:, n + 1] = U1

        return U

    def pod_prom_burgers_(self, At, nTimeSteps, u0, uxa, E, mu2, Phi, projection="Galerkin"):
        m = len(self.X) - 1

        # Allocate memory for the solution matrix
        U = np.zeros((m + 1, nTimeSteps + 1))

        # Initial condition
        U[:, 0] = u0

        M = self.compute_mass_matrix()
        K = self.compute_diffusion_matrix()

        for n in range(nTimeSteps):
            print(f"Time Step: {n}. Time: {n * At}")
            U0 = U[:, n]
            error_U = 1
            k = 0
            while (error_U > 1e-6) and (k < 20):
                print(f"Iteration: {k}, Error: {error_U}")

                # Compute convection matrix using the current solution guess
                C = self.compute_convection_matrix(U0)

                # Compute forcing vector
                F = self.compute_forcing_vector(mu2)

                # Form the system matrix A (Jacobian J) and right-hand side vector b
                A = M + At * C + At * E * K
                b = M @ U[:, n] + At * F

                # Modify A and b for boundary conditions
                A[0, :] = 0
                A[0, 0] = 1
                b[0] = uxa

                # Compute the residual R
                R = A @ U0 - b

                if projection == "Galerkin":
                    # Galerkin projection
                    Ar = Phi.T @ A @ Phi
                    br = Phi.T @ R
                elif projection == "LSPG":
                    # LSPG projection
                    J_Phi = A @ Phi
                    Ar = J_Phi.T @ J_Phi
                    br = J_Phi.T @ R
                else:
                    raise ValueError(f"Projection method '{projection}' is not available. Please use 'Galerkin' or 'LSPG'.")

                # Solve the reduced-order system for the correction δq
                delta_q = np.linalg.solve(Ar, -br)

                # Update the reduced coordinates q
                q = Phi.T @ U0 + delta_q

                # Compute the updated solution in the full-order space
                U1 = Phi @ q

                # Compute the error to check for convergence
                error_U = np.linalg.norm(delta_q) / np.linalg.norm(q)

                # Update the guess for the next iteration
                U0 = U1
                k += 1

            # Store the converged solution for this time step
            U[:, n + 1] = U1

        return U


    def ae_prom(self, At, nTimeSteps, u0, uxa, E, mu2, model):
        m = len(self.X) - 1
        latent_dim = model.encoder[-1].out_features

        # Allocate memory for the solution matrix
        U = np.zeros((m + 1, nTimeSteps + 1))

        # Initial condition
        U[:, 0] = u0

        M = self.compute_mass_matrix()
        K = self.compute_diffusion_matrix()

        for n in range(nTimeSteps):
            print(f"Time Step: {n}. Time: {n * At}")
            U0 = U[:, n]

            if (n % 11) < 10:
                print("FOM")
                # Full-Order Model (FOM) approach for the first three time steps in each sequence of four
                error_U = 1
                k = 0
                while (error_U > 1e-6) and (k < 20):
                    print(f"FOM Iteration {k}. Error: {error_U}")

                    # Compute convection matrix using the current solution guess
                    C = self.compute_convection_matrix(U0)

                    # Compute forcing vector
                    F = self.compute_forcing_vector(mu2)

                    # Form the system matrix A and right-hand side vector b
                    A = M + At * C + At * E * K
                    b = M @ U[:, n] + At * F

                    # Modify A and b for boundary conditions
                    A[0, :] = 0
                    A[0, 0] = 1
                    b[0] = uxa

                    # Solve the full-order system
                    U1 = spla.spsolve(A, b)

                    # Compute the error to check for convergence
                    error_U = np.linalg.norm(U1 - U0) / np.linalg.norm(U1)

                    # Update the guess for the next iteration
                    U0 = U1
                    k += 1

                # Plot the results for this time step
                # plt.figure()
                # plt.plot(self.X, U1, label=f'Time step {n + 1} (FOM)', color='red')
                # plt.xlabel('x')
                # plt.ylabel('u')
                # plt.xlim(0,3)
                # plt.title(f'FOM Solution at Time Step {n + 1}')
                # plt.legend()
                # plt.grid(True)
                # plt.show()

                # Store the converged solution for this time step
                U[:, n + 1] = U1

            else:
                print("Autoencoder")
                # Autoencoder-based PROM approach for every fourth time step in each sequence of four
                # Normalize the initial condition before encoding
                U0_normalized = torch.tensor(U0, dtype=torch.float32) #(torch.tensor(U0, dtype=torch.float32) - mean) / std
                q0 = model.encoder(U0_normalized.unsqueeze(0)).detach().numpy().squeeze()


                error_U = 1
                k = 0
                while (error_U > 1e-6) and (k < 100):
                    print(f"PROM Iteration {k}. Error: {error_U}")

                    # Normalize the current estimate before encoding
                    U0_normalized = torch.tensor(U0, dtype=torch.float32) #(torch.tensor(U0, dtype=torch.float32) - mean) / std
                    q0 = model.encoder(U0_normalized.unsqueeze(0)).squeeze()

                    C = self.compute_convection_matrix(U0)
                    F = self.compute_forcing_vector(mu2)
                    A = M + At * C + At * E * K

                    # Convert to LIL format to modify the structure
                    A = A.tolil()

                    # Modify A for boundary conditions
                    A[0, :] = 0
                    A[0, 0] = 1

                    # Compute right-hand side vector b
                    b = M @ U[:, n] + At * F

                    # Modify b for boundary conditions
                    b[0] = uxa

                    if k==0:
                        # Compute the Jacobian of the decoder at q0
                        jacobian = self.compute_jacobian(model.decoder, q0).detach().numpy().T

                        # Project the full-order matrices and vectors onto the reduced space
                        jacobian_pseudo_inv = np.linalg.pinv(jacobian.T)

                    Ar = jacobian_pseudo_inv @ A @ jacobian.T
                    br = jacobian_pseudo_inv @ b

                    # Solve the reduced-order system
                    q = np.linalg.solve(Ar, br)

                    # Decode
                    U1_normalized = model.decoder(torch.tensor(q, dtype=torch.float32)).detach().numpy().squeeze()

                    # Denormalize the solution
                    U1 = U1_normalized #* std + mean

                    plt.figure()
                    plt.plot(self.X, U1_normalized, label=f'AE', color='blue')
                    plt.xlabel('x')
                    plt.ylabel('u')
                    plt.xlim(0,3)
                    plt.title(f'AE decoded initial solution')
                    plt.legend()
                    plt.grid(True)
                    plt.show()

                    # Compute the error and update the solution
                    error_U = np.linalg.norm(U1 - U0) / np.linalg.norm(U1)
                    U0 = U1
                    k += 1

                # Plot the results for this time step
                plt.figure()
                plt.plot(self.X, U1, label=f'Time step {n + 1} (PROM)', color='red')
                plt.xlabel('x')
                plt.ylabel('u')
                plt.xlim(0,3)
                plt.title(f'PROM Solution at Time Step {n + 1}')
                plt.legend()
                plt.grid(True)
                plt.show()

                # Store the converged solution for this time step
                U[:, n + 1] = U1

        return U

    def compute_jacobian(self, decoder, q):
        # Ensure q requires gradients
        q = q.clone().detach().requires_grad_(True)

        # Forward pass through the decoder
        decoded = decoder(q.unsqueeze(0))

        # Plot the results for this time step
        plt.figure()
        plt.plot(self.X, decoded.detach().numpy().T, label=f'AE', color='blue')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.xlim(0,3)
        plt.title(f'AE decoded initial solution')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Initialize an empty list to store the gradients
        jacobian = []

        # Loop over each element of the decoded output
        for i in range(decoded.shape[1]):  # Assuming decoded is of shape [1, output_dim]
            # Zero the gradients
            if q.grad is not None:
                q.grad.zero_()

            # Compute the gradient of the i-th element with respect to the input q
            grad_output = torch.zeros_like(decoded)
            grad_output[0, i] = 1  # We set the i-th element to 1 to get the gradient w.r.t. q
            grad_i = torch.autograd.grad(decoded, q, grad_outputs=grad_output, retain_graph=True)[0]

            # Append the gradient to the list
            jacobian.append(grad_i)

        # Stack the list of gradients to form the Jacobian matrix
        jacobian = torch.stack(jacobian, dim=0).squeeze(1)

        return jacobian


    from scipy.sparse import lil_matrix

    def local_prom_burgers_(self, At, nTimeSteps, u0, uxa, E, mu2, kmeans, local_bases, U_global, num_global_modes, projection="Galerkin"):
        m = len(self.X) - 1

        # Allocate memory for the solution matrix
        U = np.zeros((m + 1, nTimeSteps + 1))

        # Initial condition
        U[:, 0] = u0

        M = self.compute_mass_matrix()
        K = self.compute_diffusion_matrix()

        for n in range(nTimeSteps):
            print(f"Time Step: {n}. Time: {n * At}")
            U0 = U[:, n]
            error_U = 1
            k = 0
            while (error_U > 1e-6) and (k < 20):
                print(f"Iteration: {k}, Error: {error_U}")

                # Determine the cluster for the current state
                q_global_snapshot = (U_global[:, :num_global_modes]).T @ U0
                cluster_id = kmeans.predict(q_global_snapshot.reshape(1, -1))[0]
                Phi = local_bases[cluster_id]

                # Compute convection matrix using the current solution guess
                C = self.compute_convection_matrix(U0)

                # Compute forcing vector
                F = self.compute_forcing_vector(mu2)

                # Form the system matrix A (Jacobian J) and right-hand side vector b
                A = M + At * C + At * E * K

                # Convert to LIL format to modify the structure
                A = A.tolil()

                # Modify A for boundary conditions
                A[0, :] = 0
                A[0, 0] = 1

                # Convert back to CSC format after modifications
                A = A.tocsc()

                b = M @ U[:, n] + At * F

                # Modify b for boundary conditions
                b[0] = uxa

                # Compute the residual R
                R = A @ U0 - b

                if projection == "Galerkin":
                    # Galerkin projection
                    Ar = Phi.T @ A @ Phi
                    br = Phi.T @ R
                elif projection == "LSPG":
                    # LSPG projection
                    J_Phi = A @ Phi
                    Ar = J_Phi.T @ J_Phi
                    br = J_Phi.T @ R
                else:
                    raise ValueError(f"Projection method '{projection}' is not available. Please use 'Galerkin' or 'LSPG'.")

                # Solve the reduced-order system for the correction δq
                delta_q = np.linalg.solve(Ar, -br)

                # Update the reduced coordinates q
                q = Phi.T @ U0 + delta_q

                # Compute the updated solution in the full-order space
                U1 = Phi @ q

                # Compute the error to check for convergence
                error_U = np.linalg.norm(delta_q) / np.linalg.norm(q)

                # Update the guess for the next iteration
                U0 = U1
                k += 1

            # Store the converged solution for this time step
            U[:, n + 1] = U1

        return U
    
    def local_prom_burgers(
        self,
        At,
        nTimeSteps,
        u0,
        mu1,                 # ← BC parameter renamed (was `uxa`)
        E,
        mu2,
        kmeans,
        local_bases,
        U_global,
        num_global_modes,
        projection="Galerkin"):

        m = len(self.X) - 1

        # Allocate memory for the solution matrix
        U = np.zeros((m + 1, nTimeSteps + 1))

        # Initial condition
        U[:, 0] = u0

        M = self.compute_mass_matrix()
        K = self.compute_diffusion_matrix()

        Ug_short = U_global[:, :num_global_modes]  # for cluster assignment

        for n in range(nTimeSteps):
            print(f"Time Step: {n}. Time: {n * At}")
            U0 = U[:, n]

            # ---------- 1. choose ONE local basis for this time step ----------
            q_global_snapshot = Ug_short.T @ U0
            cluster_id = kmeans.predict(q_global_snapshot.reshape(1, -1))[0]
            Phi = local_bases[cluster_id]
            # -----------------------------------------------------------------

            error_U = 1.0
            k = 0
            while (error_U > 1e-6) and (k < 20):
                print(f"Iteration: {k}, Error: {error_U}")

                # Compute convection matrix using the current solution guess
                C = self.compute_convection_matrix(U0)

                # SUPG term (added) -------------------------------------------
                S = self.compute_supg_term(U0, mu2)
                # --------------------------------------------------------------

                # Compute forcing vector
                F = self.compute_forcing_vector(mu2)

                # Form the system matrix A
                A = M + At * C + At * E * K

                # Modify A for boundary conditions
                A = A.tolil()
                A[0, :] = 0.0
                A[0, 0] = 1.0
                A = A.tocsc()

                # Right-hand side vector b (includes SUPG) --------------------
                b = M @ U[:, n] + At * F - At * S
                # --------------------------------------------------------------

                # Boundary condition
                b[0] = mu1

                # Residual
                R = A @ U0 - b

                if projection == "Galerkin":
                    Ar = Phi.T @ A @ Phi
                    br = Phi.T @ R
                elif projection == "LSPG":
                    J_Phi = A @ Phi
                    Ar = J_Phi.T @ J_Phi
                    br = J_Phi.T @ R
                else:
                    raise ValueError(f"Projection method '{projection}' is not available. Please use 'Galerkin' or 'LSPG'.")

                # Solve the reduced-order system for the correction δq
                delta_q = np.linalg.solve(Ar, -br)

                # Update the reduced coordinates q
                q = Phi.T @ U0 + delta_q

                # Compute the updated solution in the full-order space
                U1 = Phi @ q

                # Convergence check
                error_U = np.linalg.norm(delta_q) / np.linalg.norm(q)

                # Prepare next iteration
                U0 = U1
                k += 1

            # Store the converged solution for this time step
            U[:, n + 1] = U1

        return U

    def pod_quadratic_manifold(
            self,
            At: float,
            nTimeSteps: int,
            u0: np.ndarray,
            uxa: float,                # left Dirichlet BC  u(0,t)=uxa
            E: float,                  # numerical diffusion coefficient
            mu2: float,                # source-term parameter
            Phi: np.ndarray,           # (N,n)   linear POD basis   (Φ)
            H:   np.ndarray,           # (N,k)   quadratic tensor   (H)
            projection: str = "LSPG",
            newton_tol: float = 1e-6,
            newton_itmax: int = 25):
        """
        Quadratic-manifold PROM (Galerkin or LSPG).

        Returns
        -------
        U : (N, nTimeSteps+1)  full-order DOF history reconstructed from the ROM.
        """
        N, n = Phi.shape
        k     = H.shape[1]
        Xpts  = len(self.X) - 1

        # storage ----------------------------------------------------
        U = np.zeros((Xpts + 1, nTimeSteps + 1))
        U[:, 0] = u0.copy()

        # global FEM matrices ---------------------------------------
        M = self.compute_mass_matrix()      # (N,N)
        K = self.compute_diffusion_matrix()

        # ----------------------------------------------------------- #
        # helper lambdas
        # ----------------------------------------------------------- #
        def decoder(q):
            """Φ q + H q⊗q  (unique)"""
            return Phi @ q + H @ get_single_Q(n, q)

        def tangent(q):
            """∂ũ/∂q  = Φ + H dQdq"""
            dQdq = get_dQ_dq(n, q)                  # (k,n)
            return Phi + H @ dQdq                   # (N,n)

        # ----------------------------------------------------------- #
        for m in range(nTimeSteps):
            print(f"\n=== time step {m+1}/{nTimeSteps}  (t = {(m+1)*At:.2f}) ===")
            # initial Newton guess  –  linear POD projection
            q   = Phi.T @ U[:, m]
            u   = decoder(q)

            for it in range(newton_itmax):
                # PDE operators at current solution ------------------
                C = self.compute_convection_matrix(u)
                F = self.compute_forcing_vector(mu2)
                A = M + At*C + At*E*K             # Jacobian w.r.t. state

                # Dirichlet BC at node 0
                A[0, :] = 0.0
                A[0, 0] = 1.0
                b = M @ U[:, m] + At*F
                b[0] = uxa

                # residual in full space
                R = A @ u - b

                # tangent basis & reduced system --------------------
                T = tangent(q)                    # (N,n)

                if projection.lower() == "galerkin":
                    Ar = T.T @ A @ T
                    br = T.T @ R
                elif projection.lower() == "lspg":
                    JT = A @ T                    # (N,n)
                    Ar = JT.T @ JT
                    br = JT.T @ R
                else:
                    raise ValueError("projection must be 'Galerkin' or 'LSPG'")

                # Gauss–Newton update -------------------------------
                delta_q = np.linalg.solve(Ar, -br)
                q      += delta_q
                u       = decoder(q)

                rel = np.linalg.norm(delta_q) / max(1e-14, np.linalg.norm(q))
                print(f"  Newton {it:2d}:  |δq|/|q| = {rel:.3e}")

                if rel < newton_tol:
                    break
            else:
                print("  Warning: Newton did not converge")

            U[:, m+1] = u

        return U

    def pod_ann_prom_(self, At, nTimeSteps, u0, uxa, E, mu2, U_p, U_s, model, projection="LSPG"):

        m = len(self.X) - 1

        # Allocate memory for the solution matrix
        U = np.zeros((m + 1, nTimeSteps + 1))

        # Initial condition
        U[:, 0] = u0

        M = self.compute_mass_matrix()
        K = self.compute_diffusion_matrix()

        for n in range(nTimeSteps):
            print(f"Time Step: {n}. Time: {n * At}")
            U0 = U[:, n]

            # Project the current state onto the primary POD basis
            q_p = U_p.T @ U0

            error_U = 1
            k = 0
            while (error_U > 1e-6) and (k < 100):

                C = self.compute_convection_matrix(U0)
                F = self.compute_forcing_vector(mu2)
                A = M + At * C + At * E * K

                # Convert to LIL format to modify the structure
                A = A.tolil()

                # Modify A for boundary conditions
                A[0, :] = 0
                A[0, 0] = 1

                # Compute right-hand side vector b
                b = M @ U[:, n] + At * F

                # Modify b for boundary conditions
                b[0] = uxa

                # Compute the residual R
                R = A @ U0 - b

                # Compute the ANN correction term based on the reduced coordinates q_p
                q_p_tensor = torch.tensor(q_p, dtype=torch.float32).unsqueeze(0)

                # Compute the Jacobian of the ANN with respect to q_p
                ann_jacobian = self.compute_ann_jacobian(model, q_p_tensor).detach().numpy()

                # Compute dD(u)/dq
                dD_u_dq = U_p + U_s @ ann_jacobian

                if projection == "Galerkin":
                    # Galerkin projection
                    Ar = dD_u_dq.T @ A @ dD_u_dq
                    br = dD_u_dq.T @ R
                elif projection == "LSPG":
                    # LSPG projection
                    J_dD_u_dq = A @ dD_u_dq
                    Ar = J_dD_u_dq.T @ J_dD_u_dq
                    br = J_dD_u_dq.T @ R

                # Solve the reduced-order system for q_s
                delta_q_p = np.linalg.solve(Ar, -br)

                # Update the reduced coordinates q_p
                q_p += delta_q_p

                q_p_tensor = torch.tensor(q_p, dtype=torch.float32).unsqueeze(0)

                q_s = model(q_p_tensor).detach().numpy().squeeze()

                # Reconstruct the solution using the POD-ANN model
                U1 = U_p @ q_p + U_s @ q_s

                # Compute the error and update the solution
                error_U = np.linalg.norm(U1 - U0) / np.linalg.norm(U1)
                print(f"PROM Iteration {k}. Error: {error_U}")
                U0 = U1
                k += 1

            # Store the converged solution for this time step
            U[:, n + 1] = U1

        return U

    def pod_ann_prom(self, At, nTimeSteps, u0, mu1, E, mu2,
                    U_p, U_s, model, projection="LSPG"):
        """
        Online solver for the POD-ANN PROM (Galerkin or LSPG).
        """
        N  = len(self.X) - 1          # total DOFs minus duplicate node
        n  = U_p.shape[1]             # retained modes
        nbar = U_s.shape[1]           # discarded modes

        # --- storage -----------------------------------------------------------
        U = np.zeros((N + 1, nTimeSteps + 1))
        U[:, 0] = u0

        M = self.compute_mass_matrix()
        K = self.compute_diffusion_matrix()

        # ----------------------------------------------------------------------
        for nt in range(nTimeSteps):
            print(f"Time step {nt}  (t = {nt*At: .4e})")
            U0  = U[:, nt].copy()
            q_p = U_p.T @ U0           # initial reduced coordinates

            err, it = 1.0, 0
            while err > 1e-6 and it < 50:

                # element-level operators for current state
                C = self.compute_convection_matrix(U0)
                S = self.compute_supg_term(U0, mu2)
                F = self.compute_forcing_vector(mu2)

                # full system matrix and RHS
                A = M + At*C + At*E*K
                A = A.tolil()
                A[0, :] = 0.0
                A[0, 0] = 1.0

                b = M @ U[:, nt] + At*F - At*S
                b[0] = mu1

                R = A @ U0 - b                       # full residual

                # --- ANN Jacobian wrt q_p --------------------------------------
                q_p_torch = torch.tensor(q_p, dtype=torch.float32).unsqueeze(0)
                dN_dq = self.compute_ann_jacobian(model, q_p_torch)  # (nbar × n)
                dN_dq = dN_dq.numpy()

                # decoder derivative  ∂D/∂q  (N×n)
                dD_dq = U_p + U_s @ dN_dq

                # --- reduced Newton solve --------------------------------------
                if projection.lower() == "galerkin":
                    Ar = dD_dq.T @ A @ dD_dq
                    br = dD_dq.T @ R
                elif projection.lower() == "lspg":
                    JdD = A @ dD_dq
                    Ar  = JdD.T @ JdD
                    br  = JdD.T @ R
                else:
                    raise ValueError("projection must be 'Galerkin' or 'LSPG'")

                delta_q = np.linalg.solve(Ar, -br)
                q_p    += delta_q

                # reconstruct full state
                q_s = model(torch.tensor(q_p, dtype=torch.float32)).detach().numpy()
                U1  = U_p @ q_p + U_s @ q_s

                err = np.linalg.norm(delta_q) / (np.linalg.norm(q_p) + 1e-14)
                print(f"   Newton {it:2d}:  rel Δq = {err:.3e}")
                U0 = U1
                it += 1

            U[:, nt+1] = U1

        return U


    def compute_ann_jacobian(self, model, q):
        """
        Compute the Jacobian of the ANN's output with respect to its input.

        Parameters:
        - model: The trained ANN model.
        - q: The input tensor to the ANN (reduced coordinates, q_p).

        Returns:
        - jacobian: The Jacobian matrix of the ANN's output with respect to q_p.
        """

        # Ensure q requires gradients
        q = q.clone().detach().requires_grad_(True)

        # Use torch.autograd.functional.jacobian to compute the Jacobian
        jacobian = torch.autograd.functional.jacobian(model, q)

        # Remove the batch dimension (assuming batch size is 1)
        jacobian = jacobian.squeeze(0).squeeze(1)

        return jacobian.detach() # Detach to prevent unnecessary computation graph tracking

    def pod_rbf_prom_debug(self, At, nTimeSteps, u0, uxa, E, mu2, U_p, U_s, q_p_train, W, epsilon, projection="LSPG"):
        """
        POD-RBF based PROM.

        Parameters:
        - At: Time step size.
        - nTimeSteps: Number of time steps.
        - u0: Initial condition vector.
        - uxa: Boundary condition at x = a.
        - E: Diffusion coefficient.
        - mu2: Parameter mu2 for the forcing term.
        - U_p: Primary POD basis.
        - U_s: Secondary POD basis.
        - q_p_train: Training data for principal modes.
        - W: Precomputed RBF weights for secondary modes.
        - projection: Type of projection ("Galerkin" or "LSPG").
        - epsilon: The width parameter for the RBF kernel.

        Returns:
        - U: Full solution matrix over time.
        """
        m = len(self.X) - 1

        # Allocate memory for the solution matrix
        U = np.zeros((m + 1, nTimeSteps + 1))

        # Initial condition
        U[:, 0] = u0

        # Timing the mass and diffusion matrix computation
        start_time = time.time()
        M = self.compute_mass_matrix()
        K = self.compute_diffusion_matrix()
        end_time = time.time()
        print(f"Time for mass and diffusion matrix computation: {end_time - start_time:.6f} seconds")

        for n in range(nTimeSteps):
            print(f"Time Step: {n}. Time: {n * At}")

            start_time_step = time.time()

            U0 = U[:, n]

            # Project the current state onto the primary POD basis
            start_time = time.time()
            q_p = U_p.T @ U0
            end_time = time.time()
            print(f"Time for projection onto primary POD basis: {end_time - start_time:.6f} seconds")

            error_U = 1
            k = 0
            while (error_U > 5e-6) and (k < 100):
                start_iteration = time.time()

                # Compute convection matrix
                start_time = time.time()
                C = self.compute_convection_matrix(U0)
                end_time = time.time()
                print(f"Time for convection matrix computation: {end_time - start_time:.6f} seconds")

                # Compute forcing vector
                start_time = time.time()
                F = self.compute_forcing_vector(mu2)
                end_time = time.time()
                print(f"Time for forcing vector computation: {end_time - start_time:.6f} seconds")

                # Compute system matrix A
                start_time = time.time()
                A = M + At * C + At * E * K
                A = A.tolil()  # Convert to LIL format for modification
                end_time = time.time()
                print(f"Time for system matrix A computation: {end_time - start_time:.6f} seconds")

                # Apply boundary conditions to A
                A[0, :] = 0
                A[0, 0] = 1

                # Compute right-hand side vector b
                start_time = time.time()
                b = M @ U[:, n] + At * F
                b[0] = uxa  # Apply boundary conditions to b
                end_time = time.time()
                print(f"Time for right-hand side vector b computation: {end_time - start_time:.6f} seconds")

                # Compute the residual R
                start_time = time.time()
                R = A @ U0 - b
                end_time = time.time()
                print(f"Time for residual R computation: {end_time - start_time:.6f} seconds")

                # --- Time the Jacobian computation ---
                start_time = time.time()
                # Compute the Jacobian of the RBF interpolation with respect to q_p
                rbf_jacobian = self.compute_rbf_jacobian(q_p_train, W, q_p, epsilon)
                end_time = time.time()
                print(f"Time for compute_rbf_jacobian: {end_time - start_time:.6f} seconds")

                # Compute dD(u)/dq
                start_time = time.time()
                dD_u_dq = U_p + U_s @ rbf_jacobian
                end_time = time.time()
                print(f"Time for dD(u)/dq computation: {end_time - start_time:.6f} seconds")

                # Project onto the reduced-order system
                start_time = time.time()
                if projection == "Galerkin":
                    # Galerkin projection
                    Ar = dD_u_dq.T @ A @ dD_u_dq
                    br = dD_u_dq.T @ R
                elif projection == "LSPG":
                    # LSPG projection
                    J_dD_u_dq = A @ dD_u_dq
                    Ar = J_dD_u_dq.T @ J_dD_u_dq
                    br = J_dD_u_dq.T @ R
                end_time = time.time()
                print(f"Time for reduced-order system projection: {end_time - start_time:.6f} seconds")

                # Solve the reduced-order system for q_p update
                start_time = time.time()
                delta_q_p = np.linalg.solve(Ar, -br)
                end_time = time.time()
                print(f"Time for solving reduced system: {end_time - start_time:.6f} seconds")

                # Update the reduced coordinates q_p
                q_p += delta_q_p

                # --- Time the second RBF-based correction term calculation ---
                start_time = time.time()
                # Recompute q_s using the updated q_p
                q_s = self.interpolate_with_rbf(q_p_train, W, q_p, epsilon)
                end_time = time.time()
                print(f"Time for interpolate_with_rbf (second): {end_time - start_time:.6f} seconds")

                # Reconstruct the solution using the POD-RBF model
                start_time = time.time()
                U1 = U_p @ q_p + U_s @ q_s
                end_time = time.time()
                print(f"Time for solution reconstruction: {end_time - start_time:.6f} seconds")

                # Compute the error and update the solution
                error_U = np.linalg.norm(U1 - U0) / np.linalg.norm(U1)
                print(f"PROM Iteration {k}. Error: {error_U}")
                U0 = U1
                k += 1

                end_iteration = time.time()
                print(f"Time for PROM iteration {k}: {end_iteration - start_iteration:.6f} seconds")

            # Store the converged solution for this time step
            U[:, n + 1] = U1

            end_time_step = time.time()
            print(f"Time for time step {n}: {end_time_step - start_time_step:.6f} seconds")

        return U

    def pod_rbf_prom(self, At, nTimeSteps, u0, uxa, E, mu2, U_p, U_s, q_p_train, W, epsilon, projection="LSPG"):
        """
        POD-RBF based PROM.

        Parameters:
        - At: Time step size.
        - nTimeSteps: Number of time steps.
        - u0: Initial condition vector.
        - uxa: Boundary condition at x = a.
        - E: Diffusion coefficient.
        - mu2: Parameter mu2 for the forcing term.
        - U_p: Primary POD basis.
        - U_s: Secondary POD basis.
        - q_p_train: Training data for principal modes.
        - W: Precomputed RBF weights for secondary modes.
        - projection: Type of projection ("Galerkin" or "LSPG").
        - epsilon: The width parameter for the RBF kernel.

        Returns:
        - U: Full solution matrix over time.
        """
        m = len(self.X) - 1

        # Allocate memory for the solution matrix
        U = np.zeros((m + 1, nTimeSteps + 1))

        # Initial condition
        U[:, 0] = u0

        M = self.compute_mass_matrix()
        K = self.compute_diffusion_matrix()

        for n in range(nTimeSteps):
            print(f"Time Step: {n}. Time: {n * At}")

            U0 = U[:, n]

            # Project the current state onto the primary POD basis
            q_p = U_p.T @ U0

            error_U = 1
            k = 0
            while (error_U > 5e-6) and (k < 100):
                C = self.compute_convection_matrix(U0)
                F = self.compute_forcing_vector(mu2)
                A = M + At * C + At * E * K

                # Convert to LIL format to modify the structure
                A = A.tolil()

                # Modify A for boundary conditions
                A[0, :] = 0
                A[0, 0] = 1

                # Compute right-hand side vector b
                b = M @ U[:, n] + At * F

                # Modify b for boundary conditions
                b[0] = uxa

                # Compute the residual R
                R = A @ U0 - b

                # Compute the Jacobian of the RBF interpolation with respect to q_p
                rbf_jacobian = self.compute_rbf_jacobian(q_p_train, W, q_p, epsilon)

                # Compute dD(u)/dq
                dD_u_dq = U_p + U_s @ rbf_jacobian

                if projection == "Galerkin":
                    # Galerkin projection
                    Ar = dD_u_dq.T @ A @ dD_u_dq
                    br = dD_u_dq.T @ R
                elif projection == "LSPG":
                    # LSPG projection
                    J_dD_u_dq = A @ dD_u_dq
                    Ar = J_dD_u_dq.T @ J_dD_u_dq
                    br = J_dD_u_dq.T @ R

                # Solve the reduced-order system for q_p update
                delta_q_p = np.linalg.solve(Ar, -br)

                # Update the reduced coordinates q_p
                q_p += delta_q_p

                # Recompute q_s using the updated q_p
                q_s = self.interpolate_with_rbf(q_p_train, W, q_p, epsilon)

                # Reconstruct the solution using the POD-RBF model
                U1 = U_p @ q_p + U_s @ q_s

                # Compute the error and update the solution
                error_U = np.linalg.norm(U1 - U0) / np.linalg.norm(U1)
                print(f"PROM Iteration {k}. Error: {error_U}")
                U0 = U1
                k += 1

            # Store the converged solution for this time step
            U[:, n + 1] = U1

        return U

    def compute_rbf_jacobian(self, q_p_train, W, q_p_sample, epsilon):
        """
        Compute the Jacobian of the RBF interpolation with respect to q_p.

        Parameters:
        - q_p_train: Training data for principal modes.
        - W: Precomputed weights for secondary modes.
        - q_p_sample: The input sample point (reduced coordinates, q_p).
        - epsilon: The width parameter for the RBF kernel.

        Returns:
        - jacobian: The Jacobian matrix of the RBF's output with respect to q_p.
        """
        N = q_p_train.shape[0]  # Number of training points
        input_dim = q_p_train.shape[1]  # Dimension of the input (q_p)
        output_dim = W.shape[1]  # Dimension of the output (q_s)

        # Initialize the Jacobian matrix
        jacobian = np.zeros((output_dim, input_dim))  # Shape: (273, 28)

        # Set a tolerance and threshold for numerical stability
        threshold = 1e-10  # Threshold to skip very small RBF values
        tolerance = 1e-10  # Tolerance for near-zero distances

        # Precompute distances between q_p_sample and all q_p_train points
        distances = np.linalg.norm(q_p_train - q_p_sample, axis=1)

        # Precompute RBF kernel values
        phi_r = np.exp(-(epsilon * distances) ** 2)

        # Compute Jacobian contributions for each training point
        for i in range(N):
            # If the distance is too small, skip or handle separately
            if distances[i] < tolerance:
                continue

            # Skip very small RBF kernel values to avoid numerical instability
            if np.abs(phi_r[i]) < threshold:
                continue

            # Derivative of the RBF kernel with respect to q_p_sample
            dphi_dq_p = -2 * epsilon**2 * (q_p_sample - q_p_train[i]) * phi_r[i]

            # Outer product to compute the contribution to the Jacobian
            jacobian += np.outer(W[i], dphi_dq_p)

        return jacobian

    def compute_distances(self, X1, X2):
        """Compute pairwise Euclidean distances between two sets of points."""
        return np.linalg.norm(X1 - X2, axis=1)

    def gaussian_rbf(self, r, epsilon):
        """Gaussian RBF kernel function."""
        return np.exp(-(epsilon * r) ** 2)

    def interpolate_with_rbf(self, q_p_train, W, q_p_sample, epsilon):
        """
        Interpolate the secondary modes q_s using RBF interpolation.

        Parameters:
        - q_p_train: Training data for principal modes.
        - W: Precomputed weights for secondary modes.
        - q_p_sample: The input sample point (reduced coordinates, q_p).
        - epsilon: The width parameter for the RBF kernel.

        Returns:
        - q_s_pred: The predicted secondary modes for the given q_p_sample.
        """
        # Compute distances between the sample point and the training points
        dists = self.compute_distances(q_p_train, q_p_sample)  # Shape: (n_train,)

        # Compute the RBF kernel values (Gaussian RBF)
        rbf_values = self.gaussian_rbf(dists, epsilon)  # Shape: (n_train,)

        # Compute the predicted secondary modes by multiplying the RBF values with the precomputed weights
        q_s_pred = rbf_values @ W  # Shape: (output_dim,)

        return q_s_pred

    def pod_rbf_prom_nearest_neighbours_dynamic(self, At, nTimeSteps, u0, uxa, E, mu2, U_p, U_s, q_p_train, q_s_train, kdtree, epsilon, neighbors=100, projection="LSPG"):
        """
        POD-RBF based PROM using nearest neighbors dynamically.

        Parameters:
        - At: Time step size.
        - nTimeSteps: Number of time steps.
        - u0: Initial condition vector.
        - uxa: Boundary condition at x = a.
        - E: Diffusion coefficient.
        - mu2: Parameter mu2 for the forcing term.
        - U_p: Primary POD basis.
        - U_s: Secondary POD basis.
        - q_p_train: Training data for principal modes.
        - q_s_train: Training data for secondary modes.
        - kdtree: Precomputed KDTree for finding nearest neighbors.
        - epsilon: The width parameter for the RBF kernel.
        - neighbors: Number of nearest neighbors to use for interpolation.
        - projection: Type of projection ("Galerkin" or "LSPG").

        Returns:
        - U: Full solution matrix over time.
        """
        m = len(self.X) - 1

        # Allocate memory for the solution matrix
        U = np.zeros((m + 1, nTimeSteps + 1))

        # Initial condition
        U[:, 0] = u0

        M = self.compute_mass_matrix()
        K = self.compute_diffusion_matrix()

        for n in range(nTimeSteps):
            print(f"Time Step: {n}. Time: {n * At}")

            U0 = U[:, n]

            # Project the current state onto the primary POD basis
            q_p = U_p.T @ U0

            error_U = 1
            k = 0
            while (error_U > 1e-6) and (k < 100):
                C = self.compute_convection_matrix(U0)
                F = self.compute_forcing_vector(mu2)
                A = M + At * C + At * E * K

                # Convert to LIL format to modify the structure
                A = A.tolil()

                # Modify A for boundary conditions
                A[0, :] = 0
                A[0, 0] = 1

                # Compute right-hand side vector b
                b = M @ U[:, n] + At * F

                # Modify b for boundary conditions
                b[0] = uxa

                # Compute the residual R
                R = A @ U0 - b

                # Compute the Jacobian of the RBF interpolation with nearest neighbors
                rbf_jacobian = self.compute_rbf_jacobian_nearest_neighbours_dynamic(kdtree, q_p_train, q_s_train, q_p, epsilon, neighbors)

                # Compute dD(u)/dq
                dD_u_dq = U_p + U_s @ rbf_jacobian

                if projection == "Galerkin":
                    # Galerkin projection
                    Ar = dD_u_dq.T @ A @ dD_u_dq
                    br = dD_u_dq.T @ R
                elif projection == "LSPG":
                    # LSPG projection
                    J_dD_u_dq = A @ dD_u_dq
                    Ar = J_dD_u_dq.T @ J_dD_u_dq
                    br = J_dD_u_dq.T @ R

                # Solve the reduced-order system for q_p update
                delta_q_p = np.linalg.solve(Ar, -br)

                # Update the reduced coordinates q_p
                q_p += delta_q_p

                # Recompute q_s using the updated q_p and nearest neighbors dynamic interpolation
                q_s = self.interpolate_with_rbf_nearest_neighbours_dynamic(kdtree, q_p_train, q_s_train, q_p, epsilon, neighbors)

                # Reconstruct the solution using the POD-RBF model
                U1 = U_p @ q_p + U_s @ q_s

                # Compute the error and update the solution
                error_U = np.linalg.norm(U1 - U0) / np.linalg.norm(U1)
                print(f"PROM Iteration {k}. Error: {error_U}")
                U0 = U1
                k += 1

            # Store the converged solution for this time step
            U[:, n + 1] = U1

        return U

    def compute_rbf_jacobian_nearest_neighbours_dynamic_(self, kdtree, q_p_train, q_s_train, q_p_sample, epsilon, neighbors):
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

        # Initialize the Jacobian matrix
        jacobian = np.zeros((q_s_neighbors.shape[1], q_p_neighbors.shape[1]))

        # Compute pairwise distances between the neighbors
        dists_neighbors = np.linalg.norm(q_p_neighbors[:, None, :] - q_p_neighbors[None, :, :], axis=-1)

        # Compute the RBF matrix for the neighbors
        Phi_neighbors = self.gaussian_rbf(dists_neighbors, epsilon)

        # Regularization for numerical stability
        Phi_neighbors += np.eye(neighbors) * 1e-8

        # Solve for the RBF weights (W_neighbors)
        W_neighbors = np.linalg.solve(Phi_neighbors, q_s_neighbors)

        # Compute RBF kernel values between q_p_sample and its neighbors
        rbf_values = self.gaussian_rbf(dist.flatten(), epsilon)

        # Compute the Jacobian by multiplying weights and RBF kernel derivatives
        for i in range(neighbors):
            dphi_dq_p = -2 * epsilon**2 * (q_p_sample - q_p_neighbors[i]) * rbf_values[i]
            jacobian += np.outer(W_neighbors[i], dphi_dq_p)

        return jacobian

    def interpolate_with_rbf_nearest_neighbours_dynamic_(self, kdtree, q_p_train, q_s_train, q_p_sample, epsilon, neighbors):
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
        dists_neighbors = np.linalg.norm(q_p_neighbors[:, None, :] - q_p_neighbors[None, :, :], axis=-1)

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
