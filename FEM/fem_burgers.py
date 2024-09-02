import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch.autograd import grad
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt

def get_sym(qi):
    ''' Auxiliary function to get the symmetric part of q kron q '''
    size = qi.shape[0]
    vec = []

    for i in range(size):
        for j in range(size):
            if j >= i:
                vec.append(qi[i]*qi[j])

    return np.array(vec)

def get_single_Q(modes, q):
    ''' Populates Q row by row '''
    k = int(modes*(modes+1)/2)
    Q = np.empty(k)

    Q = get_sym(q)

    return Q

def get_dQ_dq(modes, q):
    """
    Compute the derivative of the quadratic terms with respect to the reduced coordinates q_p.
    This will give a matrix where each row corresponds to the derivative of a specific quadratic term
    with respect to the components of q_p.
    
    Parameters:
    - modes: int, number of modes (size of q_p).
    - q: np.array, the vector q_p of reduced coordinates.

    Returns:
    - dQ_dq: np.array, the derivative of the quadratic terms with respect to q_p.
    """
    k = int(modes * (modes + 1) / 2)
    dQ_dq = np.zeros((k, modes))
    
    index = 0
    for i in range(modes):
        for j in range(i, modes):
            if i == j:
                dQ_dq[index, i] = 2 * q[i]  # Derivative of q_i^2 w.r.t q_i
            else:
                dQ_dq[index, i] = q[j]  # Derivative of q_i * q_j w.r.t q_i
                dQ_dq[index, j] = q[i]  # Derivative of q_i * q_j w.r.t q_j
            index += 1

    return dQ_dq

def compute_derivative(U_p, H, q_p):
    """
    Compute the derivative of the quadratic manifold approximation with respect to q_p.

    Parameters:
    - U_p: np.array, the linear basis matrix (Phi_p).
    - H: np.array, the matrix H capturing the effect of secondary modes.
    - q_p: np.array, the vector of reduced coordinates in the primary space.

    Returns:
    - derivative: np.array, the derivative of the quadratic manifold approximation.
    """
    modes = len(q_p)
    dQ_dq = get_dQ_dq(modes, q_p)
    
    # The derivative of the quadratic manifold approximation
    derivative = U_p + H @ dQ_dq

    return derivative

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
                
                # Compute forcing vector
                F = self.compute_forcing_vector(mu2)

                # Form the system matrix A
                A = M + At * C + At * E * K

                # Modify A for boundary conditions
                A[0, :] = 0
                A[0, 0] = 1

                # Compute the right-hand side vector b
                b = M @ U[:, n] + At * F

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
    
    def fom_burgers_dirichlet(self, At, nTimeSteps, u0, mu1, E, mu2):
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

                # Form the system matrix A
                A = M + At * C + At * E * K

                # Modify A for boundary conditions
                A[0, :] = 0
                A[0, 0] = 1

                # Compute the right-hand side vector b
                b = M @ U[:, n] + At * F

                # Modify A for boundary conditions
                A[0, :] = 0
                A[0, 0] = 1

                # Modify b for boundary conditions
                b[0] = mu1

                # Adjust b[1] for the influence of u1*
                b[1] -= A[1, 0] * mu1  # Subtract the effect of K_{21}^1 * u1* from b[1]

                # Ensure A[1, 0] is also zeroed out if A[1, 0] was non-zero:
                A[1, 0] = 0


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
    
    def pod_prom_burgers(self, At, nTimeSteps, u0, uxa, E, mu2, Phi, projection="Galerkin"):
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

    def local_prom_burgers(self, At, nTimeSteps, u0, uxa, E, mu2, kmeans, local_bases, U_global, num_global_modes, projection="Galerkin"):
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

    def pod_quadratic_manifold(self, At, nTimeSteps, u0, uxa, E, mu2, Phi_p, H, num_modes, projection="LSPG"):
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

                # Compute q_p (linear reduced coordinates)
                q_p = Phi_p.T @ U0

                # Compute the derivative of the quadratic manifold approximation
                dD_u_dq = compute_derivative(Phi_p, H, q_p)

                if projection == "Galerkin":
                    # Galerkin projection
                    Ar = dD_u_dq.T @ A @ dD_u_dq
                    br = dD_u_dq.T @ (A @ U0 - b)
                elif projection == "LSPG":
                    # LSPG projection
                    J_dD_u_dq = A @ dD_u_dq
                    Ar = J_dD_u_dq.T @ J_dD_u_dq
                    br = J_dD_u_dq.T @ R
                else:
                    raise ValueError(f"Projection method '{projection}' is not available. Please use 'Galerkin' or 'LSPG'.")

                # Solve the reduced-order system for the correction δq_p
                delta_qp = np.linalg.solve(Ar, -br)

                # Update the reduced coordinates q_p
                q_p += delta_qp

                # Compute the updated solution in the full-order space
                U1 = Phi_p @ q_p + H @ get_single_Q(num_modes, q_p)

                # Compute the error to check for convergence
                error_U = np.linalg.norm(delta_qp) / np.linalg.norm(q_p)
                
                # Update the guess for the next iteration
                U0 = U1
                k += 1

            # Store the converged solution for this time step
            U[:, n + 1] = U1

        return U

    def pod_ann_prom(self, At, nTimeSteps, u0, uxa, E, mu2, U_p, U_s, model):

        original_data = np.load(f"../FEM/training_data/simulation_mu1_{uxa:.2f}_mu2_{mu2:.4f}.npy")
        reconstruction = U_p@(U_p.T@original_data)
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

            # if (n % 11) < 10:
            if n<8:
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
                print("POD-ANN")
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

                    # if k == 0:
                    # Compute the ANN correction term based on the reduced coordinates q_p
                    q_p_tensor = torch.tensor(q_p, dtype=torch.float32).unsqueeze(0)

                    # Compute the Jacobian of the ANN with respect to q_p
                    ann_jacobian = self.compute_ann_jacobian(model, q_p_tensor).detach().numpy().T

                    total_derivative = U_p + U_s @ ann_jacobian

                    Ar = total_derivative.T @ A @ total_derivative
                    br = total_derivative.T @ b

                    # Ar = U_p.T @ A @ U_p
                    # br = U_p.T @ b

                    sol = spla.spsolve(A, b)
                    # Solve the reduced-order system for q_s
                    q_p = np.linalg.solve(Ar, br)

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
                U1_pod = U_p @ q_p
                U1 = U_p @ q_p + U_s @ q_s
                U[:, n + 1] = U1

                # Optionally, plot the results for this time step
                plt.figure()
                plt.plot(self.X, original_data[:,n+1], label=f'Orig', color='orange')
                plt.plot(self.X, reconstruction[:,n+1], label=f'POD Rec', color='black')
                plt.plot(self.X, sol, label=f'FOM', color='green')
                plt.plot(self.X, U1_pod, label=f'POD', color='blue')
                plt.plot(self.X, U1, label=f'POD-ANN', color='red')
                plt.xlabel('x')
                plt.ylabel('u')
                plt.xlim(0, 3)
                plt.title(f'PROM Solution at Time Step {n + 1}')
                plt.legend()
                plt.grid(True)
                plt.show()

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

        # Forward pass through the ANN model
        output = model(q)

        # Initialize an empty list to store the gradients
        jacobian = []

        # Loop over each element of the output (assuming output is a vector)
        for i in range(output.shape[1]):  # Assuming output is of shape [batch_size, output_dim]
            # Zero the gradients
            if q.grad is not None:
                q.grad.zero_()

            # Compute the gradient of the i-th output element with respect to the input q
            grad_output = torch.zeros_like(output)
            grad_output[:, i] = 1  # Set the i-th element to 1 to get the gradient w.r.t. q
            grad_i = torch.autograd.grad(outputs=output, inputs=q, grad_outputs=grad_output, retain_graph=True, create_graph=True)[0]

            # Append the gradient to the list
            jacobian.append(grad_i)

        # Stack the list of gradients to form the Jacobian matrix
        jacobian = torch.stack(jacobian, dim=0).squeeze(1)

        # Transpose the Jacobian to match the expected dimensions
        jacobian = jacobian.transpose(0, 1).detach()

        return jacobian
