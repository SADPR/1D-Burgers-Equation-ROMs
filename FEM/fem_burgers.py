import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch.autograd import grad
from scipy.sparse import lil_matrix

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
            while (error_U > 0.5e-5) and (k < 20):
                # print(f"Iteration {k}. Error: {error_U}")
                # Timing the compute_convection_matrix function
                start_time = time.time()
                C = self.compute_convection_matrix(U0)
                end_time = time.time()
                # print(f'Time taken for compute_convection_matrix: {end_time - start_time:.4f} seconds')

                # Timing the compute_forcing_vector function
                start_time = time.time()
                F = self.compute_forcing_vector(mu2)
                end_time = time.time()
                # print(f'Time taken for compute_forcing_vector: {end_time - start_time:.4f} seconds')

                A = M + At * C + At * E * K

                # Modify A for boundary conditions
                A[0, :] = 0
                A[0, 0] = 1

                # Compute right-hand side vector b
                b = M @ U[:, n] + At * F

                # Modify b for boundary conditions
                b[0] = mu1

                start_time = time.time()
                U1 = spla.spsolve(A, b)
                end_time = time.time()
                # print(f'Time taken for spla.spsolve: {end_time - start_time:.4f} seconds')

                error_U = np.linalg.norm(U1 - U0) / np.linalg.norm(U1)
                U0 = U1
                k += 1
            U[:, n + 1] = U1

        return U
    
    def prom_burgers(self, At, nTimeSteps, u0, uxa, E, mu2, Phi):
        m = len(self.X) - 1

        # Allocate memory for the solution matrix
        U = np.zeros((m + 1, nTimeSteps + 1))

        # Initial condition
        U[:, 0] = u0

        M = self.compute_mass_matrix()
        K = self.compute_diffusion_matrix()

        for n in range(nTimeSteps):
            print(f"Time Step: {n}. Time: {n*At}")
            U0 = U[:, n]
            error_U = 1
            previous_error_U = float('inf')
            k = 0
            while (error_U > 0.5e-5) and (k < 20):
                print(f"Iteration {k}. Error: {error_U}")
                C = self.compute_convection_matrix(U0)
                F = self.compute_forcing_vector(mu2)
                A = M + At * C + At * E * K

                # Modify A for boundary conditions
                A[0, :] = 0
                A[0, 0] = 1

                # Compute right-hand side vector b
                b = M @ U[:, n] + At * F

                # Modify b for boundary conditions
                b[0] = uxa

                # Project the full-order matrices and vectors onto the reduced space
                Ar = Phi.T @ A @ Phi
                br = Phi.T @ b

                # Solve the reduced-order system
                Ur1 = np.linalg.solve(Ar, br)

                # Compute the error and update the solution
                error_U = np.linalg.norm(Ur1 - Phi.T @ U0) / np.linalg.norm(Ur1)

                # Check if the change in error is very small
                if abs(previous_error_U - error_U) < 1e-8:
                    print("Warning: Did not converge. Stopped due to small change in error.")
                    break

                previous_error_U = error_U  # Update the previous error

                U0 = Phi @ Ur1
                k += 1

            # Update the full-order solution with the reduced-order solution
            U[:, n + 1] = Phi @ Ur1

        return U

    def ae_prom(self, At, nTimeSteps, u0, uxa, E, mu2, model, mean, std):
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
            U0_normalized = (torch.tensor(U0, dtype=torch.float32) - mean) / std
            q0 = model.encoder(U0_normalized.unsqueeze(0)).detach().numpy().squeeze()

            error_U = 1
            previous_error_U = float('inf')
            k = 0
            while (error_U > 0.5e-5) and (k < 20):
                print(f"Iteration {k}. Error: {error_U}")
                U0_normalized = (torch.tensor(U0, dtype=torch.float32) - mean) / std
                q0 = model.encoder(U0_normalized.unsqueeze(0)).squeeze()

                C = self.compute_convection_matrix(U0)
                F = self.compute_forcing_vector(mu2)
                A = M + At * C + At * E * K

                # Modify A for boundary conditions
                A[0, :] = 0
                A[0, 0] = 1

                # Compute right-hand side vector b
                b = M @ U[:, n] + At * F

                # Modify b for boundary conditions
                b[0] = uxa

                # Compute the Jacobian of the decoder at q0
                jacobian = self.compute_jacobian(model.decoder, q0).detach().numpy().T

                # Project the full-order matrices and vectors onto the reduced space
                jacobian_pseudo_inv = np.linalg.pinv(jacobian)
                Ar = jacobian_pseudo_inv @ A @ jacobian
                br = jacobian_pseudo_inv @ b

                # Solve the reduced-order system
                dq = np.linalg.solve(Ar, br)

                # Update q0 and decode
                q1 = q0 + torch.tensor(dq, dtype=torch.float32)
                U1_normalized = model.decoder(torch.tensor(q1, dtype=torch.float32)).detach().numpy().squeeze()
                U1 = U1_normalized * std + mean

                # Compute the error and update the solution
                error_U = np.linalg.norm(U1 - U0) / np.linalg.norm(U1)
                U0 = U1
                k += 1

            U[:, n + 1] = U1

        return U

    def compute_jacobian(self, decoder, q):
        q = torch.tensor(q, requires_grad=True)
        decoded = decoder(q.unsqueeze(0))
        jacobian = []

        for i in range(decoded.shape[1]):
            grad_outputs = torch.zeros_like(decoded)
            grad_outputs[0, i] = 1
            jacobian.append(torch.autograd.grad(decoded, q, grad_outputs=grad_outputs, create_graph=True)[0])

        return torch.stack(jacobian, dim=1).squeeze(0)

    def local_prom_burgers(self, At, nTimeSteps, u0, uxa, E, mu2, kmeans, local_bases, U_global, num_global_modes):
        m = len(self.X) - 1

        # Allocate memory for the solution matrix
        U = np.zeros((m + 1, nTimeSteps + 1))

        # Initial condition
        U[:, 0] = u0

        M = self.compute_mass_matrix()
        K = self.compute_diffusion_matrix()

        for n in range(nTimeSteps):
            print(f"Time Step: {n}. Time: {n*At}")
            U0 = U[:, n]
            error_U = 1
            previous_error_U = float('inf')
            k = 0
            while (error_U > 0.5e-5) and (k < 20):
                print(f"Iteration {k}. Error: {error_U}")

                # Determine the cluster for the current state
                q_global_snapshot = (U_global[:, :num_global_modes]).T @ U0
                cluster_id = kmeans.predict(q_global_snapshot.reshape(1, -1))[0]
                Phi = local_bases[cluster_id]

                C = self.compute_convection_matrix(U0)
                F = self.compute_forcing_vector(mu2)
                A = lil_matrix(M + At * C + At * E * K)  # Convert to lil_matrix

                # Modify A for boundary conditions
                A[0, :] = 0
                A[0, 0] = 1

                # Convert back to original format if necessary
                A = A.tocsc()

                # Compute right-hand side vector b
                b = M @ U[:, n] + At * F

                # Modify b for boundary conditions
                b[0] = uxa

                # Project the full-order matrices and vectors onto the reduced space
                Ar = Phi.T @ A @ Phi
                br = Phi.T @ b

                # Solve the reduced-order system
                Ur1 = np.linalg.solve(Ar, br)

                # Compute the error and update the solution
                error_U = np.linalg.norm(Ur1 - Phi.T @ U0) / np.linalg.norm(Ur1)

                # Check if the change in error is very small
                if abs(previous_error_U - error_U) < 1e-8:
                    print("Warning: Did not converge. Stopped due to small change in error.")
                    break

                previous_error_U = error_U  # Update the previous error

                U0 = Phi @ Ur1
                k += 1

            # Update the full-order solution with the reduced-order solution
            U[:, n + 1] = Phi @ Ur1

        return U