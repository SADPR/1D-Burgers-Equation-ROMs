import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class FVBurgers:
    def __init__(self, a, b, N):
        """
        Initializes the finite volume grid for the Burgers' equation.

        Parameters
        ----------
        a : float
            Left boundary of the domain.
        b : float
            Right boundary of the domain.
        N : int
            Number of control volumes (cells) in the domain.
        """
        self.a = a
        self.b = b
        self.N = N
        self.dx = (b - a) / N

        # Include 2 ghost cells: total of N + 2 unknowns
        self.x = np.linspace(a - self.dx, b + self.dx, N + 2)  # includes ghost nodes
        self.x_centers = self.x[1:-1]  # physical cells only (exclude ghost)

    def apply_dirichlet_bc(self, U, mu1):
        """
        Applies Dirichlet boundary conditions to the solution vector U.
        """
        U[0] = mu1  # Left ghost cell
        U[-1] = U[-2]  # Right: simple outflow (zero gradient)
        return U

    def compute_flux_godunov(self, uL, uR):
        """
        Computes the Godunov flux for the scalar Burgers' equation between
        states uL (left) and uR (right).

        Parameters
        ----------
        uL : float
            Left state.
        uR : float
            Right state.

        Returns
        -------
        flux : float
            The Godunov numerical flux.
        """
        if uL > uR:
            s = 0.5 * (uL + uR)
            if s > 0:
                return 0.5 * uL**2
            else:
                return 0.5 * uR**2
        else:
            if uL >= 0:
                return 0.5 * uL**2
            elif uR <= 0:
                return 0.5 * uR**2
            else:
                return 0.0


    def compute_residual(self, U, U_prev, dt, s):
        """
        Computes the residual vector R(U) at the current Newton iteration.

        Parameters
        ----------
        U : ndarray
            Solution vector at the current iteration (including ghost cells).
        U_prev : ndarray
            Solution vector at the previous time step (including ghost cells).
        dt : float
            Time step size.
        s : ndarray
            Source term evaluated at cell centers (physical domain only, shape = (N,)).

        Returns
        -------
        R : ndarray
            Residual vector of size N (physical cells only).
        """
        R = np.zeros(self.N)

        for i in range(1, self.N + 1):  # i = 1 to N, physical cells
            uL_plus = U[i]       # left of i+1/2
            uR_plus = U[i + 1]   # right of i+1/2
            flux_plus = self.compute_flux_godunov(uL_plus, uR_plus)

            uL_minus = U[i - 1]  # left of i-1/2
            uR_minus = U[i]      # right of i-1/2
            flux_minus = self.compute_flux_godunov(uL_minus, uR_minus)

            R[i - 1] = (
                U[i] - U_prev[i]
                + (dt / self.dx) * (flux_plus - flux_minus)
                - dt * s[i - 1]  # use precomputed source at cell center
            )

        return R

    def compute_jacobian_fd(self, U, U_prev, dt, s, epsilon=1e-6):
        """
        Computes the Jacobian of the residual using finite differences.

        Parameters
        ----------
        U : ndarray
            Current iterate of the solution vector (with ghost cells).
        U_prev : ndarray
            Solution from the previous time step (with ghost cells).
        dt : float
            Time step size.
        s : ndarray
            Precomputed source term vector (physical domain only).
        epsilon : float
            Perturbation for finite difference approximation.

        Returns
        -------
        J : ndarray
            Jacobian matrix of shape (N, N), where N is number of physical cells.
        """
        N = self.N
        J = np.zeros((N, N))

        # Reference residual
        R_base = self.compute_residual(U, U_prev, dt, s)

        for j in range(N):
            U_pert = U.copy()
            U_pert[j + 1] += epsilon  # shift due to ghost cells

            R_pert = self.compute_residual(U_pert, U_prev, dt, s)
            J[:, j] = (R_pert - R_base) / epsilon

        return J

    
    def compute_source_term(self, mu2, t=None):
        """
        Computes the source term at each cell center.

        Parameters
        ----------
        mu2 : float
            Parameter controlling the exponential source variation.
        t : float or None
            Optional time argument (unused for now, but kept for generality).

        Returns
        -------
        s : ndarray
            Array of shape (N,) with the source term at each physical cell center.
        """
        return 0.02 * np.exp(mu2 * self.x_centers)
    

    def compute_flux_derivatives(self, uL, uR):
        """
        Compute derivatives of the Godunov flux with respect to left and right states.
        
        Parameters
        ----------
        uL : float
            Left state.
        uR : float
            Right state.
        
        Returns
        -------
        df_duL : float
            Derivative with respect to uL.
        df_duR : float
            Derivative with respect to uR.
        """
        if uL > uR:  # Shock
            s = 0.5 * (uL + uR)
            if s > 0:
                return uL, 0
            else:
                return 0, uR
        else:  # Rarefaction
            if uL >= 0:
                return uL, 0
            elif uR <= 0:
                return 0, uR
            else:  # uL < 0 < uR
                return 0, 0

    def compute_jacobian_analytical(self, U, U_prev, dt, s):
        """
        Computes the analytical Jacobian of the residual for Burgers' equation.
        
        Parameters
        ----------
        U : ndarray
            Current solution vector (with ghost cells, size N+2).
        U_prev : ndarray
            Previous time step solution (with ghost cells, size N+2).
        dt : float
            Time step size.
        s : ndarray
            Source term at cell centers (size N).
        
        Returns
        -------
        J : ndarray
            Jacobian matrix of shape (N, N).
        """
        N = self.N
        J = np.zeros((N, N))
        
        # Compute flux derivatives at all interfaces (0.5 to N+0.5)
        df_duL = np.zeros(N + 1)  # Interfaces j+0.5, j=0 to N
        df_duR = np.zeros(N + 1)
        for j in range(N + 1):
            uL = U[j]
            uR = U[j + 1]
            df_duL[j], df_duR[j] = self.compute_flux_derivatives(uL, uR)
        
        # Assemble tridiagonal Jacobian
        for k in range(N):
            # Sub-diagonal (k > 0)
            if k > 0:
                J[k, k - 1] = - (dt / self.dx) * df_duL[k]
            # Diagonal
            J[k, k] = 1 + (dt / self.dx) * (df_duL[k + 1] - df_duR[k])
            # Super-diagonal (k < N-1)
            if k < N - 1:
                J[k, k + 1] = (dt / self.dx) * df_duR[k + 1]
        
        return J


    def fom_burgers_newton(self, dt, n_steps, u0, mu1, mu2):
        """
        Solves the 1D inviscid Burgers' equation with a source term using
        an implicit finite volume method and Newton-Raphson.

        Parameters
        ----------
        dt : float
            Time step size.
        n_steps : int
            Number of time steps.
        u0 : ndarray
            Initial condition (shape = N, physical cells only).
        mu1 : float
            Dirichlet boundary value at x=0.
        mu2 : float
            Parameter for the exponential source term.

        Returns
        -------
        U : ndarray
            Solution matrix of shape (N, n_steps + 1).
        """
        N = self.N
        dx = self.dx

        # Initialize solution array
        U = np.zeros((N, n_steps + 1))
        U[:, 0] = u0.copy()

        # Add ghost cells
        U_ext = np.zeros(N + 2)

        # Newton parameters
        max_iter = 50
        tol = 1e-8

        s = self.compute_source_term(mu2)
        
        for n in range(n_steps):
            print(f"Step {n+1}/{n_steps}, Time = {dt*(n+1):.3f}")
            U_ext[1:-1] = U[:, n]
            U_ext[0] = mu1  # Left boundary
            U_ext[-1] = U_ext[-2]  # Right Neumann (du/dx = 0)

            u_guess = U_ext.copy()
            
            for k in range(max_iter):
                R = self.compute_residual(u_guess, U_ext, dt, s)
                J = self.compute_jacobian_analytical(u_guess, U_ext, dt, s)

                # Solve linear system
                delta_u = np.linalg.solve(J, -R)
                u_new = u_guess[1:-1] + delta_u

                # Convergence check
                rel_error = np.linalg.norm(delta_u) / np.linalg.norm(u_new)
                print(f"  Newton iter {k+1}, rel_error = {rel_error:.2e}")
                if rel_error < tol:
                    break

                # Update guess (with ghost)
                u_guess[1:-1] = u_new
                u_guess[0] = mu1
                u_guess[-1] = u_guess[-2]

            U[:, n + 1] = u_guess[1:-1]

        return U

