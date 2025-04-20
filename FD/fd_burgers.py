import numpy as np

class FDBurgers:
    """
    Finite Difference solver for the 1D inviscid Burgers' equation with a source term.
    Uses:
      - Second-order central differences for convection,
      - Backward Euler in time,
      - Newton-Raphson with analytical or finite-difference Jacobian,
      - Optional artificial viscosity for stabilization.
    """

    def __init__(self, a, b, N):
        self.a = a
        self.b = b
        self.N = N
        self.dx = (b - a) / (N - 1)
        self.x = np.linspace(a, b, N)

    def apply_dirichlet_bc(self, U, mu1):
        U[0] = mu1
        U[-1] = U[-2]
        return U

    def compute_source_term(self, mu2):
        return 0.02 * np.exp(mu2 * self.x)

    def compute_residual(self, U, U_prev, dt, s):
        R = np.zeros_like(U)
        nu_artificial = 0.25 * self.dx * np.max(np.abs(U))
        for i in range(1, self.N - 1):
            conv = (0.5 * U[i + 1]**2 - 0.5 * U[i - 1]**2) / (2 * self.dx)
            diff = nu_artificial * (U[i + 1] - 2 * U[i] + U[i - 1]) / self.dx**2
            R[i] = (U[i] - U_prev[i]) / dt + conv - s[i] - diff
        return R

    def compute_jacobian_analytical(self, U, U_prev, dt, s):
        J = np.zeros((self.N, self.N))
        nu_artificial = 0.25 * self.dx * np.max(np.abs(U))
        for i in range(1, self.N - 1):
            J[i, i - 1] = -U[i - 1] / (2 * self.dx) - nu_artificial / self.dx**2
            J[i, i + 1] = U[i + 1] / (2 * self.dx) - nu_artificial / self.dx**2
            J[i, i] = 1 / dt + 2 * nu_artificial / self.dx**2
        return J

    def compute_jacobian_fd(self, U, U_prev, dt, s, epsilon=1e-8):
        R_base = self.compute_residual(U, U_prev, dt, s)
        J = np.zeros((self.N, self.N))
        for j in range(1, self.N - 1):
            U_pert = U.copy()
            U_pert[j] += epsilon
            R_pert = self.compute_residual(U_pert, U_prev, dt, s)
            J[1:self.N - 1, j] = (R_pert[1:self.N - 1] - R_base[1:self.N - 1]) / epsilon
        return J

    def fom_burgers_newton(self, dt, n_steps, U0, mu1, mu2, max_iter=30, tol=1e-8, use_fd_jacobian=False):
        Uall = np.zeros((self.N, n_steps + 1))
        U_current = self.apply_dirichlet_bc(U0.copy(), mu1)
        Uall[:, 0] = U_current
        s = self.compute_source_term(mu2)

        for step in range(n_steps):
            t = (step + 1) * dt
            print(f"\nTime step {step+1}/{n_steps}  |  t = {t:.3f}")

            U_prev = U_current.copy()
            U_guess = U_prev.copy()

            for it in range(max_iter):
                U_guess = self.apply_dirichlet_bc(U_guess, mu1)
                R = self.compute_residual(U_guess, U_prev, dt, s)

                res_norm = np.linalg.norm(R[1:-1], ord=np.inf)
                print(f"  Newton iter {it+1:2d} | max residual = {res_norm:.3e}", end='')

                if res_norm < tol:
                    print("  Converged.")
                    break

                if use_fd_jacobian:
                    J = self.compute_jacobian_fd(U_guess, U_prev, dt, s)
                else:
                    J = self.compute_jacobian_analytical(U_guess, U_prev, dt, s)

                try:
                    dU = np.zeros_like(U_guess)
                    dU[1:-1] = np.linalg.solve(J[1:-1, 1:-1], -R[1:-1])
                except np.linalg.LinAlgError:
                    print(f"\n  Jacobian is singular at iteration {it+1}")
                    break

                rel_update = np.linalg.norm(dU[1:-1], ord=np.inf) / max(np.linalg.norm(U_guess[1:-1], ord=np.inf), 1e-15)
                print(f" | relative update = {rel_update:.2e}")

                U_new = U_guess + dU
                U_guess = U_new

                if rel_update < tol:
                    print("  Relative update small. Converged.")
                    break
            else:
                print("  Newton did not converge within max_iter.")

            U_current = self.apply_dirichlet_bc(U_guess, mu1)
            Uall[:, step + 1] = U_current

        return Uall
