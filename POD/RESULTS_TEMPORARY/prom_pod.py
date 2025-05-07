import numpy as np
import os
import sys
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../FEM/'))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Now you can import the module
from fem_burgers import FEMBurgers

if __name__ == "__main__":
    # Domain and mesh
    a, b = 0, 100

    # Mesh
    m = 511
    h = (b - a) / m
    X = np.linspace(a, b, m + 1)
    T = np.array([np.arange(1, m + 1), np.arange(2, m + 2)]).T

    # Initial condition
    u0 = np.ones_like(X)

    # Initial condition
    u0 = np.ones_like(X)

    # Time discretization
    Tf = 25
    At = 0.05
    nTimeSteps = int(Tf / At)
    E = 0.00

    # Boundary conditions
    mu1 = 4.750  # u(0,t) = 4.750

    # Parameter mu2
    mu2 = 0.0200

    # Tolerances for ROM
    tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    # Create an instance of the FEMBurgers class
    fem_burgers = FEMBurgers(X, T)

    # Directory to save results
    save_dir = "rom_solutions"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for tol in tolerances:
        print(f'PROM method (LSPG) with tolerance {tol}...')
        
        # Load reduced basis for the given tolerance
        Phi = np.load(f"../modes/U_modes_tol_{tol:.0e}.npy")

        # Compute the PROM solution
        U_PROM = fem_burgers.pod_prom_burgers(At, nTimeSteps, u0, mu1, E, mu2, Phi, projection="LSPG")

        # Save the solution
        np.save(os.path.join(save_dir, f"U_PROM_tol_{tol:.0e}.npy"), U_PROM)

    print("All simulations completed and saved.")
