import numpy as np
import os
import sys

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../FEM/'))
sys.path.append(parent_dir)
from fem_burgers import FEMBurgers

if __name__ == "__main__":
    # Domain and mesh
    a, b = 0, 100
    m = 511
    h = (b - a) / m
    X = np.linspace(a, b, m + 1)
    T = np.array([np.arange(1, m + 1), np.arange(2, m + 2)]).T

    # Initial condition
    u0 = np.ones_like(X)

    # Time discretization
    Tf = 25
    At = 0.05
    nTimeSteps = int(Tf / At)
    E = 0.00

    # Test points: each a tuple (mu1, mu2)
    test_points = [
        # (4.75, 0.0200),
        # (4.56, 0.0190),
        # (5.19, 0.0260)
        (6.20, 0.0400)
    ]

    # ROM tolerances
    tolerances = [1e-5]

    # Projection types
    projections = ["LSPG"]

    # Create an instance of the FEMBurgers class
    fem_burgers = FEMBurgers(X, T)

    # Directory to save results
    save_dir = "rom_solutions"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for mu1, mu2 in test_points:
        for tol in tolerances:
            for projection in projections:
                print(f'{projection} PROM with mu1={mu1}, mu2={mu2}, tolerance={tol}...')

                # Load reduced basis
                Phi = np.load(f"../modes/U_modes_tol_{tol:.0e}.npy")

                # Compute PROM solution
                U_PROM = fem_burgers.pod_prom_burgers(At, nTimeSteps, u0, mu1, E, mu2, Phi, projection=projection)

                # Prepare filename
                tag = "lspg" if projection.lower() == "lspg" else "galerkin"
                filename = f"U_PROM_tol_{tol:.0e}_mu1_{mu1:.3f}_mu2_{mu2:.4f}_{tag}.npy"

                # Save solution
                np.save(os.path.join(save_dir, filename), U_PROM)

    print("All simulations completed and saved.")

