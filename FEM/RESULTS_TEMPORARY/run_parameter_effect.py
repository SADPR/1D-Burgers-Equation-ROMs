import numpy as np
import os
from fem_burgers import FEMBurgers  # Import the FEMBurgers class

def run_and_save_simulations():
    # Domain
    a = 0
    b = 100

    # Mesh
    m = int(256 * 2)
    h = (b - a) / m
    X = np.linspace(a, b, m + 1)
    T = np.array([np.arange(1, m + 1), np.arange(2, m + 2)]).T

    # Initial condition
    u0 = np.ones_like(X)

    # Time discretization and numerical diffusion
    Tf = 20  # Covering up to t=20
    At = 0.07
    nTimeSteps = int(Tf / At)
    E = 0.01

    # Parameters ranges
    mu1_values = np.linspace(4.25, 5.5, 5)  # Vary mu1 across 5 values
    mu2_values = np.linspace(0.015, 0.03, 5)  # Vary mu2 across 5 values

    # Initialize the FEMBurgers class
    fem_burgers = FEMBurgers(X, T)

    # Create directory to save the results
    save_dir = "simulation_results"
    os.makedirs(save_dir, exist_ok=True)

    # Run simulations for varying mu1 with fixed mu2
    fixed_mu2 = mu2_values[2]  # Choose a middle value of mu2
    for mu1 in mu1_values:
        U_FOM = fem_burgers.fom_burgers(At, nTimeSteps, u0, mu1, E, fixed_mu2)
        filename = f"{save_dir}/U_FOM_mu1_{mu1:.2f}_mu2_{fixed_mu2:.4f}.npy"
        np.save(filename, U_FOM)

    # Run simulations for varying mu2 with fixed mu1
    fixed_mu1 = mu1_values[2]  # Choose a middle value of mu1
    for mu2 in mu2_values:
        U_FOM = fem_burgers.fom_burgers(At, nTimeSteps, u0, fixed_mu1, E, mu2)
        filename = f"{save_dir}/U_FOM_mu1_{fixed_mu1:.2f}_mu2_{mu2:.4f}.npy"
        np.save(filename, U_FOM)

if __name__ == "__main__":
    run_and_save_simulations()



