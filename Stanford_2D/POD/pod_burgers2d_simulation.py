import numpy as np
import os
import sys

# Add the Burgers_2D directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Burgers_2D')))

# Now you can import the module
from burgers_fem import FEMBurgers2D

if __name__ == "__main__":
    # Specific combination of mu1 and mu2
    mu1 = 4.875
    mu2 = 0.0225

    # Choose the desired tolerance or number of modes
    tol = 1e-10  # Example: tolerance of 1e-10
    num_modes = 95  # Example: number of modes
    use_num_modes = True  # Set to True to use number of modes, False to use tolerance

    # Directory where the POD modes are saved
    modes_directory = "modes"
    
    # Load the POD modes (reduced basis) based on the chosen option
    if use_num_modes:
        pod_modes_file = f"U_modes_rsvd_num_modes_{num_modes}.npy"
    else:
        pod_modes_file = f"U_modes_tol_{tol:.0e}.npy"

    # Load the reduced basis (POD modes)
    Phi = np.load(os.path.join(modes_directory, pod_modes_file))
    
    # Time discretization and initial condition
    At = 0.05  # Time step size
    Tf = 2.0   # Final time
    nTimeSteps = int(Tf / At)
    u0 = np.ones((Phi.shape[0] // 2, 2))  # Initial condition for both velocity components (u_x and u_y)
    E = 0.2  # Example diffusion coefficient
    
    # Define domain and mesh based on previous FOM configuration
    a, b = 0, 100
    nx, ny = 250, 250
    x = np.linspace(a, b, nx + 1)
    y = np.linspace(a, b, ny + 1)
    X_grid, Y_grid = np.meshgrid(x, y)
    X, Y = X_grid.flatten(), Y_grid.flatten()

    # Generate the connectivity matrix T (from FOM settings)
    node_indices = np.arange((nx + 1) * (ny + 1)).reshape((ny + 1, nx + 1))
    T = []
    for i in range(ny):
        for j in range(nx):
            n1 = node_indices[i, j]
            n2 = node_indices[i, j + 1]
            n3 = node_indices[i + 1, j + 1]
            n4 = node_indices[i + 1, j]
            T.append([n1 + 1, n2 + 1, n3 + 1, n4 + 1])
    T = np.array(T)

    # Create an instance of the FEMBurgers2D class
    fem_burgers_2d = FEMBurgers2D(X, Y, T)

    # Determine which description to use for output messages
    descriptor = f"num_modes_{num_modes}" if use_num_modes else f"tol_{tol:.0e}"

    # Run the Reduced Order Model (PROM) using the reduced basis (POD modes)
    print(f"Running PROM for mu1={mu1}, mu2={mu2} with {descriptor} and {Phi.shape[1]} POD modes...")
    U_PROM = fem_burgers_2d.pod_prom_burgers(At, nTimeSteps, u0, mu1, E, mu2, Phi, X, Y, projection="LSPG")

    # Save the PROM solution
    rom_file_name = f"ROM_solution_mu1_{mu1:.3f}_mu2_{mu2:.4f}_{descriptor}_mesh_250x250.npy"
    np.save(os.path.join(modes_directory, rom_file_name), U_PROM)
    print(f"Saved ROM solution to {rom_file_name}")