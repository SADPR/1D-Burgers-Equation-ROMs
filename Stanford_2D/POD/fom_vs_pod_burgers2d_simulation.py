import numpy as np
import os
import sys
import time  # Import time module for timing

# Add the Burgers_2D directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Burgers_2D')))

# Now you can import the module
from burgers_fem import FEMBurgers2D

if __name__ == "__main__":
    # Domain and mesh parameters
    a, b = 0, 100  # Domain range
    nx, ny = 250, 250  # Number of grid points in x and y directions

    # Create grid points in x and y directions
    x = np.linspace(a, b, nx + 1)
    y = np.linspace(a, b, ny + 1)
    X_grid, Y_grid = np.meshgrid(x, y)
    X, Y = X_grid.flatten(), Y_grid.flatten()  # Flatten the meshgrid into 1D arrays

    # Generate the connectivity matrix T
    node_indices = np.arange((nx + 1) * (ny + 1)).reshape((ny + 1, nx + 1))
    T = []
    for i in range(ny):
        for j in range(nx):
            n1 = node_indices[i, j]
            n2 = node_indices[i, j + 1]
            n3 = node_indices[i + 1, j + 1]
            n4 = node_indices[i + 1, j]
            T.append([n1 + 1, n2 + 1, n3 + 1, n4 + 1])  # +1 for 1-based indexing (required by FEM)
    T = np.array(T)

    # Initial conditions (assuming 2 components: u_x and u_y)
    u0 = np.ones((len(X), 2))  # Initialize u0 for both components

    # Time discretization settings
    At = 0.05  # Time step size
    Tf = 25.0  # Final time
    nTimeSteps = int(Tf / At)
    E = 0.2  # Diffusion coefficient

    # Choose specific parameter values for mu1 and mu2
    mu1 = 4.560   # Example value for mu1
    mu2 = 0.0190  # Example value for mu2

    # Create an instance of the FEMBurgers2D class
    fem_burgers_2d = FEMBurgers2D(X, Y, T)

    # Directory to save results
    results_dir = "FOM_vs_ROM_Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    ### Run the FOM simulation ###
    print(f"Running FOM simulation for mu1={mu1}, mu2={mu2}")
    start_time_fom = time.perf_counter()
    U_FOM = fem_burgers_2d.fom_burgers_2d(At, nTimeSteps, u0, mu1, E, mu2)
    end_time_fom = time.perf_counter()
    total_time_fom = end_time_fom - start_time_fom
    print(f"FOM simulation time: {total_time_fom:.2f} seconds")

    # Save the FOM solution
    fom_file_name = f"U_FOM_mu1_{mu1:.3f}_mu2_{mu2:.4f}_mesh_{nx}x{ny}.npy"
    np.save(os.path.join(results_dir, fom_file_name), U_FOM)
    print(f"Saved FOM solution to {fom_file_name}")

    ### Run the ROM simulation ###

    # Choose the desired tolerance or number of modes
    tol = 1e-10     # Example: tolerance of 1e-10
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

    # Determine which description to use for output messages
    descriptor = f"num_modes_{num_modes}" if use_num_modes else f"tol_{tol:.0e}"

    print(f"Running ROM simulation for mu1={mu1}, mu2={mu2} with {descriptor} and {Phi.shape[1]} POD modes...")
    start_time_rom = time.perf_counter()
    U_ROM = fem_burgers_2d.pod_prom_burgers(At, nTimeSteps, u0, mu1, E, mu2, Phi, X, Y, projection="LSPG")
    end_time_rom = time.perf_counter()
    total_time_rom = end_time_rom - start_time_rom
    print(f"ROM simulation time: {total_time_rom:.2f} seconds")

    # Save the ROM solution
    rom_file_name = f"U_ROM_mu1_{mu1:.3f}_mu2_{mu2:.4f}_{descriptor}_mesh_{nx}x{ny}.npy"
    np.save(os.path.join(results_dir, rom_file_name), U_ROM)
    print(f"Saved ROM solution to {rom_file_name}")

    ### Compute the relative error ###
    # Ensure that U_FOM and U_ROM have the same shape
    if U_FOM.shape != U_ROM.shape:
        print("Error: The FOM and ROM solutions have different shapes.")
        error = None
    else:
        # Compute the relative error
        error = np.linalg.norm(U_FOM - U_ROM) / np.linalg.norm(U_FOM)
        print(f"Relative error between FOM and ROM solutions: {error:.6e}")

    ### Compute the speedup ###
    speedup = total_time_fom / total_time_rom if total_time_rom > 0 else None
    print(f"Speedup of ROM over FOM: {speedup:.2f}x")

    ### Save results to an .npz file ###
    results_file_name = f"Results_mu1_{mu1:.3f}_mu2_{mu2:.4f}_{descriptor}.npz"
    np.savez(os.path.join(results_dir, results_file_name),
             total_time_fom=total_time_fom,
             total_time_rom=total_time_rom,
             speedup=speedup,
             relative_error=error)
    print(f"Saved results to {results_file_name}")
