import numpy as np
import os
import sys
import time  # Import time module for timing
from scipy.spatial import KDTree
import pickle

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
    Tf = 2.0  # Final time
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

    ### Run the POD-RBF simulation ###
    # POD-RBF parameters
    epsilon = 0.0001  # Example epsilon for the RBF kernel
    neighbors = 50  # Number of nearest neighbors for the RBF interpolation
    num_primary_modes = 10  # Number of primary modes to use
    num_secondary_modes = 140  # Number of secondary modes to use

    # Directories for modes and training data
    modes_directory = "modes"

    # Load primary and secondary POD modes
    U_p = np.load(os.path.join(modes_directory, "U_p.npy"))
    U_s = np.load(os.path.join(modes_directory, "U_s.npy"))

    # Load the training data
    with open(os.path.join(modes_directory, 'training_data.pkl'), 'rb') as f:
        data = pickle.load(f)
        q_p_train = data['q_p']
        q_s_train = data['q_s']

    # Create a KDTree for the training data
    kdtree = KDTree(q_p_train)

    print(f"Running POD-RBF PROM for mu1={mu1}, mu2={mu2} with {num_primary_modes} primary modes and {num_secondary_modes} secondary modes...")
    start_time_rbf = time.perf_counter()
    U_RBF_PROM = fem_burgers_2d.pod_rbf_prom_nearest_neighbours_dynamic_2d(
        At, nTimeSteps, u0, mu1, E, mu2, U_p, U_s, q_p_train, q_s_train, kdtree, epsilon, neighbors, projection="LSPG"
    )
    end_time_rbf = time.perf_counter()
    total_time_rbf = end_time_rbf - start_time_rbf
    print(f"POD-RBF PROM simulation time: {total_time_rbf:.2f} seconds")

    # Save the POD-RBF PROM solution
    rbf_rom_file_name = f"RBF_ROM_solution_mu1_{mu1:.3f}_mu2_{mu2:.4f}_num_primary_{num_primary_modes}_num_secondary_{num_secondary_modes}_mesh_{nx}x{ny}.npy"
    np.save(os.path.join(results_dir, rbf_rom_file_name), U_RBF_PROM)
    print(f"Saved POD-RBF PROM solution to {rbf_rom_file_name}")

    ### Compute the relative error ###
    if U_FOM.shape != U_RBF_PROM.shape:
        print("Error: The FOM and POD-RBF PROM solutions have different shapes.")
        error = None
    else:
        error = np.linalg.norm(U_FOM - U_RBF_PROM) / np.linalg.norm(U_FOM)
        print(f"Relative error between FOM and POD-RBF PROM solutions: {error:.6e}")

    ### Compute the speedup ###
    speedup = total_time_fom / total_time_rbf if total_time_rbf > 0 else None
    print(f"Speedup of POD-RBF PROM over FOM: {speedup:.2f}x")

    ### Save results to an .npz file ###
    results_file_name = f"Results_mu1_{mu1:.3f}_mu2_{mu2:.4f}_num_primary_{num_primary_modes}_num_secondary_{num_secondary_modes}.npz"
    np.savez(os.path.join(results_dir, results_file_name),
             total_time_fom=total_time_fom,
             total_time_rbf=total_time_rbf,
             speedup=speedup,
             relative_error=error)
    print(f"Saved results to {results_file_name}")
