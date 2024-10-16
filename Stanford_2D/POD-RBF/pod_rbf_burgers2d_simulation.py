import numpy as np
import os
import sys
from scipy.spatial import KDTree
import pickle

# Add the Burgers_2D directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Burgers_2D')))

# Import the FEMBurgers2D class
from burgers_fem import FEMBurgers2D

if __name__ == "__main__":
    # Specific combination of mu1 and mu2
    mu1 = 4.560
    mu2 = 0.0190

    # POD-RBF parameters
    epsilon = 0.01  # Example epsilon for the RBF kernel
    neighbors = 451  # Number of nearest neighbors for the RBF interpolation
    use_num_modes = True  # Set to True to use number of modes, False to use tolerance
    num_primary_modes = 10  # Number of primary modes to use
    num_secondary_modes = 140  # Number of secondary modes to use

    # Directories for modes and training data
    modes_directory = "modes"

    # Load primary and secondary POD modes
    U_p_file = f"U_p.npy"
    U_s_file = f"U_s.npy"
    U_p = np.load(os.path.join(modes_directory, U_p_file))
    U_s = np.load(os.path.join(modes_directory, U_s_file))

    # Load the training data
    with open(os.path.join(modes_directory, 'training_data.pkl'), 'rb') as f:
        data = pickle.load(f)
        q_p_train = data['q_p']
        q_s_train = data['q_s']

    # Create a KDTree for the training data
    kdtree = KDTree(q_p_train)

    # Time discretization and initial condition
    At = 0.05  # Time step size
    Tf = 2.0   # Final time
    nTimeSteps = int(Tf / At)
    u0 = np.ones((U_p.shape[0] // 2, 2))  # Initial condition for both velocity components (u_x and u_y)
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

    # Run the POD-RBF PROM using the reduced basis (POD modes) and RBF interpolation
    print(f"Running POD-RBF PROM for mu1={mu1}, mu2={mu2} with {num_primary_modes} primary modes and {num_secondary_modes} secondary modes...")
    U_RBF_PROM = fem_burgers_2d.pod_rbf_prom_nearest_neighbours_dynamic_2d(
        At, nTimeSteps, u0, mu1, E, mu2, U_p, U_s, q_p_train, q_s_train, kdtree, epsilon, neighbors, projection="LSPG"
    )

    # Save the POD-RBF PROM solution
    rom_file_name = f"RBF_ROM_solution_mu1_{mu1:.3f}_mu2_{mu2:.4f}_num_primary_{num_primary_modes}_num_secondary_{num_secondary_modes}_mesh_250x250.npy"
    np.save(os.path.join(modes_directory, rom_file_name), U_RBF_PROM)
    print(f"Saved POD-RBF PROM solution to {rom_file_name}")
