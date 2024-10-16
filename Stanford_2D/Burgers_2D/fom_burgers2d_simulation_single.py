import numpy as np
import os
from burgers_fem import FEMBurgers2D  # Import the FEMBurgers2D class from your module

if __name__ == "__main__":
    # Domain and mesh
    a, b = 0, 100  # Define domain range
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
    
    # Save X and Y for plotting later
    np.save('X.npy', X)
    np.save('Y.npy', Y)
    
    # Initial conditions (assuming 2 components: u_x and u_y)
    u0 = np.ones((len(X), 2))  # Initialize u0 for both components
    
    # Time discretization settings
    Tf, At = 2.0, 0.05  # Final time and time step size
    nTimeSteps = int(Tf / At)
    E = 0.2  # Diffusion coefficient
    
    # Define parameter ranges and increments for mu1 and mu2
    mu1_range = np.arange(4.25, 5.50 + 0.625, 0.625)  # mu1 values
    mu2_range = np.arange(0.015, 0.03 + 0.0075, 0.0075)  # mu2 values
    
    # Create a list of all (mu1, mu2) pairs (3 x 3 grid)
    parameter_list = [(4.25, 0.015)]
    
    # Create an instance of the FEMBurgers2D class
    fem_burgers_2d = FEMBurgers2D(X, Y, T)
    
    # Directory to save results
    results_dir = "FOM_Solutions_test"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Loop through each (mu1, mu2) parameter combination and run the simulation
    for idx, (mu1, mu2) in enumerate(parameter_list):
        print(f"Running simulation for mu1={mu1}, mu2={mu2}")
        
        # Run the FOM simulation for the current parameter combination
        U_FOM = fem_burgers_2d.fom_burgers_2d(At, nTimeSteps, u0, mu1, E, mu2)
        
        # Define a file name including parameter values and mesh size
        file_name = f"U_FOM_mu1_{mu1:.3f}_mu2_{mu2:.4f}_mesh_{nx}x{ny}.npy"
        
        # Save only the U_FOM array in an .npy file
        np.save(os.path.join(results_dir, file_name), U_FOM)
        print(f"Saved U_FOM to {file_name}")

