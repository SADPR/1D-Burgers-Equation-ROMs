import numpy as np
import os
import time
from matplotlib import pyplot as plt
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def plot_3d_surface_animation(U_FOM, U_ROM, X, Y, filename='3d_surface_animation.gif', num_timesteps=100):
    """Create an animation of side-by-side 3D surface plots of FOM and ROM over specified timesteps."""
    n_nodes = len(X)
    nTimeSteps_total = U_FOM.shape[1] - 1  # Exclude initial condition
    nTimeSteps = min(nTimeSteps_total, num_timesteps)

    x_values = np.unique(X)
    y_values = np.unique(Y)
    X_grid, Y_grid = np.meshgrid(x_values, y_values)

    # Prepare data for plotting
    U_FOM_frames = [U_FOM[:, n].reshape(len(y_values), len(x_values)) for n in range(nTimeSteps + 1)]
    U_ROM_frames = [U_ROM[:, n].reshape(len(y_values), len(x_values)) for n in range(nTimeSteps + 1)]

    # Set up the figure and axes
    fig = plt.figure(figsize=(10, 5))  # Adjust figure size as needed
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    def update_surface(n):
        ax1.clear()
        ax2.clear()

        # Plot FOM
        ax1.plot_surface(X_grid, Y_grid, U_FOM_frames[n], cmap='viridis', edgecolor='none')
        ax1.set_title(f'FOM at Time Step {n}')
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$y$')
        ax1.set_zlabel(r'$u_x(x, y)$')
        ax1.view_init(30, -60)

        # Plot ROM
        ax2.plot_surface(X_grid, Y_grid, U_ROM_frames[n], cmap='viridis', edgecolor='none')
        ax2.set_title(f'ROM Reconstruction at Time Step {n}')
        ax2.set_xlabel(r'$x$')
        ax2.set_ylabel(r'$y$')
        ax2.set_zlabel(r'$u_x(x, y)$')
        ax2.view_init(30, -60)

        return fig,

    ani = animation.FuncAnimation(fig, update_surface, frames=nTimeSteps + 1, blit=False)
    ani.save(filename, writer='pillow', fps=15)
    plt.close()
    print(f"Saved 3D surface animation to {filename}")

def plot_2d_cuts_animation_overlay(U_FOM, U_ROM, X, Y, line='x', fixed_value=50.0, filename='cuts_animation_overlay.gif', num_timesteps=500):
    """Create an animation of 2D cuts over specified timesteps, overlaying FOM and ROM in the same plot."""
    n_nodes = len(X)
    nTimeSteps_total = U_FOM.shape[1] - 1  # Exclude initial condition
    nTimeSteps = min(nTimeSteps_total, num_timesteps)

    # Adjust figure size to be rectangular (height: 3 inches, width: 10 inches)
    fig, ax = plt.subplots(figsize=(10, 3))

    if line == 'x':
        indices = np.where(np.isclose(Y, fixed_value))[0]
        x_values = X[indices]
        u_fom_values_per_frame = [U_FOM[:, n][indices] for n in range(nTimeSteps + 1)]
        u_rom_values_per_frame = [U_ROM[:, n][indices] for n in range(nTimeSteps + 1)]

        def update_cut(n):
            ax.clear()
            u_fom_values = u_fom_values_per_frame[n]
            u_rom_values = u_rom_values_per_frame[n]
            ax.plot(x_values, u_fom_values, 'k-', label='FOM')  # Black solid line for FOM
            ax.plot(x_values, u_rom_values, '-', color='darkgreen', label='ROM')  # Dark green solid line for ROM
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(rf'$u_x(x, y={fixed_value})$')
            ax.set_title(f'2D Cut along y = {fixed_value} at Time Step {n}', fontsize=12)
            ax.legend()
            ax.grid(True)
            ax.set_ylim(0, 6)  # Set y-axis limits from 0 to 6
            return ax.lines

    elif line == 'y':
        indices = np.where(np.isclose(X, fixed_value))[0]
        y_values = Y[indices]
        u_fom_values_per_frame = [U_FOM[:, n][indices] for n in range(nTimeSteps + 1)]
        u_rom_values_per_frame = [U_ROM[:, n][indices] for n in range(nTimeSteps + 1)]

        def update_cut(n):
            ax.clear()
            u_fom_values = u_fom_values_per_frame[n]
            u_rom_values = u_rom_values_per_frame[n]
            ax.plot(y_values, u_fom_values, 'k-', label='FOM')  # Black solid line for FOM
            ax.plot(y_values, u_rom_values, '-', color='darkgreen', label='ROM')  # Dark green solid line for ROM
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(rf'$u_x(x={fixed_value}, y)$')
            ax.set_title(f'2D Cut along x = {fixed_value} at Time Step {n}', fontsize=12)
            ax.legend()
            ax.grid(True)
            ax.set_ylim(0, 6)  # Set y-axis limits from 0 to 6
            return ax.lines

    ani = animation.FuncAnimation(fig, update_cut, frames=nTimeSteps + 1, blit=True)
    ani.save(filename, writer='pillow', fps=15)
    plt.close()
    print(f"Saved overlay cuts animation to {filename}")

def get_number_of_modes_for_tolerance(U_FOM, U, s, tolerance, check_reconstruction=False):
    total_energy = np.sum(s ** 2)
    cumulative_energy = np.cumsum(s ** 2)
    
    for num_modes in range(1, len(s) + 1):
        Phi = U[:, :num_modes]
        U_approx = Phi @ (Phi.T @ U_FOM)
        relative_error = np.linalg.norm(U_FOM - U_approx) / np.linalg.norm(U_FOM)
        
        if check_reconstruction:
            print(f"Reconstruction check: {num_modes} modes give a relative error of {relative_error:.6e}")
        
        if relative_error <= tolerance:
            print(f"Tolerance {tolerance:.1e}: {num_modes} modes yield an error of {relative_error:.6e}")
            return num_modes, relative_error

    return len(s), relative_error

def save_modes(U, s, svd_method, modes_for_tolerances=None, num_modes=None, check_reconstruction=False, original_matrix=None):
    """
    Save the modes based on the method chosen.
        
    Parameters:
    U - Left singular vectors (POD modes) from SVD
    s - Singular values from SVD
    svd_method - The SVD method used ('numpy', 'rsvd', or 'truncated')
    modes_for_tolerances - List of tuples (tolerance, num_modes) for saving modes (only for 'numpy')
    num_modes - Number of modes for 'rsvd' or 'truncated' methods
    check_reconstruction - Whether to check the reconstruction accuracy
    original_matrix - The original snapshot matrix for reconstruction checking
    """
    if not os.path.exists("modes"):
        os.makedirs("modes")

    if svd_method == 'numpy':
        for tol, num_modes in modes_for_tolerances:
            start_time = time.time()
            U_modes = U[:, :num_modes]
            np.save(f"modes/U_modes_tol_{tol:.0e}.npy", U_modes)
            if check_reconstruction and original_matrix is not None:
                Phi = U_modes
                U_approx = Phi @ (Phi.T @ original_matrix)
                reconstruction_error = np.linalg.norm(original_matrix - U_approx) / np.linalg.norm(original_matrix)
                print(f'Reconstruction error for tolerance {tol:.0e} with {num_modes} modes: {reconstruction_error:.6e}')
            print(f'Saved modes for tolerance {tol:.0e} with {num_modes} modes. Time taken: {time.time() - start_time:.2f} seconds.')
    else:
        start_time = time.time()
        U_modes = U[:, :num_modes]
        np.save(f"modes/U_modes_{svd_method}_num_modes_{num_modes}.npy", U_modes)
        if check_reconstruction and original_matrix is not None:
            Phi = U_modes
            U_approx = Phi @ (Phi.T @ original_matrix)
            reconstruction_error = np.linalg.norm(original_matrix - U_approx) / np.linalg.norm(original_matrix)
            print(f'Reconstruction error for {svd_method} with {num_modes} modes: {reconstruction_error:.6e}')
        print(f'Saved {num_modes} modes for {svd_method}. Time taken: {time.time() - start_time:.2f} seconds.')
        # Domain and mesh parameters
        a, b = 0, 100  # Domain range
        nx, ny = 250, 250  # Number of grid points in x and y directions

        # Create grid points in x and y directions
        x = np.linspace(a, b, nx + 1)
        y = np.linspace(a, b, ny + 1)
        X_grid, Y_grid = np.meshgrid(x, y)
        X, Y = X_grid.flatten(), Y_grid.flatten()  # Flatten the meshgrid into 1D arrays
        plot_3d_surface_animation(original_matrix[:int(original_matrix.shape[0]/2),:], U_approx[:int(original_matrix.shape[0]/2),:], X, Y, filename='3d_surface_animation.gif', num_timesteps=100)

        # Overlay cuts animation
        plot_2d_cuts_animation_overlay(original_matrix[:int(original_matrix.shape[0]/2),:], U_approx[:int(original_matrix.shape[0]/2),:], X, Y, line='x', fixed_value=50.0, filename='cuts_animation_overlay.gif', num_timesteps=100)


def compute_svd(SnapshotsMatrix, svd_method='numpy', n_components=100):
    if svd_method == 'numpy':
        U, s, Vh = np.linalg.svd(SnapshotsMatrix, full_matrices=False)
    elif svd_method == 'rsvd':
        U, s, Vh = randomized_svd(SnapshotsMatrix, n_components=n_components)
    elif svd_method == 'truncated':
        svd = TruncatedSVD(n_components=n_components)
        U = svd.fit_transform(SnapshotsMatrix)
        s = svd.singular_values_
        Vh = svd.components_
    else:
        raise ValueError("Invalid SVD method. Choose from 'numpy', 'rsvd', or 'truncated'.")
    return U, s, Vh

if __name__ == '__main__':
    overall_start_time = time.time()
    
    if not os.path.exists("modes"):
        os.makedirs("modes")

    fom_directory = "../Burgers_2D/FOM_Solutions"
    snapshot_files = [f for f in os.listdir(fom_directory) if f.endswith('.npy') and f.startswith('U_FOM')]
    # snapshot_files = [snapshot_files[0]]

    load_start_time = time.time()
    all_snapshots = []
    for file in snapshot_files:
        U_FOM = np.load(os.path.join(fom_directory, file))  
        n_time_steps_plus_one, total_dofs = U_FOM.shape
        SnapshotMatrix = U_FOM.T  
        all_snapshots.append(SnapshotMatrix)

    all_snapshots = np.hstack(all_snapshots)
    print(f"Loading and reshaping snapshots took {time.time() - load_start_time:.2f} seconds.")
    print(f"Snapshot matrix shape: {all_snapshots.shape}")

    svd_start_time = time.time()
    svd_method = 'rsvd'  # Choose 'numpy', 'rsvd', or 'truncated'
    n_components = 95   # Number of modes for 'rsvd' or 'truncated' methods
    U, s, Vh = compute_svd(all_snapshots, svd_method=svd_method, n_components=n_components)
    print(f"SVD computation ({svd_method}) took {time.time() - svd_start_time:.2f} seconds.")

    print("Singular values:")
    # print(s)

    check_reconstruction = True  # Set to True to check reconstruction accuracy

    if svd_method == 'numpy':
        tols = [1e-3, 1e-4, 1e-6, 1e-10]
        modes_for_tolerances = []

        for tol in tols:
            tol_start_time = time.time()
            num_modes, error = get_number_of_modes_for_tolerance(all_snapshots, U, s, tol, check_reconstruction=check_reconstruction)
            modes_for_tolerances.append((tol, num_modes))
            print(f"For tolerance {tol}, the reconstruction error is {error:.6e} with {num_modes} modes. Time taken: {time.time() - tol_start_time:.2f} seconds.")

        save_modes(U, s, svd_method, modes_for_tolerances=modes_for_tolerances, check_reconstruction=check_reconstruction, original_matrix=all_snapshots)
    else:
        save_modes(U, s, svd_method, num_modes=n_components, check_reconstruction=check_reconstruction, original_matrix=all_snapshots)

    print(f"Total time taken: {time.time() - overall_start_time:.2f} seconds.")
