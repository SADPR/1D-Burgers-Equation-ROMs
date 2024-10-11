import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation

def plot_3d_surface_animation(U_FOM, U_POD_RBF, X, Y, mu1, mu2, num_modes, filename='3d_surface_animation.gif', max_frames=500, num_time_steps=None):
    """Create an animation of side-by-side 3D surface plots of FOM and POD-RBF over time."""
    n_nodes = len(X)
    nTimeSteps = U_FOM.shape[1] - 1 if num_time_steps is None else min(num_time_steps, U_FOM.shape[1] - 1)
    nTimeSteps = min(nTimeSteps, max_frames)
    time_step_size = 0.05
    
    x_values = np.unique(X)
    y_values = np.unique(Y)
    X_grid, Y_grid = np.meshgrid(x_values, y_values)
    
    U_FOM_frames = [U_FOM[:n_nodes, n].reshape(len(y_values), len(x_values)) for n in range(nTimeSteps + 1)]
    U_POD_RBF_frames = [U_POD_RBF[:n_nodes, n].reshape(len(y_values), len(x_values)) for n in range(nTimeSteps + 1)]

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    def update_surface(n):
        time = n * time_step_size
        ax1.clear()
        ax2.clear()
        
        ax1.plot_surface(X_grid, Y_grid, U_FOM_frames[n], cmap='viridis', edgecolor='none')
        ax1.set_title('FOM')
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$y$')
        ax1.set_zlabel(r'$u_x(x, y)$')
        ax1.view_init(30, -60)
        
        ax2.plot_surface(X_grid, Y_grid, U_POD_RBF_frames[n], cmap='viridis', edgecolor='none')
        ax2.set_title('POD-RBF 10 Primary + 140 Secondary Modes')
        ax2.set_xlabel(r'$x$')
        ax2.set_ylabel(r'$y$')
        ax2.set_zlabel(r'$u_x(x, y)$')
        ax2.view_init(30, -60)
        
        fig.suptitle(f'Time = {time:.2f} s', fontsize=16, y=0.95)
        return fig,

    ani = animation.FuncAnimation(fig, update_surface, frames=nTimeSteps + 1, blit=False)
    filename = f"3d_surface_animation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.gif"
    ani.save(filename, writer='pillow', fps=30)
    plt.close()
    print(f"Saved 3D surface animation to {filename}")

def plot_2d_cuts_animation(U_FOM, U_POD_RBF, X, Y, line, fixed_value, mu1, mu2, num_modes, filename, max_frames=500, num_time_steps=None):
    """Create an animation of 2D cuts over time for FOM and POD-RBF in the same plot."""
    n_nodes = len(X)
    nTimeSteps = U_FOM.shape[1] - 1 if num_time_steps is None else min(num_time_steps, U_FOM.shape[1] - 1)
    nTimeSteps = min(nTimeSteps, max_frames)
    time_step_size = 0.05
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if line == 'x':
        indices = np.where(np.isclose(Y, fixed_value))[0]
        x_values = X[indices]
        u_fom_values_per_frame = [U_FOM[:n_nodes, n][indices] for n in range(nTimeSteps + 1)]
        u_pod_rbf_values_per_frame = [U_POD_RBF[:n_nodes, n][indices] for n in range(nTimeSteps + 1)]

        def update_cut(n):
            time = n * time_step_size
            ax.clear()
            u_fom_values = u_fom_values_per_frame[n]
            u_pod_rbf_values = u_pod_rbf_values_per_frame[n]
            ax.plot(x_values, u_fom_values, 'k-', linewidth=2.5, label='FOM')
            ax.plot(x_values, u_pod_rbf_values, 'g-', linewidth=2, label='POD-RBF 10+140 Modes')
            ax.set_ylim(0.5, 5.5)
            ax.set_title(f'2D Cut along y = {fixed_value}, Time = {time:.2f} s')
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$u_x(x, y)$')
            ax.legend()
            ax.grid()
            return ax,

    elif line == 'y':
        indices = np.where(np.isclose(X, fixed_value))[0]
        y_values = Y[indices]
        u_fom_values_per_frame = [U_FOM[:n_nodes, n][indices] for n in range(nTimeSteps + 1)]
        u_pod_rbf_values_per_frame = [U_POD_RBF[:n_nodes, n][indices] for n in range(nTimeSteps + 1)]

        def update_cut(n):
            time = n * time_step_size
            ax.clear()
            u_fom_values = u_fom_values_per_frame[n]
            u_pod_rbf_values = u_pod_rbf_values_per_frame[n]
            ax.plot(y_values, u_fom_values, 'k-', linewidth=2.5, label='FOM')
            ax.plot(y_values, u_pod_rbf_values, 'g-', linewidth=2, label='POD-RBF 10+140 Modes')
            ax.set_ylim(0.5, 5.5)
            ax.set_title(f'2D Cut along x = {fixed_value}, Time = {time:.2f} s')
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$u_x(x, y)$')
            ax.legend()
            ax.grid()
            return ax,

    ani = animation.FuncAnimation(fig, update_cut, frames=nTimeSteps + 1, blit=False)
    filename = f"{filename}_mu1_{mu1:.3f}_mu2_{mu2:.4f}.gif"
    ani.save(filename, writer='pillow', fps=30)
    plt.close()
    print(f"Saved 2D cuts animation to {filename}")

# Example usage
if __name__ == "__main__":
    # Parameters
    mu1 = 4.560
    mu2 = 0.0190
    num_modes = 95
    fixed_value = 50.0
    num_time_steps = 40  # Specify the number of time steps to include in the animation

    # Domain and mesh parameters
    a, b = 0, 100  # Domain range
    nx, ny = 250, 250  # Number of grid points in x and y directions
    x = np.linspace(a, b, nx + 1)
    y = np.linspace(a, b, ny + 1)
    X, Y = np.meshgrid(x, y)
    X, Y = X.flatten(), Y.flatten()

    # Load data
    U_FOM = np.load(os.path.join(f'../../POD/FOM_vs_ROM_Results/U_FOM_mu1_4.560_mu2_0.0190_mesh_250x250.npy')).T
    U_POD_RBF = np.load(os.path.join(f"RBF_ROM_solution_mu1_4.560_mu2_0.0190_num_primary_10_num_secondary_140_mesh_250x250.npy")).T

    # Generate visualizations
    plot_3d_surface_animation(U_FOM, U_POD_RBF, X, Y, mu1, mu2, num_modes, num_time_steps=num_time_steps)
    plot_2d_cuts_animation(U_FOM, U_POD_RBF, X, Y, 'x', fixed_value, mu1, mu2, num_modes, '2d_cuts_x_animation', num_time_steps=num_time_steps)
    plot_2d_cuts_animation(U_FOM, U_POD_RBF, X, Y, 'y', fixed_value, mu1, mu2, num_modes, '2d_cuts_y_animation', num_time_steps=num_time_steps)


