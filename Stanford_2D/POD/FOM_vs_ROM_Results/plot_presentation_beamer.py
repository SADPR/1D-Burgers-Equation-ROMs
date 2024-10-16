import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation

def plot_3d_surface_animation(U_FOM, U_POD, X, Y, mu1, mu2, num_modes, filename='3d_surface_animation.mp4', max_frames=500):
    """Create an animation of side-by-side 3D surface plots of FOM and POD over time."""
    n_nodes = len(X)
    nTimeSteps = U_FOM.shape[1] - 1
    nTimeSteps = min(nTimeSteps, max_frames)
    time_step_size = 0.05
    
    x_values = np.unique(X)
    y_values = np.unique(Y)
    X_grid, Y_grid = np.meshgrid(x_values, y_values)
    
    U_FOM_frames = [U_FOM[:n_nodes, n].reshape(len(y_values), len(x_values)) for n in range(nTimeSteps + 1)]
    U_POD_frames = [U_POD[:n_nodes, n].reshape(len(y_values), len(x_values)) for n in range(nTimeSteps + 1)]

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
        
        ax2.plot_surface(X_grid, Y_grid, U_POD_frames[n], cmap='viridis', edgecolor='none')
        ax2.set_title('POD 10 Primary + 95 Secondary Modes')
        ax2.set_xlabel(r'$x$')
        ax2.set_ylabel(r'$y$')
        ax2.set_zlabel(r'$u_x(x, y)$')
        ax2.view_init(30, -60)
        
        fig.suptitle(f'Time = {time:.2f} s', fontsize=16, y=0.95)
        return fig,

    ani = animation.FuncAnimation(fig, update_surface, frames=nTimeSteps + 1, blit=False)
    filename = f"3d_surface_animation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.mp4"
    ani.save(filename, writer='ffmpeg', fps=30)
    plt.close()
    print(f"Saved 3D surface animation to {filename}")


def plot_2d_cuts_animation(U_FOM, U_POD, X, Y, line, fixed_value, mu1, mu2, num_modes, filename, max_frames=500):
    """Create an animation of 2D cuts over time for FOM and POD in the same plot."""
    n_nodes = len(X)
    nTimeSteps = U_FOM.shape[1] - 1
    nTimeSteps = min(nTimeSteps, max_frames)
    time_step_size = 0.05
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if line == 'x':
        indices = np.where(np.isclose(Y, fixed_value))[0]
        x_values = X[indices]
        u_fom_values_per_frame = [U_FOM[:n_nodes, n][indices] for n in range(nTimeSteps + 1)]
        u_pod_values_per_frame = [U_POD[:n_nodes, n][indices] for n in range(nTimeSteps + 1)]

        def update_cut(n):
            time = n * time_step_size
            ax.clear()
            u_fom_values = u_fom_values_per_frame[n]
            u_pod_values = u_pod_values_per_frame[n]
            ax.plot(x_values, u_fom_values, 'k-', linewidth=2.5, label='FOM')
            ax.plot(x_values, u_pod_values, 'r-', linewidth=2, label='POD 95 Modes')
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
        u_pod_values_per_frame = [U_POD[:n_nodes, n][indices] for n in range(nTimeSteps + 1)]

        def update_cut(n):
            time = n * time_step_size
            ax.clear()
            u_fom_values = u_fom_values_per_frame[n]
            u_pod_values = u_pod_values_per_frame[n]
            ax.plot(y_values, u_fom_values, 'k-', linewidth=2.5, label='FOM')
            ax.plot(y_values, u_pod_values, 'r-', linewidth=2, label='POD 95 Modes')
            ax.set_ylim(0.5, 5.5)
            ax.set_title(f'2D Cut along x = {fixed_value}, Time = {time:.2f} s')
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$u_x(x, y)$')
            ax.legend()
            ax.grid()
            return ax,

    ani = animation.FuncAnimation(fig, update_cut, frames=nTimeSteps + 1, blit=False)
    filename = f"pod_{filename}_mu1_{mu1:.3f}_mu2_{mu2:.4f}.mp4"
    ani.save(filename, writer='ffmpeg', fps=30)
    plt.close()
    print(f"Saved 2D cuts animation to {filename}")


def plot_2d_heatmap_animation(U_FOM, U_POD, X, Y, mu1, mu2, filename='heatmap_animation.mp4', max_frames=500):
    """Create an animation of side-by-side heatmaps of FOM and POD over time."""
    n_nodes = len(X)
    nTimeSteps = U_FOM.shape[1] - 1
    nTimeSteps = min(nTimeSteps, max_frames)
    time_step_size = 0.05
    
    x_values = np.unique(X)
    y_values = np.unique(Y)
    X_grid, Y_grid = np.meshgrid(x_values, y_values)
    
    U_FOM_frames = [U_FOM[:n_nodes, n].reshape(len(y_values), len(x_values)) for n in range(nTimeSteps + 1)]
    U_POD_frames = [U_POD[:n_nodes, n].reshape(len(y_values), len(x_values)) for n in range(nTimeSteps + 1)]

    vmin = min(U_FOM[:n_nodes, :].min(), U_POD[:n_nodes, :].min())
    vmax = max(U_FOM[:n_nodes, :].max(), U_POD[:n_nodes, :].max())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    def update_heatmap(n):
        time = n * time_step_size
        ax1.clear()
        ax2.clear()
        
        im1 = ax1.pcolormesh(X_grid, Y_grid, U_FOM_frames[n], cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
        im2 = ax2.pcolormesh(X_grid, Y_grid, U_POD_frames[n], cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
        ax1.set_title(f'FOM')
        ax2.set_title(f'POD 95 Modes')
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        fig.suptitle(f'Time = {time:.2f} s', fontsize=16, y=0.95)
        return [im1, im2]

    ani = animation.FuncAnimation(fig, update_heatmap, frames=nTimeSteps + 1, blit=False)
    filename = f"heatmap_animation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.mp4"
    ani.save(filename, writer='ffmpeg', fps=30)
    plt.close()
    print(f"Saved heatmap animation to {filename}")


def plot_2d_cuts_overlay(U_FOM, U_POD, X, Y, line, fixed_value, mu1, mu2, num_modes, filename, time_steps, time_step_size=0.05):
    """Create static overlays of 2D cuts for FOM and POD at specified time steps."""
    n_nodes = len(X)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if line == 'x':
        indices = np.where(np.isclose(Y, fixed_value))[0]
        x_values = X[indices]
        
        for t in time_steps:
            index = int(t / time_step_size)
            u_fom_values = U_FOM[:n_nodes, index][indices]
            u_pod_values = U_POD[:n_nodes, index][indices]
            ax.plot(x_values, u_fom_values, 'k-', linewidth=2.5, label='FOM' if t == time_steps[0] else "")
            ax.plot(x_values, u_pod_values, 'r-', linewidth=2, label='POD 95 Modes' if t == time_steps[0] else "")
            
        ax.set_title(f'2D Cut Overlay along y={fixed_value}')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$u_x(x, y)$')
        ax.legend()
        ax.grid()
        ax.set_ylim(0.5, 5.5)
    
    elif line == 'y':
        indices = np.where(np.isclose(X, fixed_value))[0]
        y_values = Y[indices]
        
        for t in time_steps:
            index = int(t / time_step_size)
            u_fom_values = U_FOM[:n_nodes, index][indices]
            u_pod_values = U_POD[:n_nodes, index][indices]
            ax.plot(y_values, u_fom_values, 'k-', linewidth=2.5, label='FOM' if t == time_steps[0] else "")
            ax.plot(y_values, u_pod_values, 'r-', linewidth=2, label='POD 95 Modes' if t == time_steps[0] else "")
        
        ax.set_title(f'2D Cut Overlay along x={fixed_value}')
        ax.set_xlabel(r'$y$')
        ax.set_ylabel(r'$u_x(x, y)$')
        ax.legend()
        ax.grid()
        ax.set_ylim(0.5, 5.5)

    plt.tight_layout()
    filename = f"pod_{filename}_mu1_{mu1:.3f}_mu2_{mu2:.4f}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved 2D cuts overlay to {filename}")

def plot_2d_heatmap_overlay(U_FOM, U_POD, X, Y, mu1, mu2, num_modes, filename, time_steps, time_step_size=0.05):
    """Create a side-by-side static heatmap overlay for specified time steps."""
    n_nodes = len(X)
    x_values = np.unique(X)
    y_values = np.unique(Y)
    X_grid, Y_grid = np.meshgrid(x_values, y_values)
    
    fig, axes = plt.subplots(2, len(time_steps), figsize=(15, 6))
    
    vmin = min(U_FOM[:n_nodes, :].min(), U_POD[:n_nodes, :].min())
    vmax = max(U_FOM[:n_nodes, :].max(), U_POD[:n_nodes, :].max())
    
    for i, t in enumerate(time_steps):
        index = int(t / time_step_size)
        U_FOM_frame = U_FOM[:n_nodes, index].reshape(len(y_values), len(x_values))
        U_POD_frame = U_POD[:n_nodes, index].reshape(len(y_values), len(x_values))
        
        axes[0, i].pcolormesh(X_grid, Y_grid, U_FOM_frame, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
        axes[1, i].pcolormesh(X_grid, Y_grid, U_POD_frame, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'FOM, t={t}')
        axes[1, i].set_title(f'POD {num_modes} Modes, t={t}')
        axes[0, i].invert_yaxis()
        axes[1, i].invert_yaxis()
    
    plt.tight_layout()
    filename = f"pod_{filename}_mu1_{mu1:.3f}_mu2_{mu2:.4f}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved heatmap overlay to {filename}")


# Example usage
if __name__ == "__main__":
    # Parameters
    mu1 = 4.560
    mu2 = 0.0190
    num_modes = 95
    fixed_value = 50.0
    time_steps = [0, 5, 10, 15, 17.5, 20, 25]
    overlay_time_steps = [5, 10, 15, 20]

    # Domain and mesh parameters
    a, b = 0, 100  # Domain range
    nx, ny = 250, 250  # Number of grid points in x and y directions
    x = np.linspace(a, b, nx + 1)
    y = np.linspace(a, b, ny + 1)
    X, Y = np.meshgrid(x, y)
    X, Y = X.flatten(), Y.flatten()

    # Load data
    U_FOM = np.load(os.path.join(f'U_FOM_mu1_4.560_mu2_0.0190_mesh_250x250.npy')).T
    U_POD = np.load(os.path.join(f"U_ROM_mu1_4.560_mu2_0.0190_num_modes_95_mesh_250x250.npy")).T

    # Generate visualizations
    # plot_3d_surface_animation(U_FOM, U_POD_RBF, X, Y, mu1, mu2, num_modes)
    # plot_2d_cuts_animation(U_FOM, U_POD_RBF, X, Y, 'x', fixed_value, mu1, mu2, num_modes, '2d_cuts_x_animation')
    # plot_2d_cuts_animation(U_FOM, U_POD_RBF, X, Y, 'y', fixed_value, mu1, mu2, num_modes, '2d_cuts_y_animation')
    plot_2d_cuts_overlay(U_FOM, U_POD, X, Y, 'x', fixed_value, mu1, mu2, num_modes, '2d_cuts_x_overlay', time_steps)
    plot_2d_cuts_overlay(U_FOM, U_POD, X, Y, 'y', fixed_value, mu1, mu2, num_modes, '2d_cuts_y_overlay', time_steps)
    # plot_2d_heatmap_animation(U_FOM, U_POD_RBF, X, Y, mu1, mu2)
    plot_2d_heatmap_overlay(U_FOM, U_POD, X, Y, mu1, mu2, num_modes, 'heatmap_overlay', overlay_time_steps)

