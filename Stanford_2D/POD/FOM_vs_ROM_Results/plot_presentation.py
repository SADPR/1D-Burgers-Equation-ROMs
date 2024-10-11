import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation

def plot_3d_surface_animation(U_FOM, U_ROM, X, Y, mu1, mu2, num_modes, filename='3d_surface_animation.gif', max_frames=500):
    """Create an animation of side-by-side 3D surface plots of FOM and ROM over time."""
    n_nodes = len(X)
    nTimeSteps = U_FOM.shape[1] - 1
    nTimeSteps = min(nTimeSteps, max_frames)
    
    x_values = np.unique(X)
    y_values = np.unique(Y)
    X_grid, Y_grid = np.meshgrid(x_values, y_values)
    
    U_FOM_frames = [U_FOM[:n_nodes, n].reshape(len(y_values), len(x_values)) for n in range(nTimeSteps + 1)]
    U_ROM_frames = [U_ROM[:n_nodes, n].reshape(len(y_values), len(x_values)) for n in range(nTimeSteps + 1)]
    
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    def update_surface(n):
        ax1.clear()
        ax2.clear()
        
        ax1.plot_surface(X_grid, Y_grid, U_FOM_frames[n], cmap='viridis', edgecolor='none')
        ax1.set_title(f'FOM at Time Step {n}')
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$y$')
        ax1.set_zlabel(r'$u_x(x, y)$')
        ax1.view_init(30, -60)
        
        ax2.plot_surface(X_grid, Y_grid, U_ROM_frames[n], cmap='viridis', edgecolor='none')
        ax2.set_title(f'POD-PROM 95 Modes at Time Step {n}')
        ax2.set_xlabel(r'$x$')
        ax2.set_ylabel(r'$y$')
        ax2.set_zlabel(r'$u_x(x, y)$')
        ax2.view_init(30, -60)
        
        return fig,

    ani = animation.FuncAnimation(fig, update_surface, frames=nTimeSteps + 1, blit=False)
    filename = f"3d_surface_animation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.gif"
    ani.save(filename, writer='pillow', fps=30)
    plt.close()
    print(f"Saved 3D surface animation to {filename}")

def plot_2d_cuts_animation(U_FOM, U_ROM, X, Y, line, fixed_value, mu1, mu2, num_modes, filename, max_frames=500):
    """Create an animation of 2D cuts over time for FOM and ROM in the same plot."""
    n_nodes = len(X)
    nTimeSteps = U_FOM.shape[1] - 1
    nTimeSteps = min(nTimeSteps, max_frames)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if line == 'x':
        indices = np.where(np.isclose(Y, fixed_value))[0]
        x_values = X[indices]
        u_fom_values_per_frame = [U_FOM[:n_nodes, n][indices] for n in range(nTimeSteps + 1)]
        u_rom_values_per_frame = [U_ROM[:n_nodes, n][indices] for n in range(nTimeSteps + 1)]
        
        def update_cut(n):
            ax.clear()
            u_fom_values = u_fom_values_per_frame[n]
            u_rom_values = u_rom_values_per_frame[n]
            ax.plot(x_values, u_fom_values, 'k-', linewidth=2.5, label='FOM')
            ax.plot(x_values, u_rom_values, 'g-', linewidth=2, label='POD-PROM 95 Modes')
            ax.set_ylim(0.5, 5.5)
            ax.set_title(f'2D Cut along y = {fixed_value}, Time Step {n}')
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$u_x(x, y)$')
            ax.legend()
            ax.grid()
            return ax,

    elif line == 'y':
        indices = np.where(np.isclose(X, fixed_value))[0]
        y_values = Y[indices]
        u_fom_values_per_frame = [U_FOM[:n_nodes, n][indices] for n in range(nTimeSteps + 1)]
        u_rom_values_per_frame = [U_ROM[:n_nodes, n][indices] for n in range(nTimeSteps + 1)]
        
        def update_cut(n):
            ax.clear()
            u_fom_values = u_fom_values_per_frame[n]
            u_rom_values = u_rom_values_per_frame[n]
            ax.plot(y_values, u_fom_values, 'k-', linewidth=2.5, label='FOM')
            ax.plot(y_values, u_rom_values, 'g-', linewidth=2, label='POD-PROM 95 Modes')
            ax.set_ylim(0.5, 5.5)
            ax.set_title(f'2D Cut along x = {fixed_value}, Time Step {n}')
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

def plot_2d_cuts_overlay(U_FOM, U_ROM, X, Y, line, fixed_value, mu1, mu2, num_modes, filename, time_steps, time_step_size=0.05):
    """Create static overlays of 2D cuts for FOM and ROM at specified time steps."""
    n_nodes = len(X)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if line == 'x':
        indices = np.where(np.isclose(Y, fixed_value))[0]
        x_values = X[indices]
        
        for t in time_steps:
            index = int(t / time_step_size)
            u_fom_values = U_FOM[:n_nodes, index][indices]
            u_rom_values = U_ROM[:n_nodes, index][indices]
            ax.plot(x_values, u_fom_values, 'k-', linewidth=2.5, label='FOM' if t == time_steps[0] else "")
            ax.plot(x_values, u_rom_values, '-', color='darkgreen', linewidth=2, label='POD-PROM 95 Modes' if t == time_steps[0] else "")
            
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
            u_rom_values = U_ROM[:n_nodes, index][indices]
            ax.plot(y_values, u_fom_values, 'k-', linewidth=2.5, label='FOM' if t == time_steps[0] else "")
            ax.plot(y_values, u_rom_values, '-', color='darkgreen', linewidth=2, label='POD-PROM 95 Modes' if t == time_steps[0] else "")
        
        ax.set_title(f'2D Cut Overlay along x={fixed_value}')
        ax.set_xlabel(r'$y$')
        ax.set_ylabel(r'$u_x(x, y)$')
        ax.legend()
        ax.grid()
        ax.set_ylim(0.5, 5.5)
    
    filename = f"{filename}_mu1_{mu1:.3f}_mu2_{mu2:.4f}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved 2D cuts overlay to {filename}")

def plot_2d_heatmap_animation(U_FOM, U_ROM, X, Y, mu1, mu2, filename='heatmap_animation.gif', max_frames=500):
    """Create an animation of side-by-side heatmaps of FOM and ROM over time."""
    n_nodes = len(X)
    nTimeSteps = U_FOM.shape[1] - 1
    nTimeSteps = min(nTimeSteps, max_frames)
    
    x_values = np.unique(X)
    y_values = np.unique(Y)
    X_grid, Y_grid = np.meshgrid(x_values, y_values)
    
    U_FOM_frames = [U_FOM[:n_nodes, n].reshape(len(y_values), len(x_values)) for n in range(nTimeSteps + 1)]
    U_ROM_frames = [U_ROM[:n_nodes, n].reshape(len(y_values), len(x_values)) for n in range(nTimeSteps + 1)]
    
    vmin = min(U_FOM[:n_nodes, :].min(), U_ROM[:n_nodes, :].min())
    vmax = max(U_FOM[:n_nodes, :].max(), U_ROM[:n_nodes, :].max())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    def update_heatmap(n):
        ax1.clear()
        ax2.clear()
        
        im1 = ax1.pcolormesh(X_grid, Y_grid, U_FOM_frames[n], cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
        im2 = ax2.pcolormesh(X_grid, Y_grid, U_ROM_frames[n], cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
        ax1.set_title(f'FOM at Time Step {n}')
        ax2.set_title(f'POD-PROM 95 Modes at Time Step {n}')
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        return [im1, im2]

    ani = animation.FuncAnimation(fig, update_heatmap, frames=nTimeSteps + 1, blit=False)
    filename = f"heatmap_animation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.gif"
    ani.save(filename, writer='pillow', fps=30)
    plt.close()
    print(f"Saved heatmap animation to {filename}")

def plot_2d_heatmap_overlay(U_FOM, U_ROM, X, Y, mu1, mu2, num_modes, filename, time_steps, time_step_size=0.05):
    """Create a side-by-side static heatmap overlay for specified time steps."""
    n_nodes = len(X)
    x_values = np.unique(X)
    y_values = np.unique(Y)
    X_grid, Y_grid = np.meshgrid(x_values, y_values)
    
    fig, axes = plt.subplots(2, len(time_steps), figsize=(15, 6))
    
    vmin = min(U_FOM[:n_nodes, :].min(), U_ROM[:n_nodes, :].min())
    vmax = max(U_FOM[:n_nodes, :].max(), U_ROM[:n_nodes, :].max())
    
    for i, t in enumerate(time_steps):
        index = int(t / time_step_size)
        U_FOM_frame = U_FOM[:n_nodes, index].reshape(len(y_values), len(x_values))
        U_ROM_frame = U_ROM[:n_nodes, index].reshape(len(y_values), len(x_values))
        
        axes[0, i].pcolormesh(X_grid, Y_grid, U_FOM_frame, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
        axes[1, i].pcolormesh(X_grid, Y_grid, U_ROM_frame, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'FOM, t={t}')
        axes[1, i].set_title(f'POD-PROM 95 Modes, t={t}')
        axes[0, i].invert_yaxis()
        axes[1, i].invert_yaxis()
    
    plt.tight_layout()
    filename = f"{filename}_mu1_{mu1:.3f}_mu2_{mu2:.4f}.png"
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
    time_steps = [0,5,10,15,17.5,20]
    overlay_time_steps = [5, 10, 15, 20]

    # Domain and mesh parameters
    a, b = 0, 100  # Domain range
    nx, ny = 250, 250  # Number of grid points in x and y directions

    # Create grid points in x and y directions
    x = np.linspace(a, b, nx + 1)
    y = np.linspace(a, b, ny + 1)
    X_grid, Y_grid = np.meshgrid(x, y)
    X, Y = X_grid.flatten(), Y_grid.flatten()  # Flatten the meshgrid into 1D arrays

    # Load data
    results_dir = "."
    fom_file = f"U_FOM_mu1_{mu1:.3f}_mu2_{mu2:.4f}_mesh_250x250.npy"
    rom_file = f"U_ROM_mu1_{mu1:.3f}_mu2_{mu2:.4f}_num_modes_{num_modes}_mesh_250x250.npy"
    U_FOM = np.load(os.path.join(results_dir, fom_file)).T
    U_ROM = np.load(os.path.join(results_dir, rom_file)).T

    # Generate visualizations
    plot_3d_surface_animation(U_FOM, U_ROM, X, Y, mu1, mu2, num_modes)
    plot_2d_cuts_animation(U_FOM, U_ROM, X, Y, 'x', fixed_value, mu1, mu2, num_modes, '2d_cuts_x_animation')
    plot_2d_cuts_animation(U_FOM, U_ROM, X, Y, 'y', fixed_value, mu1, mu2, num_modes, '2d_cuts_y_animation')
    plot_2d_cuts_overlay(U_FOM, U_ROM, X, Y, 'x', fixed_value, mu1, mu2, num_modes, '2d_cuts_x_overlay', time_steps)
    plot_2d_cuts_overlay(U_FOM, U_ROM, X, Y, 'y', fixed_value, mu1, mu2, num_modes, '2d_cuts_y_overlay', time_steps)
    plot_2d_heatmap_animation(U_FOM, U_ROM, X, Y, mu1, mu2)
    plot_2d_heatmap_overlay(U_FOM, U_ROM, X, Y, mu1, mu2, num_modes, 'heatmap_overlay', overlay_time_steps)
