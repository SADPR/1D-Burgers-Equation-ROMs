import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

# Enable LaTeX text rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def plot_3d_surface_animation(U, X, Y, filename='surface_animation_fine_mesh.gif', max_frames=500):
    n_nodes, nTimeSteps_plus1, _ = U.shape
    nTimeSteps = min(nTimeSteps_plus1 - 1, max_frames)  # Limit to max_frames

    x_values = np.unique(X)
    y_values = np.unique(Y)
    X_grid, Y_grid = np.meshgrid(x_values, y_values)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    z_min = U[:, :, 0].min()
    z_max = U[:, :, 0].max()

    U_reshaped = [U[:, n, 0].reshape(len(y_values), len(x_values)) for n in range(nTimeSteps + 1)]

    def update_surface(n):
        ax.clear()
        u_x_values = U_reshaped[n]
        surf = ax.plot_surface(X_grid, Y_grid, u_x_values, cmap='viridis', edgecolor='none')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$u_x(x, y)$')
        ax.set_title(f'3D Surface Plot at Time Step {n}', fontsize=12)
        ax.set_zlim([z_min, z_max])
        ax.view_init(30, -60)
        ax.grid(True)
        return surf,

    ani = animation.FuncAnimation(fig, update_surface, frames=nTimeSteps + 1, blit=True)
    ani.save(filename, writer='pillow', fps=30)
    plt.close()

def plot_3d_side_view_animation(U, X, Y, filename='side_view_animation_fine_mesh.gif', max_frames=500):
    n_nodes, nTimeSteps_plus1, _ = U.shape
    nTimeSteps = min(nTimeSteps_plus1 - 1, max_frames)

    x_values = np.unique(X)
    y_values = np.unique(Y)
    X_grid, Y_grid = np.meshgrid(x_values, y_values)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    z_min = U[:, :, 0].min()
    z_max = U[:, :, 0].max()

    U_reshaped = [U[:, n, 0].reshape(len(y_values), len(x_values)) for n in range(nTimeSteps + 1)]

    def update_side_view(n):
        ax.clear()
        u_x_values = U_reshaped[n]
        surf = ax.plot_surface(X_grid, Y_grid, u_x_values, cmap='viridis', edgecolor='none')
        ax.set_xlabel(r'$x$')
        ax.set_zlabel(r'$u_x(x, y)$')
        ax.set_title(f'3D Side View (x vs. $u_x$) at Time Step {n}', fontsize=12)
        ax.set_zlim([z_min, z_max])
        ax.view_init(elev=0, azim=-90)
        ax.grid(True)
        return surf,

    ani = animation.FuncAnimation(fig, update_side_view, frames=nTimeSteps + 1, blit=True)
    ani.save(filename, writer='pillow', fps=30)
    plt.close()

def plot_2d_cuts_animation(U, X, Y, line='x', fixed_value=50.0, filename='cuts_animation_fine_mesh.gif', max_frames=500):
    n_nodes, nTimeSteps_plus1, _ = U.shape
    nTimeSteps = min(nTimeSteps_plus1 - 1, max_frames)

    fig, ax = plt.subplots()

    if line == 'x':
        indices = np.where(np.isclose(Y, fixed_value))[0]
        x_values = X[indices]
        u_values_per_frame = [U[indices, n, 0] for n in range(nTimeSteps + 1)]

        def update_cut(n):
            ax.clear()
            u_x_values = u_values_per_frame[n]
            ax.plot(x_values, u_x_values, label=f'Time Step {n}')
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(rf'$u_x(x, y={fixed_value})$')
            ax.set_title(f'2D Cut along y = {fixed_value} at Time Step {n}', fontsize=12)
            ax.legend()
            ax.grid(True)
            ax.set_ylim(-1, 6)
            return ax.lines

    elif line == 'y':
        indices = np.where(np.isclose(X, fixed_value))[0]
        y_values = Y[indices]
        u_values_per_frame = [U[indices, n, 0] for n in range(nTimeSteps + 1)]

        def update_cut(n):
            ax.clear()
            u_x_values = u_values_per_frame[n]
            ax.plot(y_values, u_x_values, label=f'Time Step {n}')
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(rf'$u_x(x={fixed_value}, y)$')
            ax.set_title(f'2D Cut along x = {fixed_value} at Time Step {n}', fontsize=12)
            ax.legend()
            ax.grid(True)
            ax.set_ylim(-1, 6)
            return ax.lines

    ani = animation.FuncAnimation(fig, update_cut, frames=nTimeSteps + 1, blit=True)
    ani.save(filename, writer='pillow', fps=30)
    plt.close()

def plot_2d_heatmap_animation(U, X, Y, filename='heatmap_animation_fine_mesh.gif', max_frames=500):
    n_nodes, nTimeSteps_plus1, _ = U.shape
    nTimeSteps = min(nTimeSteps_plus1 - 1, max_frames)

    x_values = np.unique(X)
    y_values = np.unique(Y)
    X_grid, Y_grid = np.meshgrid(x_values, y_values)

    z_min = U[:, :, 0].min()
    z_max = U[:, :, 0].max()

    U_reshaped = [U[:, n, 0].reshape(len(y_values), len(x_values)) for n in range(nTimeSteps + 1)]

    fig, ax = plt.subplots()

    def update_heatmap(n):
        ax.clear()
        u_x_values = U_reshaped[n]
        heatmap = ax.pcolormesh(X_grid, Y_grid, u_x_values, cmap='viridis', shading='auto')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_title(f'Heatmap of $u_x(x, y)$ at Time Step {n}', fontsize=12)
        ax.set_xlim([X.min(), X.max()])
        ax.set_ylim([Y.min(), Y.max()])
        ax.invert_yaxis()  # This inverts the y-axis
        ax.set_aspect('equal')
        ax.grid(True)
        return heatmap,

    ani = animation.FuncAnimation(fig, update_heatmap, frames=nTimeSteps + 1, blit=True)
    ani.save(filename, writer='pillow', fps=30)
    plt.close()

def plot_100th_step(U, X, Y):
    n_nodes, nTimeSteps_plus1, _ = U.shape
    time_step = min(376, nTimeSteps_plus1 - 1)

    x_values = np.unique(X)
    y_values = np.unique(Y)
    X_grid, Y_grid = np.meshgrid(x_values, y_values)

    u_x_values = U[:, time_step, 0].reshape(len(y_values), len(x_values))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_grid, Y_grid, u_x_values, cmap='viridis', edgecolor='none')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$u_x(x, y)$')
    ax.set_title(f'3D Surface Plot at Time Step {time_step}', fontsize=12)
    ax.view_init(30, -60)
    plt.show()

if __name__ == "__main__":
    U = np.load('U_FOM_fine_mesh.npy')  # Shape: (n_nodes, nTimeSteps + 1, 2)
    X = np.load('X_fine_mesh.npy')
    Y = np.load('Y_fine_mesh.npy')

    plot_3d_surface_animation(U, X, Y, filename='surface_animation_fine_mesh.gif', max_frames=500)
    # plot_3d_side_view_animation(U, X, Y, filename='side_view_animation_fine_mesh.gif', max_frames=500)
    plot_2d_cuts_animation(U, X, Y, line='x', fixed_value=50.0, filename='cuts_animation_x_fine_mesh.gif', max_frames=500)
    plot_2d_cuts_animation(U, X, Y, line='y', fixed_value=50.0, filename='cuts_animation_y_fine_mesh.gif', max_frames=500)
    plot_2d_heatmap_animation(U, X, Y, filename='heatmap_animation_fine_mesh.gif', max_frames=500)

    # plot_100th_step(U, X, Y)
