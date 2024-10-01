import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

def plot_3d_surface_animation(U, X, Y, filename='surface_animation.gif'):
    """
    Create a GIF animation of the 3D surface plot over time.

    Parameters:
    U : np.ndarray
        Solution matrix with shape (n_nodes, nTimeSteps + 1, 2).
    X, Y : np.ndarray
        The x and y coordinates of the nodes.
    filename : str
        The filename for the saved GIF animation.
    """
    n_nodes, nTimeSteps_plus1, _ = U.shape
    nTimeSteps = nTimeSteps_plus1 - 1

    # Create a grid of points
    x_values = np.unique(X)
    y_values = np.unique(Y)
    X_grid, Y_grid = np.meshgrid(x_values, y_values)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update_surface(n):
        ax.clear()
        u_x_values = U[:, n, 0].reshape(len(y_values), len(x_values))
        surf = ax.plot_surface(X_grid, Y_grid, u_x_values, cmap='viridis', edgecolor='none')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$u_x(x, y)$')
        ax.set_title(f'3D Surface Plot at Time Step {n}')
        ax.set_zlim([U[:, :, 0].min(), U[:, :, 0].max()])
        return surf,

    ani = animation.FuncAnimation(fig, update_surface, frames=nTimeSteps_plus1, blit=False)
    ani.save(filename, writer='pillow', fps=2)
    plt.close()

def plot_2d_cuts_animation(U, X, Y, line='x', fixed_value=50.0, filename='cuts_animation.gif'):
    """
    Create a GIF animation of the 2D cuts over time.

    Parameters:
    U : np.ndarray
        Solution matrix with shape (n_nodes, nTimeSteps + 1, 2).
    X, Y : np.ndarray
        The x and y coordinates of the nodes.
    line : str
        The line along which to cut ('x' or 'y').
    fixed_value : float
        The value of y (for line='x') or x (for line='y') to plot the cut.
    filename : str
        The filename for the saved GIF animation.
    """
    n_nodes, nTimeSteps_plus1, _ = U.shape
    nTimeSteps = nTimeSteps_plus1 - 1

    fig, ax = plt.subplots()

    if line == 'x':
        indices = np.where(np.isclose(Y, fixed_value))[0]
        x_values = X[indices]

        def update_cut(n):
            ax.clear()
            u_x_values = U[indices, n, 0]  # u_x component at time n
            ax.plot(x_values, u_x_values, label=f'Time Step {n}')
            ax.set_xlabel('$x$')
            ax.set_ylabel(f'$u_x(x, y={fixed_value})$')
            ax.set_title(f'2D Cut along y = {fixed_value} at Time Step {n}')
            ax.legend()
            ax.grid()
            ax.set_ylim(-1,6)
            return ax.lines

    elif line == 'y':
        indices = np.where(np.isclose(X, fixed_value))[0]
        y_values = Y[indices]

        def update_cut(n):
            ax.clear()
            u_x_values = U[indices, n, 0]  # u_x component at time n
            ax.plot(y_values, u_x_values, label=f'Time Step {n}')
            ax.set_xlabel('$y$')
            ax.set_ylabel(f'$u_x(x={fixed_value}, y)$')
            ax.set_title(f'2D Cut along x = {fixed_value} at Time Step {n}')
            ax.legend()
            ax.grid()
            ax.set_ylim(-1, 6)
            return ax.lines

    ani = animation.FuncAnimation(fig, update_cut, frames=nTimeSteps_plus1, blit=False)
    ani.save(filename, writer='pillow', fps=2)
    plt.close()

if __name__ == "__main__":
    # Load the saved solution data
    U = np.load('U_FOM.npy')  # Shape: (n_nodes, nTimeSteps + 1, 2)
    # Assuming X and Y are saved or can be reconstructed
    # For this example, let's assume you have saved X and Y as well
    X = np.load('X.npy')
    Y = np.load('Y.npy')

    # Generate animations
    plot_3d_surface_animation(U, X, Y, filename='surface_animation.gif')
    plot_2d_cuts_animation(U, X, Y, line='x', fixed_value=50.0, filename='cuts_animation_x.gif')
    plot_2d_cuts_animation(U, X, Y, line='y', fixed_value=50.0, filename='cuts_animation_y.gif')
