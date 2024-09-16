import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
from matplotlib.animation import FuncAnimation, PillowWriter

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../FEM/'))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Now you can import the module
from fem_burgers import FEMBurgers

if __name__ == "__main__":
    # Domain
    a = 0
    b = 100

    # Mesh
    m = int(256 * 2)
    h = (b - a) / m
    X = np.linspace(a, b, m + 1)
    T = np.array([np.arange(1, m + 1), np.arange(2, m + 2)]).T

    # Initial condition
    u0 = np.ones_like(X)

    umax = np.max(u0) + 0.1
    umin = np.min(u0) - 0.1

    # Boundary conditions
    uxa = 4.76  # u(0,t) = 4.76

    # Time discretization and numerical diffusion
    Tf = 5.0
    At = 0.07
    nTimeSteps = int(Tf / At) + 1
    E = 0.01

    # Parameter mu2
    mu2 = 0.0182

    # Create an instance of the FEMBurgers class
    fem_burgers = FEMBurgers(X, T)

    # Load the U_p and U_s matrices
    U_p = np.load('U_p.npy')
    U_s = np.load('U_s.npy')

    # Load the RBF model data (q_p_train, q_s_train, and KDTree)
    with open('training_data.pkl', 'rb') as f:
        data = pickle.load(f)
        kdtree = data['KDTree']  # Precomputed KDTree for nearest neighbors
        q_p_train = data['q_p']  # Principal modes training data
        q_s_train = data['q_s']  # Secondary modes training data

    # Solution using POD-RBF-based PROM with nearest neighbors dynamic interpolation
    print('POD-RBF PROM method...')
    import time
    start = time.time()
    U_POD_RBF_PROM = fem_burgers.pod_rbf_prom_nearest_neighbours_dynamic(
        At, nTimeSteps, u0, uxa, E, mu2, U_p, U_s, q_p_train, q_s_train, kdtree, epsilon=1.0, neighbors=20
    )
    end = time.time()
    print(f"Time taken: {end-start}")

    # Save the solution as a .npy file
    np.save("U_POD_RBF_PROM_solution.npy", U_POD_RBF_PROM)
    print("Solution saved to U_POD_RBF_PROM_solution.npy")

    # Visualization and animation
    fig, ax = plt.subplots()
    line, = ax.plot(X, U_POD_RBF_PROM[:, 0], label='Solution over time')
    ax.set_xlim(a, b)
    ax.set_ylim(0, 6)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

    def update(frame):
        line.set_ydata(U_POD_RBF_PROM[:, frame])
        ax.set_title(f't = {frame * At:.2f}')
        return line,

    ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, blit=True)

    # Save animation as GIF
    # ani.save("burgers_equation_prom_rbf.gif", writer=PillowWriter(fps=10))

    plt.show()







