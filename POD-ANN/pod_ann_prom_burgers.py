import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import sys
from matplotlib.animation import FuncAnimation, PillowWriter

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../FEM/'))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Now you can import the module
from fem_burgers import FEMBurgers

# Define the POD-ANN model class (as used for training)
class POD_ANN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(POD_ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, output_dim)
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))
        x = self.elu(self.fc4(x))
        x = self.elu(self.fc5(x))
        x = self.fc6(x)
        return x

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
    Tf = 35.0
    At = 0.07
    nTimeSteps = int(Tf / At) + 1
    E = 0.01

    # Parameter mu2
    mu2 = 0.0182

    # Create an instance of the FEMBurgers class
    fem_burgers = FEMBurgers(X, T)

    # Load the trained POD-ANN model
    model = torch.load(f'pod_ann_model.pth')
    model.eval()

    # Load the U_p and U_s matrices
    U_p = np.load('U_p.npy')
    U_s = np.load('U_s.npy')

    # Solution using POD-ANN-based PROM
    print('POD-ANN PROM method...')
    import time
    start = time.time()
    
    U_POD_ANN_PROM = fem_burgers.pod_ann_prom(At, nTimeSteps, u0, uxa, E, mu2, U_p, U_s, model)

    end = time.time()
    print(f"Time taken: {end-start}")

    # Save the solution as a .npy file
    np.save("U_POD_ANN_PROM_solution.npy", U_POD_ANN_PROM)
    print("Solution saved to U_POD_ANN_PROM_solution.npy")

    # # Visualization and animation
    # fig, ax = plt.subplots()
    # line, = ax.plot(X, U_POD_ANN_PROM[:, 0], label='Solution over time')
    # ax.set_xlim(a, b)
    # ax.set_ylim(0, 6)
    # ax.set_xlabel('x')
    # ax.set_ylabel('u')
    # ax.legend()

    # def update(frame):
    #     line.set_ydata(U_POD_ANN_PROM[:, frame])
    #     ax.set_title(f't = {frame * At:.2f}')
    #     return line,

    # ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, blit=True)

    # # Save animation as GIF
    # ani.save("burgers_equation_prom.gif", writer=PillowWriter(fps=10))

    # plt.show()





