import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from fem_burgers import FEMBurgers
import os

# Definir parámetros de prueba
test_samples = np.array([
    # [4.560, 0.0190],
    # [4.750, 0.0200],
    # [5.190, 0.0260]
    [6.20, 0.0400]
])

# Crear carpetas
os.makedirs("fem_testing_data", exist_ok=True)
os.makedirs("fem_testing_gifs", exist_ok=True)

# Guardar combinaciones de parámetros
np.save("fem_testing_data/parameter_combinations_test.npy", test_samples)

# Visualizar la grilla de prueba
plt.scatter(test_samples[:, 0], test_samples[:, 1], color='red')
plt.xlabel('$\mu_1$')
plt.ylabel('$\mu_2$')
plt.title('Testing Grid Sampling ($\mu_1$, $\mu_2$)')
plt.savefig("fem_testing_data/testing_grid_plot.pdf")
plt.close()

# Ejecutar simulaciones
for i, (mu1, mu2) in enumerate(test_samples):
    print(f"Testing sample {i+1}/{len(test_samples)}: mu1={mu1:.3f}, mu2={mu2:.4f}")

    # Dominio y malla
    a, b = 0, 100
    m = 511
    X = np.linspace(a, b, m + 1)
    T = np.array([np.arange(1, m + 1), np.arange(2, m + 2)]).T

    # Condición inicial
    u0 = np.ones_like(X)

    # Discretización temporal
    Tf = 25
    At = 0.05
    nTimeSteps = int(Tf / At)
    E = 0.00

    # Instanciar y simular
    fem_burgers = FEMBurgers(X, T)
    U_FOM = fem_burgers.fom_burgers(At, nTimeSteps, u0, mu1, E, mu2)

    # Guardar resultados
    file_name = f"fem_testing_data/fem_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy"
    np.save(file_name, U_FOM)

    # Guardar animación GIF
    fig, ax = plt.subplots()
    line, = ax.plot(X, U_FOM[:, 0], label='Solution over time')
    ax.set_xlim(a, b)
    ax.set_ylim(0, 8)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

    def update(frame):
        line.set_ydata(U_FOM[:, frame])
        ax.set_title(f't = {frame * At:.2f}')
        return line,

    ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, blit=True)
    gif_file = f"fem_testing_gifs/fem_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.gif"
    ani.save(gif_file, writer=PillowWriter(fps=10))
    plt.close(fig)
