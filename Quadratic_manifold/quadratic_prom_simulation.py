import numpy as np
import os, sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# ------------------------------------------------------------------
# 1.  Path handling
# ------------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../FEM"))
sys.path.append(parent_dir)

from fem_burgers import FEMBurgers          # after path injection

# ------------------------------------------------------------------
# 2.  Mesh and PDE parameters  (unchanged)
# ------------------------------------------------------------------
a, b = 0, 100
m = 511
X = np.linspace(a, b, m + 1)
T = np.array([np.arange(1, m + 1), np.arange(2, m + 2)]).T

u0  = np.ones_like(X)
uxa = 5.190

Tf, At = 25.0, 0.05
nTimeSteps = int(Tf / At)
E   = 0.0
mu2 = 0.0260

# ------------------------------------------------------------------
# 3.  Load Φ and H produced by build_manifold.py
# ------------------------------------------------------------------
Phi = np.load("Phi.npy")            # (N,n)
H   = np.load("H.npy")              # (N,k)

n_modes = Phi.shape[1]              # n     (for log / filename only)

# optional consistency check
assert H.shape[1] == n_modes*(n_modes+1)//2, "Φ and H dimensions mismatch"

# ------------------------------------------------------------------
# 4.  Instantiate FEM model
# ------------------------------------------------------------------
fem = FEMBurgers(X, T)

# ------------------------------------------------------------------
# 5.  Run quadratic PROM
# ------------------------------------------------------------------
print(f"Running quadratic PROM (LSPG) with n = {n_modes} modes")
U_PROM = fem.pod_quadratic_manifold(
    At, nTimeSteps,
    u0, uxa,
    E, mu2,
    Phi, H,
    projection="LSPG"
)

# ------------------------------------------------------------------
# 6.  Save & visualise
# ------------------------------------------------------------------
os.makedirs("quadratic_rom_solutions", exist_ok=True)
np.save(f"quadratic_rom_solutions/quadratic_PROM_U_PROM_{n_modes}_modes_mu1_{uxa:.3f}_mu2_{mu2:.4f}.npy", U_PROM)
print("Simulation complete – results saved.")

U = U_PROM
fig, ax = plt.subplots()
line, = ax.plot(X, U[:, 0], label='Solution over time')
ax.set_xlim(a, b)
ax.set_ylim(0, 6)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.legend()

def update(frame):
    line.set_ydata(U[:, frame])
    ax.set_title(f't = {frame * At:.2f}')
    return line,

ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, blit=True)

plt.show()
