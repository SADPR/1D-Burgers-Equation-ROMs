"""
Quick reconstruction check + optional animation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from quad_utils import build_Q, get_sym

# ------------ load ---------------- #
# S   = np.load("../FEM/fem_training_data/fem_simulation_mu1_4.250_mu2_0.0150.npy")[:,0:]   # same shape (N,Ns)
S   = np.load("../FEM/fem_testing_data/fem_simulation_mu1_4.750_mu2_0.0200.npy")[:,0:]   # same shape (N,Ns)
Phi = np.load("Phi.npy")
H   = np.load("H.npy")

N, Ns = S.shape
n = Phi.shape[1]

q   = Phi.T @ S
Q   = build_Q(q)
S_r = Phi @ q + H @ Q

rel_err = np.linalg.norm(S - S_r) / np.linalg.norm(S)
print(f"training-set relative Frobenius error : {rel_err:.3e}")

# ------------ animation (optional) ------------- #
fig, ax = plt.subplots()
ax.set_ylim(np.min(S), np.max(S))
line1, = ax.plot(S[:,0], lw=1, label="exact")
line2, = ax.plot(S_r[:,0], lw=1, label="recon")
ax.legend()

def update(f):
    line1.set_ydata(S[:,f])
    line2.set_ydata(S_r[:,f])
    ax.set_title(f"snapshot {f}")
    return line1, line2

ani = FuncAnimation(fig, update, frames=Ns, interval=60, blit=True)
plt.show()
