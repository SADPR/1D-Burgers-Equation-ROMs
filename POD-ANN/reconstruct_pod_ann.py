"""
torch_pod_ann_reconstruction.py
Reconstruct one snapshot file with a trained latent-space POD-ANN decoder
and create a comparison GIF.
"""

import time, os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# ────────────────────────────────────────────────────────────────
# ANN architecture (must match training)
# ────────────────────────────────────────────────────────────────
class POD_ANN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
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
        return self.fc6(x)


# ────────────────────────────────────────────────────────────────
# Helper: GIF of FOM vs reconstruction
# ────────────────────────────────────────────────────────────────
def create_combined_gif(X, U_fom, U_rec, n_steps, dt, n_ret, n_dis,
                        fname="pod_ann_reconstruction.gif"):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(U_fom.min(), U_fom.max())
    line_fom, = ax.plot(X, U_fom[:, 0], 'b-',  label='FOM')
    label_rec = fr"POD-ANN ($n={n_ret}$, $\bar n={n_dis}$)"
    line_rec, = ax.plot(X, U_rec[:, 0], 'g--', label=label_rec)
    ax.set_xlabel('x'); ax.set_ylabel('u'); ax.legend()

    def update(k):
        line_fom.set_ydata(U_fom[:, k])
        line_rec.set_ydata(U_rec[:, k])
        ax.set_title(f"t = {k*dt:.2f}")
        return line_fom, line_rec

    ani = FuncAnimation(fig, update, frames=n_steps+1, blit=True)
    ani.save(fname, writer=PillowWriter(fps=10))
    plt.close(fig)


# ────────────────────────────────────────────────────────────────
# Reconstruction routine
# ────────────────────────────────────────────────────────────────
def reconstruct_snapshot_with_pod_ann(snapshot_file, U_p, U_s, model):
    """Return (FOM_snapshots, reconstructed_snapshots)."""
    snapshots = np.load(snapshot_file)          # shape (N × N_t)
    U_combined = np.hstack((U_p, U_s))          # (N × n_tot)

    # retained coords q_p
    q = U_combined.T @ snapshots
    n_ret = U_p.shape[1]
    q_p   = q[:n_ret, :]                        # (n × N_t)

    # ANN prediction of discarded coords
    with torch.no_grad():
        q_p_tensor = torch.tensor(q_p.T, dtype=torch.float32)
        q_s_tensor = model(q_p_tensor)          # (N_t × n̄)
        q_s        = q_s_tensor.numpy().T       # (n̄ × N_t)

    snapshots_rec = U_p @ q_p + U_s @ q_s       # (N × N_t)
    return snapshots, snapshots_rec


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Load POD bases
    U_p = np.load("U_p.npy")        # (N × 17)
    U_s = np.load("U_s.npy")        # (N × 79)
    n_ret, n_dis = U_p.shape[1], U_s.shape[1]

    # Load trained model (entire object saved)
    model = torch.load("pod_ann_model.pth", map_location="cpu")
    model.eval()

    snapshot_file = "../FEM/fem_training_data/fem_simulation_mu1_4.250_mu2_0.0150.npy"

    t0 = time.time()
    U_fom, U_rec = reconstruct_snapshot_with_pod_ann(snapshot_file, U_p, U_s, model)
    print(f"reconstruction time  {time.time()-t0:.2f} s")

    rel_err = np.linalg.norm(U_fom - U_rec) / np.linalg.norm(U_fom)
    print(f"relative L2 error    {rel_err:.3e}")

    np.save("pod_ann_reconstruction.npy", U_rec)

    # Mesh and time grid (match thesis settings)
    a, b = 0.0, 100.0
    m    = 511                     # N = m+1
    X    = np.linspace(a, b, m+1)

    Tf   = 5.0
    dt   = 0.05
    n_ts = int(Tf / dt)

    create_combined_gif(X, U_fom, U_rec, n_ts, dt, n_ret, n_dis)





