"""
torch_pod_dl_reconstruction.py
Reconstruct one snapshot file with a trained POD-DL autoencoder
and create a comparison GIF.
"""

import time, os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# ────────────────────────────────────────────────────────────────
# Autoencoder architecture (must match training)
# ────────────────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_layers=[128]):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ELU())
            prev_dim = h
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ELU())
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# ────────────────────────────────────────────────────────────────
# Helper: create GIF
# ────────────────────────────────────────────────────────────────
def create_combined_gif(X, U_fom, U_rec, n_steps, dt, n_ret,
                        fname="pod_dl_reconstruction.gif"):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(U_fom.min(), U_fom.max())
    line_fom, = ax.plot(X, U_fom[:, 0], 'b-',  label='FOM')
    line_rec, = ax.plot(X, U_rec[:, 0], 'g--', label=f'POD-DL ($n={n_ret}$)')
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
def reconstruct_snapshot_with_autoencoder(snapshot_file, V, q_mean, q_std, model):
    snapshots = np.load(snapshot_file)           # (N × N_t)
    q = V.T @ snapshots                          # (n × N_t)
    q_norm = (q - q_mean) / q_std                # normalize (elementwise)

    with torch.no_grad():
        q_tensor = torch.tensor(q_norm.T, dtype=torch.float32)  # (N_t × n)
        q_rec_tensor = model(q_tensor)                          # (N_t × n)
        q_rec_norm = q_rec_tensor.numpy().T                     # (n × N_t)

    q_rec = q_rec_norm * q_std + q_mean                         # denormalize
    snapshots_rec = V @ q_rec                                   # (N × N_t)
    return snapshots, snapshots_rec


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Load trained components
    V = np.load("V.npy")                        # (N × n)
    q_mean = np.load("q_mean.npy")              # (n × 1)
    q_std = np.load("q_std.npy")                # (n × 1)
    n_ret = V.shape[1]

    # Load trained model (state_dict version)
    latent_dim = 5
    model = Autoencoder(input_dim=n_ret, latent_dim=latent_dim)
    model.load_state_dict(torch.load("autoencoder_model.pth", map_location="cpu"))
    model.eval()

    # Snapshot file to reconstruct
    snapshot_file = "../FEM/fem_training_data/fem_simulation_mu1_4.250_mu2_0.0150.npy"

    # Run reconstruction
    t0 = time.time()
    U_fom, U_rec = reconstruct_snapshot_with_autoencoder(snapshot_file, V, q_mean, q_std, model)
    print(f"Reconstruction time: {time.time() - t0:.2f} s")

    rel_err = np.linalg.norm(U_fom - U_rec) / np.linalg.norm(U_fom)
    print(f"Relative L2 error  : {rel_err:.3e}")

    np.save("pod_dl_reconstruction.npy", U_rec)

    # Create GIF
    a, b = 0.0, 100.0
    m = V.shape[0] - 1
    X = np.linspace(a, b, m+1)
    Tf = 5.0
    dt = 0.05
    n_ts = int(Tf / dt)

    create_combined_gif(X, U_fom, U_rec, n_ts, dt, n_ret,
                        fname="pod_dl_reconstruction.gif")
