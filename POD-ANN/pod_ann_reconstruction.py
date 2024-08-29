import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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

# Reconstruct using POD-ANN
def reconstruct_snapshot_with_pod_ann(snapshot_file, U_combined, U_p, U_s, model, r):
    # Load the snapshot file
    snapshots = np.load(snapshot_file)

    # Project onto the combined POD basis
    q = U_combined.T @ snapshots
    q_p = q[:r, :]
    q_s_original = q[r:, :]

    # Initialize the model and set it to evaluation mode
    model.eval()

    # Reconstruct the snapshots
    reconstructed_snapshots_ann = []
    reconstructed_snapshots_best = []
    for i in range(q_p.shape[1]):
        q_p_sample = torch.tensor(q_p[:, i], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_s_pred = model(q_p_sample)
        q_s_pred = q_s_pred.numpy().T
        q_s_pred_denorm = q_s_pred #* q_s_std + q_s_mean
        reconstructed_snapshot_ann = U_p @ q_p[:, i] + U_s @ q_s_pred_denorm.reshape(-1)
        reconstructed_snapshots_ann.append(reconstructed_snapshot_ann)

        # Best possible reconstruction
        reconstructed_snapshot_best = U_p @ q_p[:, i] + U_s @ q_s_original[:, i]
        reconstructed_snapshots_best.append(reconstructed_snapshot_best)

    # Convert lists to arrays and return
    reconstructed_snapshots_ann = np.array(reconstructed_snapshots_ann).squeeze().T
    reconstructed_snapshots_best = np.array(reconstructed_snapshots_best).squeeze().T
    return reconstructed_snapshots_ann, reconstructed_snapshots_best

def save_npy_files(reconstructed_snapshots_ann, reconstructed_snapshots_best):
    np.save("reconstructed_snapshots_ann.npy", reconstructed_snapshots_ann)
    np.save("reconstructed_snapshots_best.npy", reconstructed_snapshots_best)
    print("Reconstructed snapshots saved as .npy files.")

def plot_and_save_gif(X, reconstructed_snapshots_ann, reconstructed_snapshots_best, At):
    nTimeSteps = reconstructed_snapshots_ann.shape[1]
    
    fig, ax = plt.subplots()
    line_ann, = ax.plot(X, reconstructed_snapshots_ann[:, 0], label='Reconstructed Snapshot (ANN-PROM)')
    line_best, = ax.plot(X, reconstructed_snapshots_best[:, 0], label='Best Possible Reconstruction', linestyle='dashed')
    ax.set_xlim(0, 100)
    ax.set_ylim(np.min(reconstructed_snapshots_ann), np.max(reconstructed_snapshots_ann))
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

    def update(frame):
        line_ann.set_ydata(reconstructed_snapshots_ann[:, frame])
        line_best.set_ydata(reconstructed_snapshots_best[:, frame])
        ax.set_title(f't = {frame * At:.2f}')
        return line_ann, line_best

    ani = FuncAnimation(fig, update, frames=nTimeSteps, blit=True)

    # Save animation as GIF
    ani.save("reconstructed_snapshots_comparison.gif", writer=PillowWriter(fps=10))
    plt.show()
    print("GIF saved as 'reconstructed_snapshots_comparison.gif'.")

if __name__ == "__main__":
    # File to reconstruct
    snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'

    # Load the U_p, U_s, and U_combined matrices
    U_p = np.load('U_p.npy')
    U_s = np.load('U_s.npy')
    U_combined = np.hstack((U_p, U_s))

    # Load the ANN model and other necessary parameters
    input_dim = U_p.shape[1]
    latent_dim = U_s.shape[1]
    model = torch.load(f'pod_ann_model.pth')

    # Load normalization parameters (if used)
    # q_p_mean = np.load("q_p_mean.npy")
    # q_p_std = np.load("q_p_std.npy")
    # q_s_mean = np.load("q_s_mean.npy")
    # q_s_std = np.load("q_s_std.npy")

    # Reconstruct the snapshots
    reconstructed_snapshots_ann, reconstructed_snapshots_best = reconstruct_snapshot_with_pod_ann(
        snapshot_file, U_combined, U_p, U_s, model, input_dim
    )

    # Save the reconstructed snapshots as .npy files
    save_npy_files(reconstructed_snapshots_ann, reconstructed_snapshots_best)

    # Plot and save the reconstruction as a GIF
    X = np.linspace(0, 100, reconstructed_snapshots_ann.shape[0])
    At = 0.07
    plot_and_save_gif(X, reconstructed_snapshots_ann, reconstructed_snapshots_best, At)
