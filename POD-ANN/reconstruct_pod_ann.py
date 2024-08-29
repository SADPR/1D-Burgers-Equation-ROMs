import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import random
import torch
import torch.nn as nn

# Define the ANN model
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

# Function to create gif with all snapshots overlaid
def create_combined_gif(X, original_snapshot, ann_reconstructed, nTimeSteps, At, latent_dim):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(0, 8)
    
    line_original, = ax.plot(X, original_snapshot[:, 0], 'b-', label='Original Snapshot')
    line_ann, = ax.plot(X, ann_reconstructed[:, 0], 'g--', label=f'POD-ANN Reconstructed (inf modes={latent_dim}, sup modes={301})')
    
    ax.set_title('Snapshot Comparison')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

    def update(frame):
        line_original.set_ydata(original_snapshot[:, frame])
        line_ann.set_ydata(ann_reconstructed[:, frame])
        ax.set_title(f'Snapshot Comparison at t = {frame * At:.2f}')
        return line_original, line_ann

    ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, blit=True)

    # Save animation as GIF
    ani.save("pod_ann_reconstruction.gif", writer=PillowWriter(fps=10))

    plt.show()

# Function to reconstruct snapshot using POD-ANN
def reconstruct_snapshot_with_pod_ann(snapshot_file, U, U_i, U_s, model, r, q_i_mean, q_i_std, q_s_mean, q_s_std):
    # Load the snapshot file
    snapshots = np.load(snapshot_file)

    # Project onto the POD basis
    q = U.T @ snapshots
    q_i = q[:r, :]

    # Normalize the q_i modes
    q_i_normalized = (q_i - q_i_mean) / q_i_std

    # Initialize the model and set it to evaluation mode
    model.eval()

    # Reconstruct the snapshots
    reconstructed_snapshots_ann = []
    for i in range(q_i_normalized.shape[1]):
        q_i_sample = torch.tensor(q_i_normalized[:, i], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_s_pred = model(q_i_sample)
        q_s_pred = q_s_pred.numpy().T
        q_s_pred_denorm = q_s_pred * q_s_std + q_s_mean
        reconstructed_snapshot_ann = U_i @ q_i[:, i] + U_s @ q_s_pred_denorm.reshape(-1)
        reconstructed_snapshots_ann.append(reconstructed_snapshot_ann)

    # Convert lists to arrays and return
    reconstructed_snapshots_ann = np.array(reconstructed_snapshots_ann).squeeze().T
    return reconstructed_snapshots_ann

if __name__ == '__main__':

    # Load the trained ANN model
    pod_ann_model = torch.load('pod_ann_complete.pth')
    pod_ann_model.eval()

    # Load the mean and std values
    data_path = '../training_data/'  # Replace with your data folder
    files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
    all_snapshots = []

    for file in files:
        snapshots = np.load(file)
        all_snapshots.append(snapshots)

    all_snapshots = np.hstack(all_snapshots)  # Ensure shape is (248000, 513)
    mean = np.mean(all_snapshots)
    std = np.std(all_snapshots)

    # Load a random snapshot from the training_data directory
    snapshot_file = '../training_data/simulation_mu1_4.76_mu2_0.0182.npy'
    snapshot = np.load(snapshot_file)

    # Prepare training data for POD-ANN
    U_i = np.load('U_i.npy')
    U_s = np.load('U_s.npy')
    U = np.hstack((U_i, U_s))
    q = U.T @ all_snapshots  # Project snapshots onto the POD basis
    q_i = q[:16, :]  # Inferior modes
    q_s = q[16:301, :]  # Superior modes

    # Normalize the q_i and q_s modes for the neural network
    q_i_mean = np.mean(q_i, axis=1, keepdims=True)
    q_i_std = np.std(q_i, axis=1, keepdims=True)
    q_i_normalized = (q_i - q_i_mean) / q_i_std

    q_s_mean = np.mean(q_s, axis=1, keepdims=True)
    q_s_std = np.std(q_s, axis=1, keepdims=True)
    q_s_normalized = (q_s - q_s_mean) / q_s_std

    # Reconstruct the snapshot using POD-ANN
    pod_ann_reconstructed = reconstruct_snapshot_with_pod_ann(
        snapshot_file, U, U_i, U_s, pod_ann_model, 16, q_i_mean, q_i_std, q_s_mean, q_s_std
    )

    np.save("pod_ann_reconstruction.npy", pod_ann_reconstructed)

    # Domain
    a = 0
    b = 100
    m = int(256 * 2)
    X = np.linspace(a, b, m + 1)

    # Time discretization and numerical diffusion
    Tf = 35
    At = 0.07
    nTimeSteps = int(Tf / At)

    # Create the combined GIF
    create_combined_gif(X, snapshot, pod_ann_reconstructed, nTimeSteps, At, 16)





