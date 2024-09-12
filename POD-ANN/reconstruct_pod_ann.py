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
# def reconstruct_snapshot_with_pod_ann(snapshot_file, U, U_p, U_s, model, r, q_p_mean, q_p_std, q_s_mean, q_s_std):
def reconstruct_snapshot_with_pod_ann(snapshot_file, U, U_p, U_s, model, r):
    # Load the snapshot file
    snapshots = np.load(snapshot_file)

    # Project onto the POD basis
    q = U.T @ snapshots
    q_p = q[:r, :]

    # Normalize the q_p modes
    # q_p_normalized = (q_p - q_p_mean) / q_p_std

    # Initialize the model and set it to evaluation mode
    model.eval()

    # Reconstruct the snapshots
    reconstructed_snapshots_ann = []
    for i in range(q_p.shape[1]):
        q_p_sample = torch.tensor(q_p[:, i], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_s_pred = model(q_p_sample)
        q_s_pred = q_s_pred.numpy().T
        q_s_pred_denorm = q_s_pred #* q_s_std + q_s_mean
        reconstructed_snapshot_ann = U_p @ q_p[:, i] + U_s @ q_s_pred.reshape(-1)
        reconstructed_snapshots_ann.append(reconstructed_snapshot_ann)

    # Convert lists to arrays and return
    reconstructed_snapshots_ann = np.array(reconstructed_snapshots_ann).squeeze().T
    return reconstructed_snapshots_ann

if __name__ == '__main__':

    # Load the trained ANN model
    pod_ann_model = torch.load('pod_ann_model.pth')
    pod_ann_model.eval()

    # Load the mean and std values
    # data_path = '../FEM/training_data/'  # Replace with your data folder
    # files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
    # all_snapshots = []

    # for file in files:
    #     snapshots = np.load(file)
    #     all_snapshots.append(snapshots)

    # all_snapshots = np.hstack(all_snapshots)  # Ensure shape is (248000, 513)
    # mean = np.mean(all_snapshots)
    # std = np.std(all_snapshots)

    # Load a random snapshot from the training_data directory
    # snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'
    snapshot_file = '../FEM/testing_data/simulations/simulation_mu1_4.85_mu2_0.0222.npy'
    snapshot = np.load(snapshot_file)

    # Prepare training data for POD-ANN
    U_p = np.load('U_p.npy')
    U_s = np.load('U_s.npy')
    U = np.hstack((U_p, U_s))
    # q = U.T @ all_snapshots  # Project snapshots onto the POD basis
    # q_p = q[:28, :]  # Principal modes
    # q_s = q[28:301, :]  # Secondary modes

    # # Normalize the q_p and q_s modes for the neural network
    # q_p_mean = np.mean(q_p, axis=1, keepdims=True)
    # q_p_std = np.std(q_p, axis=1, keepdims=True)
    # q_p_normalized = (q_p - q_p_mean) / q_p_std

    # q_s_mean = np.mean(q_s, axis=1, keepdims=True)
    # q_s_std = np.std(q_s, axis=1, keepdims=True)
    # q_s_normalized = (q_s - q_s_mean) / q_s_std

    # Reconstruct the snapshot using POD-ANN
    # pod_ann_reconstructed = reconstruct_snapshot_with_pod_ann(
    #     snapshot_file, U, U_p, U_s, pod_ann_model, 28, q_p_mean, q_p_std, q_s_mean, q_s_std
    # )
    import time
    start = time.time()
    pod_ann_reconstructed = reconstruct_snapshot_with_pod_ann(
        snapshot_file, U, U_p, U_s, pod_ann_model, 28
    )
    end = time.time()
    print(f"Time {start-end}")
    print(f"Error: {np.linalg.norm(snapshot-pod_ann_reconstructed)/np.linalg.norm(snapshot)}")

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
    create_combined_gif(X, snapshot, pod_ann_reconstructed, nTimeSteps, At, 28)





