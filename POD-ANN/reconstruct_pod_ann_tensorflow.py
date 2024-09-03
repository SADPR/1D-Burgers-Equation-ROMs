import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import tensorflow as tf

# Define the ANN model in TensorFlow (if needed for defining new models)
class POD_ANN(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(POD_ANN, self).__init__()
        self.fc1 = tf.keras.layers.Dense(32, activation='elu')
        self.fc2 = tf.keras.layers.Dense(64, activation='elu')
        self.fc3 = tf.keras.layers.Dense(128, activation='elu')
        self.fc4 = tf.keras.layers.Dense(256, activation='elu')
        self.fc5 = tf.keras.layers.Dense(256, activation='elu')
        self.fc6 = tf.keras.layers.Dense(output_dim)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return self.fc6(x)

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
def reconstruct_snapshot_with_pod_ann(snapshot_file, U, U_p, U_s, model, r):
    # Load the snapshot file
    snapshots = np.load(snapshot_file)

    # Project onto the POD basis
    q = np.dot(U.T, snapshots)
    q_p = q[:r, :]

    # Reconstruct the snapshots
    reconstructed_snapshots_ann = []
    for i in range(q_p.shape[1]):
        q_p_sample = tf.convert_to_tensor(q_p[:, i], dtype=tf.float32)
        q_p_sample = tf.expand_dims(q_p_sample, axis=0)  # Add batch dimension

        q_s_pred = model(q_p_sample)
        q_s_pred = q_s_pred.numpy().T
        reconstructed_snapshot_ann = np.dot(U_p, q_p[:, i]) + np.dot(U_s, q_s_pred.reshape(-1))
        reconstructed_snapshots_ann.append(reconstructed_snapshot_ann)

    # Convert lists to arrays and return
    reconstructed_snapshots_ann = np.array(reconstructed_snapshots_ann).squeeze().T
    return reconstructed_snapshots_ann

if __name__ == '__main__':

    # Load the saved model (SavedModel format)
    pod_ann_model = tf.keras.models.load_model('pod_ann_model')

    # Ensure the model is not trainable (optional)
    pod_ann_model.trainable = False

    # Load the POD basis matrices
    r = 28
    U_p = np.load('U_p.npy')
    U_s = np.load('U_s.npy')
    U = np.hstack((U_p, U_s))

    # Load a snapshot from the training_data directory
    snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'
    snapshot = np.load(snapshot_file)

    # Reconstruct the snapshot using POD-ANN
    pod_ann_reconstructed = reconstruct_snapshot_with_pod_ann(
        snapshot_file, U, U_p, U_s, pod_ann_model, r
    )

    # Save the reconstructed snapshots
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
    create_combined_gif(X, snapshot, pod_ann_reconstructed, nTimeSteps, At, r)

