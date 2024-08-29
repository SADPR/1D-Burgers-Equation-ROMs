import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Define the DenseAutoencoder model (similar structure to PyTorch)
class DenseAutoencoder(tf.keras.Model):
    def __init__(self, input_dim, latent_dim):
        super(DenseAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(513, activation='elu'),
            tf.keras.layers.Dense(256, activation='elu'),
            tf.keras.layers.Dense(128, activation='elu'),
            tf.keras.layers.Dense(64, activation='elu'),
            tf.keras.layers.Dense(32, activation='elu'),
            tf.keras.layers.Dense(latent_dim),
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(32, activation='elu'),
            tf.keras.layers.Dense(64, activation='elu'),
            tf.keras.layers.Dense(128, activation='elu'),
            tf.keras.layers.Dense(256, activation='elu'),
            tf.keras.layers.Dense(513, activation='elu'),
            tf.keras.layers.Dense(input_dim),
        ])

    def call(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

# Function to create a GIF with all snapshots overlaid
def create_combined_gif(X, original_snapshot, autoencoder_reconstructed, nTimeSteps, At, latent_dim):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(0, 8)

    line_original, = ax.plot(X, original_snapshot[:, 0], 'b-', label='Original Snapshot')
    line_autoencoder, = ax.plot(X, autoencoder_reconstructed[:, 0], 'r--', label=f'Autoencoder Reconstructed (latent dim={latent_dim})')

    ax.set_title('Snapshot Comparison')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

    def update(frame):
        line_original.set_ydata(original_snapshot[:, frame])
        line_autoencoder.set_ydata(autoencoder_reconstructed[:, frame])
        ax.set_title(f'Snapshot Comparison at t = {frame * At:.2f}')
        return line_original, line_autoencoder

    ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, blit=True)

    # Save animation as GIF
    ani.save(f"ae_reconstruction_latent_{latent_dim}_tensorflow.gif", writer=PillowWriter(fps=10))

    plt.show()

if __name__ == '__main__':
    latent_dim = 28
    input_dim = 513

    # Load the trained autoencoder model
    autoencoder_model = tf.keras.models.load_model(f'dense_autoencoder_complete_latent_{latent_dim}_tensorflow.h5', custom_objects={'DenseAutoencoder': DenseAutoencoder})
    autoencoder_model.summary()

    # Load a specific snapshot for reconstruction
    snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'
    snapshot = np.load(snapshot_file)

    # Convert to TensorFlow tensor
    snapshot = tf.convert_to_tensor(snapshot.T, dtype=tf.float32)  # Transpose to match input dimension

    # Reconstruct the snapshot using the autoencoder
    reconstructed_snapshot = autoencoder_model(snapshot)

    # Convert back to numpy array for plotting
    snapshot = snapshot.numpy().T
    reconstructed_snapshot = reconstructed_snapshot.numpy().T

    # Save the reconstructed snapshots
    np.save(f'reconstructed_snapshots_latent_{latent_dim}_tensorflow.npy', reconstructed_snapshot)

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
    create_combined_gif(X, snapshot, reconstructed_snapshot, nTimeSteps, At, latent_dim)
