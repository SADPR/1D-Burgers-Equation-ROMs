import numpy as np


if __name__ == '__main__':
    # Load the reconstructed snapshots from .npy files
    dense_reconstructed = np.load('dense_reconstructed_snapshot.npy')
    conv_reconstructed = np.load('conv_reconstructed_snapshot.npy')
    conv_reconstructed = conv_reconstructed.reshape(conv_reconstructed.shape[0], conv_reconstructed.shape[2])

    # Load the original snapshot
    snapshot_file = '../training_data/simulation_mu1_4.76_mu2_0.0182.npy'
    snapshot = np.load(snapshot_file)

    # Preprocess for Conv Autoencoder (remove last point for conv)
    snapshot = snapshot.T

    dense_error = np.linalg.norm(snapshot-dense_reconstructed)/np.linalg.norm(snapshot)
    conv_error = np.linalg.norm(snapshot-conv_reconstructed)/np.linalg.norm(snapshot)

    print(f"Dense autoencoder error: {dense_error}")
    print(f"Conv autoencoder error: {conv_error}")