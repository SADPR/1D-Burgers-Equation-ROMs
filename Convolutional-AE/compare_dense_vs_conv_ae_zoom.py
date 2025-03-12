import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox, TransformedBbox
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from matplotlib.animation import FuncAnimation, PillowWriter

def custom_mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)

    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2

def animate_zoom(X_dense, X_conv, original_snapshot_dense, original_snapshot_conv, dense_reconstructed, conv_reconstructed, nTimeSteps, At, latent_dim_dense, latent_dim_conv, zoom_window_size):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(min(X_dense[0], X_conv[0]), max(X_dense[-1], X_conv[-1]))
    ax.set_ylim(0, 10)

    # Transpose the snapshots to have time steps as the first dimension
    original_snapshot_dense = original_snapshot_dense.T
    dense_reconstructed = dense_reconstructed.T
    original_snapshot_conv = original_snapshot_conv.T
    conv_reconstructed = conv_reconstructed.T

    # Initial plot
    line_original_dense, = ax.plot(X_dense, original_snapshot_dense[:, 0], 'b-', label='Original Snapshot')
    line_dense, = ax.plot(X_dense, dense_reconstructed[:, 0], 'g--', label=f'Dense Autoencoder Reconstructed (latent dim={latent_dim_dense})')
    line_conv, = ax.plot(X_conv, conv_reconstructed[:, 0], 'r--', label=f'Conv Autoencoder Reconstructed (latent dim={latent_dim_conv})')

    ax.set_title('Snapshot Comparison')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

    # Create the zoom box (inset)
    axins = ax.inset_axes([0.05, 0.6, 0.35, 0.35])  # [x0, y0, width, height]
    axins.plot(X_dense, original_snapshot_dense[:, 0], 'b-')
    axins.plot(X_dense, dense_reconstructed[:, 0], 'g--')
    axins.plot(X_conv, conv_reconstructed[:, 0], 'r--')

    # Initial mark of the inset using the custom mark_inset function
    inset_mark = custom_mark_inset(ax, axins, loc1a=1, loc1b=2, loc2a=3, loc2b=4, fc="none", ec="0.5", linestyle="--")

    # Function to update the plots for each frame
    def update(frame):
        nonlocal inset_mark

        # Update main plot
        line_original_dense.set_ydata(original_snapshot_dense[:, frame])
        line_dense.set_ydata(dense_reconstructed[:, frame])
        line_conv.set_ydata(conv_reconstructed[:, frame])

        # Detect discontinuity
        differences = np.abs(np.diff(original_snapshot_dense[:, frame]))
        discontinuity_index = np.argmax(differences)
        zoom_center = X_dense[discontinuity_index]

        # Update zoom box
        zoom_start = max(zoom_center - zoom_window_size, X_dense[0])
        zoom_end = min(zoom_center + zoom_window_size/2, X_dense[-1])

        # Get the y_max of the zoomed region
        zoomed_data_original = original_snapshot_dense[(X_dense >= zoom_start) & (X_dense <= zoom_end), frame]
        y_max = zoomed_data_original.max()

        axins.cla()
        axins.plot(X_dense, original_snapshot_dense[:, frame], 'b-')
        axins.plot(X_dense, dense_reconstructed[:, frame], 'g--')
        axins.plot(X_conv, conv_reconstructed[:, frame], 'r--')
        axins.set_xlim(zoom_start, zoom_end)
        axins.set_ylim(y_max - 0.5, y_max + 0.5)

        ax.set_title(f'Snapshot Comparison at t = {frame * At:.2f}')

        # Remove previous inset mark and create new one
        for artist in inset_mark:
            artist.remove()

        inset_mark = custom_mark_inset(ax, axins, loc1a=4, loc1b=1, loc2a=3, loc2b=2, fc="none", ec="0.5", linestyle="--")

    # Create the animation
    ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, interval=100)

    # Save animation as GIF
    ani.save("dense_vs_conv_ae_comparison.gif", writer=PillowWriter(fps=10))

    plt.show()

if __name__ == '__main__':
    # Load the reconstructed snapshots from .npy files
    dense_reconstructed = np.load('dense_reconstructed_snapshot.npy')
    conv_reconstructed = np.load('conv_reconstructed_snapshot.npy')

    # Load the original snapshot
    snapshot_file = '../training_data/simulation_mu1_4.76_mu2_0.0182.npy'
    snapshot = np.load(snapshot_file)

    # Preprocess for Dense Autoencoder
    snapshot_dense = snapshot.T

    # Preprocess for Conv Autoencoder (remove last point for conv)
    snapshot_conv = snapshot.T

    # Domain
    a = 0
    b = 100
    m_dense = int(256 * 2)
    m_conv = int(256 * 2)   # Remove one node for conv
    X_dense = np.linspace(a, b, m_dense + 1)
    X_conv = np.linspace(a, b, m_conv + 1)

    # Time discretization and numerical diffusion
    Tf = 35
    At = 0.07
    nTimeSteps = int(Tf / At)

    # Create dynamic zoom GIF
    zoom_window_size = 4  # Width of the zoom window
    conv_reconstructed = conv_reconstructed.reshape(conv_reconstructed.shape[0], conv_reconstructed.shape[2])
    animate_zoom(X_dense, X_conv, snapshot_dense, snapshot_conv, dense_reconstructed, conv_reconstructed, nTimeSteps, At, 16, 3, zoom_window_size)
