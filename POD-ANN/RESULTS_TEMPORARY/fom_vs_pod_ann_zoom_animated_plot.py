import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox, TransformedBbox
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from matplotlib.animation import FuncAnimation, PillowWriter

# Enable LaTeX text rendering (optional)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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

def animate_zoom(X, original_snapshot_fom, pod_ann_reconstructed, nTimeSteps, At, zoom_window_size):
    fig, ax = plt.subplots(figsize=(7, 6))  # Consistent figure size

    # Set the grid and axis limits
    ax.grid(True)
    ax.set_xlim(min(X), max(X))
    ax.set_ylim(0, 10)

    # Transpose the snapshots to have time steps as the first dimension
    original_snapshot_fom = original_snapshot_fom
    pod_ann_reconstructed = pod_ann_reconstructed

    # Initial plot (black for FOM, red for POD-ANN)
    line_original_fom, = ax.plot(X, original_snapshot_fom[:, 0], 'k-', label='FOM Snapshot')  # Black for FOM
    line_pod_ann, = ax.plot(X, pod_ann_reconstructed[:, 0], 'r--', label=f'POD-ANN (28 primary modes, 301 secondary modes)')  # Red for POD-ANN

    # Set title, labels, and legend
    ax.set_title('Snapshot Comparison')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u$')
    ax.legend(loc='upper right')

    # Create the zoom box (inset)
    axins = ax.inset_axes([0.05, 0.6, 0.35, 0.35])  # Position and size of inset
    axins.plot(X, original_snapshot_fom[:, 0], 'k-')
    axins.plot(X, pod_ann_reconstructed[:, 0], 'r--')

    # Initial mark of the inset using the custom mark_inset function
    inset_mark = custom_mark_inset(ax, axins, loc1a=4, loc1b=1, loc2a=3, loc2b=2, fc="none", ec="0.5", linestyle="--")

    # Function to update the plots for each frame
    def update(frame):
        nonlocal inset_mark

        # Update main plot
        line_original_fom.set_ydata(original_snapshot_fom[:, frame])
        line_pod_ann.set_ydata(pod_ann_reconstructed[:, frame])

        # Detect discontinuity for zoom
        differences = np.abs(np.diff(original_snapshot_fom[:, frame]))
        discontinuity_index = np.argmax(differences)
        zoom_center = X[discontinuity_index]

        # Update zoom box
        zoom_start = max(zoom_center - zoom_window_size, X[0])
        zoom_end = min(zoom_center + zoom_window_size/2, X[-1])

        # Get the y_max of the zoomed region
        zoomed_data_original = original_snapshot_fom[(X >= zoom_start) & (X <= zoom_end), frame]
        y_max = zoomed_data_original.max()

        axins.cla()
        axins.plot(X, original_snapshot_fom[:, frame], 'k-')
        axins.plot(X, pod_ann_reconstructed[:, frame], 'r--')
        axins.set_xlim(zoom_start, zoom_end)
        axins.set_ylim(y_max - 0.5, y_max + 0.5)

        # Update title with consistent time format
        ax.set_title(f'Snapshot Comparison at t = {frame * At:.2f}')

        # Remove previous inset mark and create a new one
        for artist in inset_mark:
            artist.remove()
        inset_mark = custom_mark_inset(ax, axins, loc1a=4, loc1b=1, loc2a=3, loc2b=2, fc="none", ec="0.5", linestyle="--")

    # Create the animation
    ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, interval=1000 / 30)  # Set FPS to 30

    # Save animation as GIF with PillowWriter at 30 fps
    ani.save("fom_vs_pod_ann_comparison.gif", writer=PillowWriter(fps=30))

    plt.close()

if __name__ == '__main__':
    # Load the reconstructed snapshots from .npy files
    pod_ann_reconstructed = np.load('U_POD_ANN_PROM_solution.npy')[:, :-1]

    # Load the original FOM snapshot
    fom_filename = 'simulation_mu1_4.76_mu2_0.0182.npy'
    original_snapshot_fom = np.load(fom_filename)

    # Domain
    a = 0
    b = 100
    m = int(256 * 2)
    X = np.linspace(a, b, m + 1)

    # Time discretization and numerical diffusion
    Tf = 35
    At = 0.07
    nTimeSteps = int(Tf / At)

    # Create dynamic zoom GIF with window size of 4
    zoom_window_size = 4  # Width of the zoom window
    animate_zoom(X, original_snapshot_fom, pod_ann_reconstructed, nTimeSteps, At, zoom_window_size)


