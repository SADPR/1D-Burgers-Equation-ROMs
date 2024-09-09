import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from matplotlib.transforms import TransformedBbox
from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector

# Enable LaTeX text rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def compute_l2_norm_error(U_FOM, U_ROM):
    return np.linalg.norm(U_FOM - U_ROM) / np.linalg.norm(U_FOM)

# Custom mark_inset function for better control over zoom box
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

def plot_fom_vs_pod_ann_with_zoom():
    # Load domain
    a = 0
    b = 100
    m = int(256 * 2)
    X = np.linspace(a, b, m + 1)

    # Time discretization
    At = 0.07
    times_of_interest = [7, 14, 21]  # seconds
    time_indices = [int(t / At) for t in times_of_interest]

    # Load FOM results
    fom_filename = "simulation_mu1_4.76_mu2_0.0182.npy"
    U_FOM = np.load(fom_filename)

    # Load POD-ANN results
    pod_ann_filename = "U_POD_ANN_PROM_solution.npy"
    U_POD_ANN_PROM = np.load(pod_ann_filename)[:, :-1]  # Remove last timestep if necessary

    # Plotting with zoom boxes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot FOM results with increased line width
    ax.plot(X, U_FOM[:, time_indices[0]], color='black', linestyle='-', linewidth=3, label='FOM')
    for t_index in time_indices[1:]:
        ax.plot(X, U_FOM[:, t_index], color='black', linestyle='-', linewidth=3)

    # Plot POD-ANN results with increased line width
    ax.plot(X, U_POD_ANN_PROM[:, time_indices[0]], color='red', linestyle='--', linewidth=3, label='POD-ANN (28 primary modes, 301 secondary modes)')
    for t_index in time_indices[1:]:
        ax.plot(X, U_POD_ANN_PROM[:, t_index], color='red', linestyle='--', linewidth=3)

    # Zoom box 1 for time_index = 0 (same as before, downward discontinuity)
    differences = np.abs(np.diff(U_FOM[:, time_indices[0]]))
    discontinuity_index = np.argmax(differences)
    zoom_center = X[discontinuity_index]
    zoom_window_size = 5
    axins1 = ax.inset_axes([0.05, 0.6, 0.25, 0.25])  # zoomed-in box for time_index=0

    axins1.plot(X, U_FOM[:, time_indices[0]], color='black', linestyle='-', linewidth=2.5)  # Increased line width in zoom
    axins1.plot(X, U_POD_ANN_PROM[:, time_indices[0]], color='red', linestyle='--', linewidth=2.5)  # Increased line width in zoom

    zoom_start = max(zoom_center - zoom_window_size / 2, X[0])
    zoom_end = min(zoom_center, X[-1])
    zoomed_data_original = U_FOM[(X >= zoom_start) & (X <= zoom_end), time_indices[0]]
    y_max = zoomed_data_original.max()
    axins1.set_xlim(zoom_start, zoom_end)
    axins1.set_ylim(y_max - 0.2, y_max + 0.2)

    custom_mark_inset(ax, axins1, loc1a=4, loc1b=1, loc2a=3, loc2b=2, fc="none", ec="0.5")

    # Zoom box 2 for time_index = 2 (hardcoded at x=67.5, y=2.05)
    axins2 = ax.inset_axes([0.6, 0.6, 0.25, 0.25])  # zoomed-in box for time_index=2

    axins2.plot(X, U_FOM[:, time_indices[2]], color='black', linestyle='-', linewidth=2.5)  # Increased line width in zoom
    axins2.plot(X, U_POD_ANN_PROM[:, time_indices[2]], color='red', linestyle='--', linewidth=2.5)  # Increased line width in zoom

    # Set zoom limits based on the hardcoded center at x=67.5, y=2.05
    zoom_center2_x = 68
    zoom_center2_y = 2.15
    zoom_window_size_x2 = 2.5  # Width for the zoom window in the x-direction
    zoom_window_size_y2 = 0.4  # Height for the zoom window in the y-direction

    axins2.set_xlim(zoom_center2_x - zoom_window_size_x2 / 2, zoom_center2_x + zoom_window_size_x2 / 2)
    axins2.set_ylim(zoom_center2_y - zoom_window_size_y2 / 2, zoom_center2_y + zoom_window_size_y2 / 2)

    custom_mark_inset(ax, axins2, loc1a=3, loc1b=2, loc2a=4, loc2b=1, fc="none", ec="0.5")

    # Calculate and print L2 norm error for POD-ANN
    l2_error_pod_ann = compute_l2_norm_error(U_FOM, U_POD_ANN_PROM)
    print(f"POD-ANN (28 primary modes, 301 secondary modes) L2 Error = {l2_error_pod_ann:.6f}")

    # Annotate the plot
    ax.set_title(f'FOM vs PROM-LSPG with POD-ANN (28 Primary Modes, 301 Secondary Modes)')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u$')
    ax.set_ylim(0, 10)
    ax.set_xlim(0, 100)
    ax.legend(loc='upper right')
    ax.grid(True)

    # Adjust the layout
    plt.tight_layout()

    # Save the figure as a PDF
    plt.savefig("fom_vs_pod_ann_zoom_comparison_28_modes.pdf", format='pdf')
    plt.show()

# Run the comparison plot with zoom
plot_fom_vs_pod_ann_with_zoom()








