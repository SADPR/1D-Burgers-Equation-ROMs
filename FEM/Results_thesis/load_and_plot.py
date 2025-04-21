import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set matplotlib to use LaTeX for text rendering and serif font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def plot_simulation_results_3d():
    # Load domain
    a = 0
    b = 100
    m = int(256 * 2)
    X = np.linspace(a, b, m + 1)

    # Time discretization
    At = 0.05
    times_of_interest = [5, 10, 15, 20]
    time_indices = [int(t / At) for t in times_of_interest]

    # Parameters ranges
    mu1_values = np.linspace(4.250, 5.500, 3)  # Same values as used in simulation
    mu2_values = np.linspace(0.0150, 0.0300, 3)  # Same values as used in simulation
    
    # Define the colors
    colors_mu1 = ['#FF0000', '#0000FF', '#006400', '#000000', '#FF8C00']  # Red, Blue, Dark Green, Black, Dark Orange

    # Directory where results are saved
    save_dir = "simulation_results"

    # Plot 1: Effect of Varying mu1 for Fixed mu2
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    fixed_mu2 = 0.0225  # Set the fixed value of mu2

    for i, mu1 in enumerate(mu1_values):
        filename = f"{save_dir}/fem_simulation_mu1_{mu1:.3f}_mu2_{fixed_mu2:.4f}.npy"
        U_FOM = np.load(filename)
        
        # Plotting the entire time series in one go, using a single color per mu1 value
        ax.plot(X, U_FOM[:, time_indices[0]], zs=times_of_interest[0], zdir='y', color=colors_mu1[i], label=f'$\mu_1={mu1:.3f}$')
        ax.plot(X, U_FOM[:, time_indices[1]], zs=times_of_interest[1], zdir='y', color=colors_mu1[i])
        ax.plot(X, U_FOM[:, time_indices[2]], zs=times_of_interest[2], zdir='y', color=colors_mu1[i])
        ax.plot(X, U_FOM[:, time_indices[3]], zs=times_of_interest[3], zdir='y', color=colors_mu1[i])

    ax.set_xlabel(r'$x$')
    ax.set_zlabel(r'$u$')
    ax.set_ylabel(r'$t$')
    ax.set_xlim([0, 100])
    ax.set_ylim([min(times_of_interest), max(times_of_interest)])
    ax.set_zlim([0.5,8.0])
    ax.view_init(elev=30, azim=-60)  # Adjust the viewing angle for better visibility

    # Add the legend with only one label per mu1 value
    ax.legend()

    # Center the title of the plot and adjust the spacing
    plt.title(r'3D Plot: Effect of Varying $\mu_1$ for Fixed $\mu_2=0.0225$', pad=20, loc='center')
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save the figure as a PDF
    plt.savefig("plot_mu1_variation.pdf", format='pdf')

    plt.show()

    # Plot 2: Effect of Varying mu2 for Fixed mu1
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    fixed_mu1 = 4.250  # Set the fixed value of mu1 (choose a middle value)

    for i, mu2 in enumerate(mu2_values):
        filename = f"{save_dir}/fem_simulation_mu1_{fixed_mu1:.3f}_mu2_{mu2:.4f}.npy"
        U_FOM = np.load(filename)
        
        # Plotting the entire time series in one go, using a single color per mu2 value
        ax.plot(X, U_FOM[:, time_indices[0]], zs=times_of_interest[0], zdir='y', color=colors_mu1[i], label=f'$\mu_2={mu2:.4f}$')
        ax.plot(X, U_FOM[:, time_indices[1]], zs=times_of_interest[1], zdir='y', color=colors_mu1[i])
        ax.plot(X, U_FOM[:, time_indices[2]], zs=times_of_interest[2], zdir='y', color=colors_mu1[i])
        ax.plot(X, U_FOM[:, time_indices[3]], zs=times_of_interest[3], zdir='y', color=colors_mu1[i])

    ax.set_xlabel(r'$x$')
    ax.set_zlabel(r'$u$')
    ax.set_ylabel(r'$t$')
    ax.set_xlim([0, 100])
    ax.set_ylim([min(times_of_interest), max(times_of_interest)])
    ax.set_zlim([0.5,8.0])
    ax.view_init(elev=30, azim=-60)  # Adjust the viewing angle for better visibility

    # Add the legend with only one label per mu2 value
    ax.legend()

    # Center the title of the plot and adjust the spacing
    plt.title(r'3D Plot: Effect of Varying $\mu_2$ for Fixed $\mu_1=4.250$', pad=20, loc='center')
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save the figure as a PDF
    plt.savefig("plot_mu2_variation.pdf", format='pdf')

    plt.show()

def plot_simulation_results_subplots():
    # Load domain
    a = 0
    b = 100
    m = int(256 * 2)
    X = np.linspace(a, b, m + 1)

    # Time discretization
    At = 0.07
    times_of_interest = [5, 10, 15, 20]
    time_indices = [int(t / At) for t in times_of_interest]

    # Parameters ranges
    mu1_values = np.linspace(4.250, 5.500, 3)  # Same values as used in simulation
    mu2_values = np.linspace(0.0150, 0.0300, 3)  # Same values as used in simulation
    
    colors_mu1 = ['#FF0000', '#0000FF', '#006400', '#000000', '#FF8C00']  # Red, Blue, Dark Green, Black, Dark Orange

    # Directory where results are saved
    save_dir = "simulation_results"

    # Subplot for varying mu1 with fixed mu2
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    axs = axs.flatten()

    fixed_mu2 = 0.0225  # Set the fixed value of mu2

    for ax, t_index in zip(axs, time_indices):
        for i, mu1 in enumerate(mu1_values):
            filename = f"{save_dir}/fem_simulation_mu1_{mu1:.3f}_mu2_{fixed_mu2:.4f}.npy"
            U_FOM = np.load(filename)
            ax.plot(X, U_FOM[:, t_index], label=f'$\mu_1$={mu1:.3f}', color=colors_mu1[i])
        
        ax.set_title(f'$t = {times_of_interest[time_indices.index(t_index)]} s$')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$u$')
        ax.set_xlim([0, 100])
        ax.set_ylim([0.5,6.5])
        ax.grid(True)
        if t_index == time_indices[0]:
            ax.legend()

    plt.suptitle(r'2D Plot: Effect of Varying $\mu_1$ for Fixed $\mu_2=0.0225$', fontsize=16)
    
    # Adjust the layout and spacing
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rectangle in which subplots will fit
    plt.subplots_adjust(top=0.9)  # Adjust top to make space for suptitle

    # Save the figure as a PDF
    plt.savefig("subplot_mu1_variation.pdf", format='pdf')

    plt.show()

    # Subplot for varying mu2 with fixed mu1
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    axs = axs.flatten()

    fixed_mu1 = 4.250  # Set the fixed value of mu1

    for ax, t_index in zip(axs, time_indices):
        for i, mu2 in enumerate(mu2_values):
            filename = f"{save_dir}/fem_simulation_mu1_{fixed_mu1:.3f}_mu2_{mu2:.4f}.npy"
            U_FOM = np.load(filename)
            ax.plot(X, U_FOM[:, t_index], label=f'$\mu_2$={mu2:.4f}', color=colors_mu1[i])
        
        ax.set_title(f'$t = {times_of_interest[time_indices.index(t_index)]} s$')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$u$')
        ax.set_xlim([0, 100])
        ax.set_ylim([0.5,6.5])
        ax.grid(True)
        if t_index == time_indices[0]:
            ax.legend()

    plt.suptitle(r'2D Plot: Effect of Varying $\mu_2$ for Fixed $\mu_1=4.250$', fontsize=16)

    # Adjust the layout and spacing
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rectangle in which subplots will fit
    plt.subplots_adjust(top=0.9)  # Adjust top to make space for suptitle

    # Save the figure as a PDF
    plt.savefig("subplot_mu2_variation.pdf", format='pdf')

    plt.show()


def plot_simulation_results_subplots_horizontal():
    # Load domain
    a = 0
    b = 100
    m = int(256 * 2)
    X = np.linspace(a, b, m + 1)

    # Time discretization
    At = 0.07
    times_of_interest = [5, 10, 15, 20]
    time_indices = [int(t / At) for t in times_of_interest]

    # Parameters ranges
    mu1_values = np.linspace(4.250, 5.500, 3)  # Same values as used in simulation
    mu2_values = np.linspace(0.0150, 0.0300, 3)  # Same values as used in simulation
    
    colors_mu1 = ['#FF0000', '#0000FF', '#006400', '#000000', '#FF8C00']  # Red, Blue, Dark Green, Black, Dark Orange

    # Directory where results are saved
    save_dir = "simulation_results"

    # Subplot for varying mu1 with fixed mu2
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))  # 1x4 grid layout

    fixed_mu2 = 0.0225  # Set the fixed value of mu2

    for ax, t_index in zip(axs, time_indices):
        for i, mu1 in enumerate(mu1_values):
            filename = f"{save_dir}/fem_simulation_mu1_{mu1:.3f}_mu2_{fixed_mu2:.4f}.npy"
            U_FOM = np.load(filename)
            ax.plot(X, U_FOM[:, t_index], label=f'$\mu_1$={mu1:.3f}', color=colors_mu1[i])
        
        ax.set_title(f'$t = {times_of_interest[time_indices.index(t_index)]} s$')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$u$')
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 7])
        ax.grid(True)
        if t_index == time_indices[0]:
            ax.legend()

    plt.suptitle(r'2D Plot: Effect of Varying $\mu_1$ for Fixed $\mu_2=0.0225$', fontsize=16)

    # Save the figure as a PDF
    plt.savefig("subplot_mu1_variation.pdf", format='pdf')

    plt.show()

    # Subplot for varying mu2 with fixed mu1
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))  # 1x4 grid layout

    fixed_mu1 = 4.250  # Set the fixed value of mu1

    for ax, t_index in zip(axs, time_indices):
        for i, mu2 in enumerate(mu2_values):
            filename = f"{save_dir}/fem_simulation_mu1_{fixed_mu1:.3f}_mu2_{mu2:.4f}.npy"
            U_FOM = np.load(filename)
            ax.plot(X, U_FOM[:, t_index], label=f'$\mu_2$={mu2:.4f}', color=colors_mu1[i])
        
        ax.set_title(f'$t = {times_of_interest[time_indices.index(t_index)]} s$')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$u$')
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 7])
        ax.grid(True)
        if t_index == time_indices[0]:
            ax.legend()

    plt.suptitle(r'2D Plot: Effect of Varying $\mu_2$ for Fixed $\mu_1=4.250$', fontsize=16)

    # Save the figure as a PDF
    plt.savefig("subplot_mu2_variation.pdf", format='pdf')

    plt.show()


if __name__ == "__main__":
    plot_simulation_results_3d()
    plot_simulation_results_subplots()
    # plot_simulation_results_subplots_horizontal()








