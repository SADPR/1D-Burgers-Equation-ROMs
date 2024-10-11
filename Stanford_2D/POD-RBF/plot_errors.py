import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

# Enable LaTeX text rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Kernel LaTeX equations dictionary
kernel_equations = {
    'gaussian': r'$\varphi(r) = \exp\left(-(\epsilon r)^2\right)$',
    'multiquadric': r'$\varphi(r) = \sqrt{1 + (\epsilon r)^2}$',
    'inverse_multiquadric': r'$\varphi(r) = \frac{1}{\sqrt{1 + (\epsilon r)^2}}$',
    'linear': r'$\varphi(r) = r$',
    'cubic': r'$\varphi(r) = r^3$',
    'thin_plate_spline': r'$\varphi(r) = r^2 \log(r)$',
    'power': r'$\varphi(r) = r^p$ (where $p$ is a positive real number)',
    'exponential': r'$\varphi(r) = \exp(-\epsilon r)$',
    'polyharmonic_spline': r'$\varphi(r) = r^2 \log(r)$'
}

# Function to create 3D surface plots for error visualization and save them
def plot_and_save_3d_errors_with_suggestion(csv_file, lambda_weight=0.1):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Create a directory for saving the plots
    results_dir = "Error_Plots"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Get the maximum number of neighbors for normalization
    max_neighbors = df['neighbors'].max()

    # Group by kernel type for plotting
    for kernel in df['kernel_type'].unique():
        # Filter the DataFrame for the current kernel
        kernel_data = df[df['kernel_type'] == kernel]

        # Extract values for plotting
        epsilon_vals = np.log10(kernel_data['epsilon'])  # Log scale for epsilon
        neighbors_vals = kernel_data['neighbors']
        error_vals = kernel_data['reconstruction_error']

        # Find the best (minimum) error for the current kernel
        best_idx = kernel_data['reconstruction_error'].idxmin()
        best_epsilon = kernel_data.loc[best_idx, 'epsilon']
        best_neighbors = kernel_data.loc[best_idx, 'neighbors']
        best_error = kernel_data.loc[best_idx, 'reconstruction_error']

        # Calculate the trade-off score for each configuration
        kernel_data['trade_off_score'] = kernel_data['reconstruction_error'] + \
                                         lambda_weight * (kernel_data['neighbors'] / max_neighbors)

        # Find the configuration with the lowest trade-off score
        best_tradeoff_idx = kernel_data['trade_off_score'].idxmin()
        suggested_epsilon = kernel_data.loc[best_tradeoff_idx, 'epsilon']
        suggested_neighbors = kernel_data.loc[best_tradeoff_idx, 'neighbors']
        suggested_error = kernel_data.loc[best_tradeoff_idx, 'reconstruction_error']

        # Create a 3D plot for reconstruction errors
        fig = plt.figure(figsize=(10, 10))  # Square figure
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_trisurf(epsilon_vals, neighbors_vals, error_vals, cmap=cm.viridis, edgecolor='none', alpha=0.9)

        # Set axis labels and title with LaTeX formatting
        ax.set_xlabel(r'$\log(\epsilon)$', fontsize=14)
        ax.set_ylabel(r'Neighbors', fontsize=14)
        ax.set_zlabel(r'Error', fontsize=14)
        ax.set_title(rf'Reconstruction Error for Kernel: {kernel}' '\n' + rf'{kernel_equations.get(kernel, "Unknown")}', fontsize=16)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        # Annotate the best configuration and the suggested trade-off configuration
        ax.text2D(0.05, -0.1, f"Best Error: {best_error:.4f}\n"
                               f"Neighbors: {best_neighbors}, Epsilon: {best_epsilon}\n\n"
                               f"Suggested (Trade-off): Error: {suggested_error:.4f}\n"
                               f"Neighbors: {suggested_neighbors}, Epsilon: {suggested_epsilon}",
                  transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

        # Save the figure as a PNG file
        plot_filename = os.path.join(results_dir, f'3D_Error_Plot_{kernel}.png')
        plt.savefig(plot_filename)
        print(f"Plot saved: {plot_filename}")

        # Close the plot to free memory
        plt.close(fig)
        # plt.show()

    print("3D surface plots for reconstruction errors with trade-off suggestions have been generated and saved.")

# Example usage
if __name__ == '__main__':
    # File containing the exploration results
    results_csv = 'FOM_vs_POD-RBF_Exploration_Results.csv'

    # Generate and save the 3D error plots with trade-off suggestions
    plot_and_save_3d_errors_with_suggestion(results_csv, lambda_weight=0.1)


