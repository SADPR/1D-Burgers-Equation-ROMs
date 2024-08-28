import numpy as np
import matplotlib.pyplot as plt

# Enable LaTeX text rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def load_losses(latent_dim):
    """Load the training and validation losses for a given latent dimension."""
    train_losses = np.load(f'train_losses_latent_{latent_dim}.npy')
    val_losses = np.load(f'val_losses_latent_{latent_dim}.npy')
    return train_losses, val_losses

# Load the losses for each latent dimension
latent_dims = [3, 16, 28]
train_losses = []
val_losses = []

for dim in latent_dims:
    train_loss, val_loss = load_losses(dim)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Plotting the losses with logarithmic y-axis
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

for i, dim in enumerate(latent_dims):
    axes[i, 0].plot(train_losses[i], label=f'Train Losses (Latent {dim})', color='blue')
    axes[i, 1].plot(val_losses[i], label=f'Validation Losses (Latent {dim})', color='red')

    axes[i, 0].set_yscale('log')
    axes[i, 1].set_yscale('log')

    axes[i, 0].set_title(f'Training Losses (Latent Dimension = {dim})')
    axes[i, 1].set_title(f'Validation Losses (Latent Dimension = {dim})')

    for ax in axes[i, :]:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)

# Adjust layout
plt.tight_layout()

# Save the figure as a PDF
plt.savefig('train_val_losses_comparison_logy.pdf', format='pdf')

plt.show()

