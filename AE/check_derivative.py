import torch
import torch.nn as nn
import numpy as np

# Define the DenseAutoencoder class
class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DenseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 513),
            nn.ELU(),
            nn.Linear(513, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, 513),
            nn.ELU(),
            nn.Linear(513, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

    def compute_jacobian(self, q):
        # Ensure q requires gradients
        q = q.clone().detach().requires_grad_(True)
        
        # Forward pass through the decoder
        decoded = self.decoder(q.unsqueeze(0))
        
        # Initialize the Jacobian list
        jacobian = []
        
        # Loop over each output element
        for i in range(decoded.shape[1]):
            grad_outputs = torch.zeros_like(decoded)
            grad_outputs[0, i] = 1  # Isolate the i-th output element
            grad_q = torch.autograd.grad(outputs=decoded, inputs=q, grad_outputs=grad_outputs, create_graph=True)[0]
            jacobian.append(grad_q)
        
        # Stack the gradients to form the Jacobian matrix
        return torch.stack(jacobian, dim=1).squeeze(0)


# Function to compute the finite difference Jacobian
def finite_difference_jacobian(model, q, h=1e-5):
    q = q.clone().detach()
    num_outputs = model.decoder(q.unsqueeze(0)).shape[1]
    num_inputs = q.shape[0]
    jacobian = np.zeros((num_outputs, num_inputs))

    for i in range(num_inputs):
        q_plus = q.clone()
        q_minus = q.clone()

        q_plus[i] += h
        q_minus[i] -= h

        output_plus = model.decoder(q_plus.unsqueeze(0)).detach().numpy().squeeze()
        output_minus = model.decoder(q_minus.unsqueeze(0)).detach().numpy().squeeze()

        jacobian[:, i] = (output_plus - output_minus) / (2 * h)

    return jacobian

# Function to compare the analytical and numerical Jacobians
def compare_jacobians(analytical_jacobian, numerical_jacobian):
    difference = np.linalg.norm(analytical_jacobian - numerical_jacobian)
    print(f"Difference between analytical and finite difference Jacobians: {difference}")
    return difference

# Example usage
if __name__ == "__main__":
    # Load your trained autoencoder model
    input_dim = 513
    latent_dim = 28
    model = torch.load(f'dense_autoencoder_complete_latent_{latent_dim}.pth')
    model.eval()

    # Load the snapshot
    snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'
    snapshot = np.load(snapshot_file)  # Assuming the snapshot is transposed to match the input dimension

    # Select a specific column, e.g., column 100
    snapshot_column = snapshot[:, 100]

    # Convert the selected column to a PyTorch tensor
    snapshot_tensor = torch.tensor(snapshot_column, dtype=torch.float32)

    # Encode the selected snapshot column to obtain q
    q = model.encoder(snapshot_tensor.unsqueeze(0)).detach().squeeze(0)

    # Compute the analytical Jacobian
    analytical_jacobian = model.compute_jacobian(q).detach().numpy().T

    # Compute the numerical Jacobian using finite differences
    numerical_jacobian = finite_difference_jacobian(model, q)

    # Compare the two Jacobians
    difference = compare_jacobians(analytical_jacobian, numerical_jacobian)

    # Define a tolerance to check if the difference is acceptable
    tolerance = 1e-6
    if difference < tolerance:
        print("The Jacobian seems to be correctly implemented.")
    else:
        print("There might be an issue with the Jacobian implementation.")



