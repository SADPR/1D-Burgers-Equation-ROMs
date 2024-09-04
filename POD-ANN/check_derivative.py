import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# Define the ANN model
class POD_ANN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(POD_ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, output_dim)
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))
        x = self.elu(self.fc4(x))
        x = self.elu(self.fc5(x))
        x = self.fc6(x)
        return x

def gradient_check_pod_ann(U_p, snapshot_column, model, epsilon_values):
    # Project the snapshot_column onto the primary POD basis to get q_p
    q_p = U_p.T @ snapshot_column

    # Make q_p_tensor a leaf tensor with requires_grad=True
    q_p_tensor = torch.tensor(q_p, dtype=torch.float32).unsqueeze(0)
    
    # Compute the Jacobian
    g = torch.autograd.functional.jacobian(model, q_p_tensor).detach().numpy().squeeze()

    # Generate a random unit vector v of the same size as q_p
    v = np.random.randn(*q_p.shape)
    v /= np.linalg.norm(v) + 1e-10  # Add small value to prevent division by zero
    v_tensor = torch.tensor(v, dtype=torch.float32).unsqueeze(0)

    # Initialize list to store errors
    errors = []

    for epsilon in epsilon_values:
        # Perturb q_p and compute the ANN output for the perturbed q_p
        q_p_perturbed = q_p_tensor + epsilon * v_tensor
        q_s_perturbed = model(q_p_perturbed)

        # Calculate the error term
        q_s_perturbed_np = q_s_perturbed.detach().numpy().squeeze()
        q_s_np = model(q_p_tensor).detach().numpy().squeeze()  # Recalculate the original q_s
        error = np.linalg.norm(q_s_perturbed_np - q_s_np - epsilon * (g @ v))
        errors.append(error)

    # Plot the errors against epsilon
    plt.figure(figsize=(8, 6))
    plt.loglog(epsilon_values, errors, marker='o', label='Computed Error')

    # Add reference lines for linear (O(epsilon)) and quadratic (O(epsilon^2)) behavior
    plt.loglog(epsilon_values, epsilon_values * errors[0] / epsilon_values[0], 'r--', label=r'O($\epsilon$) Reference')
    plt.loglog(epsilon_values, epsilon_values**2 * errors[0] / epsilon_values[0]**2, 'g--', label=r'O($\epsilon^2$) Reference')

    plt.xlabel('epsilon')
    plt.ylabel('Error')
    plt.title('Gradient Check Error vs. Epsilon')
    plt.grid(True)
    plt.legend()

    # Compute and print the slope of log(err) vs log(epsilon)
    slopes = np.diff(np.log(errors)) / np.diff(np.log(epsilon_values))
    print("Slopes between consecutive points on the log-log plot:", slopes)

    plt.show()

# Example usage
snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'
snapshot = np.load(snapshot_file)  # Assuming the snapshot is in the correct shape

# Select a specific column, e.g., column 100
snapshot_column = snapshot[:, 100]

U_p = np.load('U_p.npy')  # Load your primary POD basis
epsilon_values = np.logspace(np.log10(1e-12), np.log10(10), 20)

model = torch.load('pod_ann_model.pth')  # Load your trained POD-ANN model

gradient_check_pod_ann(U_p, snapshot_column, model, epsilon_values)




