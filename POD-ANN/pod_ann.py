import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from torch.optim.lr_scheduler import CyclicLR

# Load snapshot data
data_path = '../FEM/training_data/'
files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
all_snapshots = []

for file in files:
    snapshots = np.load(file)
    all_snapshots.append(snapshots)

all_snapshots = np.hstack(all_snapshots)  # Ensure shape is (248000, 513)

# Perform SVD on the snapshots without normalization
U, S, VT = np.linalg.svd(all_snapshots, full_matrices=False)

# Set the number of modes for principal and secondary modes
r = 28  # Number of principal modes
R = 301  # Total number of modes

U_p = U[:, :r]
U_s = U[:, r:R]
U_combined = np.hstack((U_p, U_s))  # Combine U_p and U_s into a single matrix

# Save U_p and U_s
np.save('U_p.npy', U_p)
np.save('U_s.npy', U_s)

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

# Prepare training data
q = U_combined.T @ all_snapshots  # Project snapshots onto the combined POD basis
q_p = q[:r, :]  # Principal modes
q_s = q[r:R, :]  # Secondary modes

# # Normalize the q_p and q_s modes for the neural network
q_p_mean = np.mean(q_p, axis=1, keepdims=True)
q_p_std = np.std(q_p, axis=1, keepdims=True)
# q_p_normalized = (q_p - q_p_mean) / q_p_std

q_s_mean = np.mean(q_s, axis=1, keepdims=True)
q_s_std = np.std(q_s, axis=1, keepdims=True)
# q_s_normalized = (q_s - q_s_mean) / q_s_std

# Convert to PyTorch tensors
q_p_tensor = torch.tensor(q_p.T, dtype=torch.float32)  # Transpose to (n_samples, r)
q_s_tensor = torch.tensor(q_s.T, dtype=torch.float32)  # Transpose to (n_samples, R-r)

# Split data into training and testing sets
q_p_train, q_p_test, q_s_train, q_s_test = train_test_split(q_p_tensor, q_s_tensor, test_size=0.2, random_state=42)

# Create data loaders
train_dataset = torch.utils.data.TensorDataset(q_p_train, q_s_train)
test_dataset = torch.utils.data.TensorDataset(q_p_test, q_s_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
input_dim = r
output_dim = R - r
model = POD_ANN(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, mode='triangular')


# Train the ANN model
def train_ANN(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=50, clip_value=1.0):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # Clip gradients
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)

        val_loss /= len(test_loader.dataset)
        scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}')


# Train the model
train_ANN(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=100)

# Save the complete model
torch.save(model, 'pod_ann_model.pth')





