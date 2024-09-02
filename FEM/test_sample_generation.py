import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import os

# Load the existing LHS samples and parameter combinations
lhs_samples = np.load('parameter_combinations/lhs_samples.npy')

# Define the original parameter ranges for mu1 and mu2
param_ranges = [(4.25, 5.5), (0.015, 0.03)]

# Number of testing samples to generate within each subrange
n_testing_samples = 1  # One per subrange

# Divide the parameter space into a 3x3 grid
mu1_ranges = np.linspace(param_ranges[0][0], param_ranges[0][1], 4)  # 3 intervals for mu1
mu2_ranges = np.linspace(param_ranges[1][0], param_ranges[1][1], 4)  # 3 intervals for mu2

final_testing_samples = []

# Iterate over each subrange
for i in range(3):
    for j in range(3):
        # Define the subrange
        mu1_subrange = (mu1_ranges[i], mu1_ranges[i+1])
        mu2_subrange = (mu2_ranges[j], mu2_ranges[j+1])

        # Generate random samples within this subrange
        n_random_samples = 1000  # Generate 1000 samples per subrange
        random_samples = np.random.rand(n_random_samples, 2)
        random_samples[:, 0] = random_samples[:, 0] * (mu1_subrange[1] - mu1_subrange[0]) + mu1_subrange[0]
        random_samples[:, 1] = random_samples[:, 1] * (mu2_subrange[1] - mu2_subrange[0]) + mu2_subrange[0]

        # Calculate the minimum distance from each random sample to the existing LHS samples
        min_distances = np.min(distance.cdist(random_samples, lhs_samples), axis=1)

        # Select the farthest point in this subrange
        farthest_index = np.argmax(min_distances)
        farthest_sample = random_samples[farthest_index]

        # Add the selected sample to the final list
        final_testing_samples.append(farthest_sample)

final_testing_samples = np.array(final_testing_samples)

# Expand the parameter ranges for extrapolation
expanded_param_ranges = [
    (param_ranges[0][0] - 0.25, param_ranges[0][1] + 0.25),
    (param_ranges[1][0] - 0.005, param_ranges[1][1] + 0.005)
]

# Define the 4 extrapolation points outside the original parameter range
extrapolation_points = np.array([
    [expanded_param_ranges[0][0], expanded_param_ranges[1][0]],  # Below both ranges
    [expanded_param_ranges[0][1], expanded_param_ranges[1][0]],  # Above mu1, below mu2
    [expanded_param_ranges[0][0], expanded_param_ranges[1][1]],  # Below mu1, above mu2
    [expanded_param_ranges[0][1], expanded_param_ranges[1][1]]   # Above both ranges
])

# Combine the selected 9 points and 4 extrapolation points
final_testing_samples = np.vstack([final_testing_samples, extrapolation_points])

# Save the final testing samples
os.makedirs("testing_data", exist_ok=True)
np.save('testing_data/final_testing_samples.npy', final_testing_samples)

# Visualize the training and final testing samples together
plt.scatter(lhs_samples[:, 0], lhs_samples[:, 1], color='blue', label='LHS Samples (Training)')
plt.scatter(final_testing_samples[:, 0], final_testing_samples[:, 1], color='red', label='Testing Samples')
plt.xlabel('$\mu_1$')
plt.ylabel('$\mu_2$')
plt.title('Training and Testing Samples in Parameter Space')
plt.legend()
plt.savefig('testing_data/training_testing_sampling_plot.pdf')
plt.show()

# Print out the selected final testing samples
print("Selected Final Testing Samples:")
for i, (mu1, mu2) in enumerate(final_testing_samples):
    print(f"Testing sample {i+1}: mu1 = {mu1:.4f}, mu2 = {mu2:.4f}")
