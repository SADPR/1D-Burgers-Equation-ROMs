import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.animation import FuncAnimation

# Define the moving Gaussian function
def gaussian_function(x, center):
    return np.exp(-(x-center)**2)

# Create a range of x values
x = np.linspace(-3, 3, 400)

# Define the centers where the Gaussian peak will move
centers = np.linspace(-1, 1, 20)  # Moving peak from -1 to 1

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Moving Gaussian Function with Zoomed Inset')
ax.set_ylim(-0.1, 1.1)
ax.set_xlim(-3, 3)

# Create the zoomed-in region (inset)
axins = zoomed_inset_axes(ax, 3, loc='upper left')
plt.xticks(visible=False)
plt.yticks(visible=False)

# Create the initial inset marking
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")

# Function to update the Gaussian and the zoomed-in region for each frame
def update(frame):
    center = centers[frame]
    y = gaussian_function(x, center)

    ax.cla()  # Clear the main axis and replot
    ax.plot(x, y, label='Moving Gaussian Function')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-3, 3)
    ax.set_title(f'Moving Gaussian Function with Center at x = {center:.1f}')

    axins.cla()  # Clear the inset and replot
    axins.plot(x, y)
    axins.set_xlim(center - 0.2, center + 0.2)
    axins.set_ylim(0.95, 1.05)

    # Re-draw the inset marking
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")

# Create the animation
ani = FuncAnimation(fig, update, frames=len(centers), repeat=True, interval=200)

# Save the animation as a GIF
# ani.save('/mnt/data/moving_gaussian_function_animation.gif', writer='imagemagick')

# Display the animation
plt.show()















