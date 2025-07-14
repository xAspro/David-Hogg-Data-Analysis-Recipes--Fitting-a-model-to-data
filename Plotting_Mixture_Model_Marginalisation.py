import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Fixed data
x = np.linspace(0, 100, 500)
y1 = 100 * np.exp(-0.5 * ((x - 50) / 10) ** 2)

# Animation function
def animate(frame):
    ax.clear()
    
    # Current n_subdivisions value
    n_subdivisions = 20 + frame * 10  # 20, 30, 40, ..., 200
    
    # Generate noise
    y2 = np.random.uniform(0, 1, size=n_subdivisions*len(x))
    y3 = np.add.reduceat(y2, np.arange(0, len(y2), n_subdivisions))
    y4 = y1 + y3
    
    # Plot y1 with shading
    ax.plot(x, y1, color='blue', label='Gaussian')
    ax.fill_between(x, y1, color='blue', alpha=0.3)
    
    # Plot y3 with shading
    ax.plot(x, y3, color='green', label='Uniform Noise')
    ax.fill_between(x, y3, color='green', alpha=0.3)
    
    # Plot y4
    ax.plot(x, y4, color='red', label='Total')
    
    # Shade the gap between y4 and y1 (which is y3)
    ax.fill_between(x, y1, y4, color='green', alpha=0.3)
    
    # Set labels and title
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(rf'$P_b=1$ case noise propagation')
    
    # Set consistent y-limits for better comparison
    ax.set_ylim(0, 250)

# Create animation
# From 20 to 200 in steps of 10: frames = (200-20)/10 + 1 = 19 frames
frames = range(19)
anim = FuncAnimation(fig, animate, frames=frames, interval=500, repeat=True)

# Save as GIF
anim.save('mixture_model_marginalisation.gif', writer='pillow', fps=2)

plt.show()

print("GIF saved as 'mixture_model_marginalisation.gif'")