import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# QLF Function
def phi_M(M, logphistar, Mstar, alpha, beta):
    numerator = 10**logphistar
    denom = 10**(0.4 * (1 + alpha) * (M - Mstar)) + 10**(0.4 * (1 + beta) * (M - Mstar))
    return numerator / denom

# Parameter Ranges
logphistar_vals = np.linspace(-8.5, -6, 50)
Mstar_vals      = np.linspace(-28, -25, 50)
alpha_vals      = np.linspace(-4.5, -1.5, 50)
beta_vals       = np.linspace(-2.5, -0.5, 50)

# Fixed Initial Parameters
logphistar_0 = -8.5  # Start with first value of logphistar_vals
Mstar_0 = -28
alpha_0 = -4.5       # Start with first value of alpha_vals
beta_0 = -2.5        # Start with first value of beta_vals

M = np.linspace(-32, -18, 400)

fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot([], [], lw=2, label='QLF', color='blue')
text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, va='top')

# Create reference lines
vline = ax.axvline(x=Mstar_0, color='red', linestyle='--', alpha=0.7, label='M*')
hline = ax.axhline(y=10**logphistar_0/2, color='orange', linestyle='--', alpha=0.7, label='φ*/2')

ax.set_xlim(-18, -32)
ax.set_ylim(1e-12, 1e-3)
ax.set_yscale('log')
ax.set_xlabel('M1450')
ax.set_ylabel(r'$\phi(M)$ [Mpc$^{-3}$ mag$^{-1}$]')
ax.set_title("Quasar Luminosity Function (QLF)")
ax.legend(loc='upper right')

def update(frame):
    if frame < 50:
        # Vary logphistar, keep others at initial values
        logphistar = logphistar_vals[frame]
        Mstar, alpha, beta = Mstar_0, alpha_0, beta_0
    elif frame < 100:
        # Vary Mstar, use final logphistar value, keep others at initial values
        logphistar = logphistar_vals[-1]  # Final value from previous phase
        Mstar = Mstar_vals[frame - 50]
        alpha, beta = alpha_0, beta_0
    elif frame < 150:
        # Vary alpha, use final values from previous phases
        logphistar = logphistar_vals[-1]  # Final value
        Mstar = Mstar_vals[-1]           # Final value
        alpha = alpha_vals[frame - 100]
        beta = beta_0
    else:
        # Vary beta, use final values from all previous phases
        logphistar = logphistar_vals[-1]  # Final value
        Mstar = Mstar_vals[-1]           # Final value
        alpha = alpha_vals[-1]           # Final value
        beta = beta_vals[frame - 150]

    y = phi_M(M, logphistar, Mstar, alpha, beta)
    line.set_data(M, y)
    
    # Update reference lines
    phi_star = 10**logphistar
    phi_half = phi_star / 2
    
    # Update vertical line position for M*
    vline.set_xdata([Mstar, Mstar])
    
    # Update horizontal line position for φ*/2
    hline.set_ydata([phi_half, phi_half])
    
    text.set_text(f"logφ* = {logphistar:.2f}\nM* = {Mstar:.2f}\nα = {alpha:.2f}\nβ = {beta:.2f}\nφ*/2 = {phi_half:.2e}")
    
    return line, text, vline, hline

# Change total frames to 200 (50 frames per parameter)
ani = FuncAnimation(fig, update, frames=200, interval=300, blit=True)

# Save the animation
ani.save("qlf_parameter_demo.mp4", fps=5, dpi=150)

plt.show()