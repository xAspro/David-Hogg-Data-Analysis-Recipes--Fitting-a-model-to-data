"""
Extra code, not part of David Hogg.
This code is to learn Bayesian statistics using MCMC,
and look at its counterpart, that is by updating the data.
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from matplotlib.animation import FuncAnimation
from datetime import datetime
import sys
from scipy.stats import beta
import time

start_time = time.time()


p_head = 0.15  # Probability of head in a coin flip
n = 100  # Number of samples
res=0.001  # Resolution for the posterior calculation

def make_sample_data(p_head, n=1000):
    p = [1 - p_head, p_head] 
    return np.random.choice(2, n, p=p)

data = make_sample_data(p_head, n)

def calculate_probability_using_batch_method(data):
    def log_prior(p, data):
        if 0 < p < 1:
            return 0
        else:
            return -np.inf
        
    def log_likelihood(p, data):
        return np.sum(data * np.log(p) + (1 - data) * np.log(1 - p))

    def log_posterior(p, data):
        if log_prior(p, data) == -np.inf:
            return -np.inf

        return log_likelihood(p, data) + log_prior(p, data)

    def mcmc(data, n_walkers=100, n_samples=1000, burn_in=100):
        sampler = emcee.EnsembleSampler(nwalkers=n_walkers, dim=1, 
                                        log_prob_fn=log_posterior, 
                                        args=[data])

        p0 = np.random.uniform(0, 1, size=(n_walkers, 1))
        sampler.run_mcmc(p0, n_samples + burn_in, progress=True)

        # Get all samples (no discard)
        all_samples = sampler.get_chain(flat=True)
        # Get samples after burn-in
        samples = sampler.get_chain(discard=burn_in, flat=True)

        # Plot chains (no discard)
        plt.figure(figsize=(10, 4))
        plt.plot(all_samples, alpha=0.5)
        plt.title("Chains (no discard)")
        plt.xlabel("Step")
        plt.ylabel("p")
        plt.show()

        # Plot corner (no discard)
        try:
            corner.corner(all_samples, labels=["p"], show_titles=True)
            plt.suptitle("Corner plot (no discard)")
            plt.show()
        except ImportError:
            print("corner package not installed, skipping corner plot.")

        # Plot chains (with burn-in discarded)
        plt.figure(figsize=(10, 4))
        plt.plot(samples, alpha=0.5, marker='.', linestyle='None')
        plt.title("Samples after burn-in")
        plt.xlabel("Sample")
        plt.ylabel("p")
        plt.show()

        # Plot corner (with burn-in discarded)
        try:
            corner.corner(samples, labels=["p"], show_titles=True)
            plt.suptitle("Corner plot (after burn-in)")
            plt.show()
        except ImportError:
            print("corner package not installed, skipping corner plot.")

        return samples

    samples = mcmc(data)
    return samples
# samples = calculate_probability_using_batch_method(data)


def calculate_probability_by_updating_the_data(data, res=0.001):
    theta = np.arange(0, 1 + res, res)
    prior = np.ones_like(theta) * 1 / len(theta)
    last_frame = -1

    def compute_posterior(new_data, prior):
        likelihood = theta * new_data + (1 - theta) * (1 - new_data)
        unnormalized_posterior = prior * likelihood

        evidence = np.trapezoid(unnormalized_posterior, theta)
        posterior = unnormalized_posterior / evidence

        if evidence == 0:
            sys.exit("\nEvidence is zero, cannot normalize posterior.\n")

        return posterior

    fig, ax = plt.subplots()
    line, = ax.plot(theta, prior)
    # ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("p")
    ax.set_ylabel("Posterior")
    title = ax.set_title("Frame 1")

    def update(frame):
        nonlocal prior, last_frame
        posterior = prior

        if frame > last_frame:
            last_frame = frame

            new_data = data[frame]
            # print(f"\nFrame {frame + 1}, new_data: {new_data}")
            # print(f"Prior: {prior}")
            posterior = compute_posterior(new_data, prior)
            # print(f"Posterior: {posterior}")
            line.set_ydata(posterior)
            prior = posterior 
        
        title.set_text(f"Frame {frame + 1}")  # Update the existing title

        # Remove previous lines if they exist
        for artist in getattr(ax, "_bayes_artists", []):
            artist.remove()
        ax._bayes_artists = []

        # Calculate mean, median, and 1-sigma region for the current posterior
        mean = np.trapezoid(theta * posterior, theta)
        cdf = np.cumsum(posterior)
        cdf = cdf / cdf[-1]
        p16 = np.interp(0.16, cdf, theta)
        p50 = np.interp(0.50, cdf, theta)
        p84 = np.interp(0.84, cdf, theta)

        # Plot median
        median_line = ax.axvline(p50, color='g', linestyle='--', label='Median')
        # Plot 1-sigma region
        sigma_patch = ax.axvspan(p16, p84, color='orange', alpha=0.3, label='1-sigma')

        # Plot Beta function (analytical posterior) for current data slice
        wins = np.sum(data[:frame+1])
        losses = (frame + 1) - wins
        beta_pdf = beta.pdf(theta, wins + 1, losses + 1)
        beta_line, = ax.plot(theta, beta_pdf / np.trapezoid(beta_pdf, theta), 'b--', label='Beta posterior')

        # Add legend (only once)
        if not hasattr(ax, "_legend_added"):
            ax.legend()
            ax._legend_added = True

        # Store artists for removal in next frame
        ax._bayes_artists = [median_line, sigma_patch, beta_line]
        max_post = np.max(posterior)
        if max_post > ax.get_ylim()[1] * 0.95:
            ax.set_ylim(0, max_post * 1.1)

        return line, title


    def get_variable_frames(data):
        frames = []
        for i in range(len(data)):
            if i < 20:
                repeats = 10
            elif i < 100:
                repeats = 5
            else:
                repeats = 1
            frames.extend([i] * repeats)
        # print("frames:", frames)
        return frames

    def reset():
        nonlocal prior, last_frame
        prior = np.ones_like(theta) * 1 / len(theta)
        last_frame = -1
        line.set_ydata(prior)
        title.set_text("Frame 1")
        ax.set_ylim(0, 1)
        return line, title

    frames = get_variable_frames(data)

    ani = FuncAnimation(fig, update, frames=frames, blit=False, repeat=False, interval=50)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"check4_bayes_theorem_animation_{now}.mp4"
    ani.save(filename, writer='ffmpeg')
    reset()
    ani.save(filename.replace('.mp4', '.mov'), writer='ffmpeg')
    reset()
    print(f"Animation saved as {filename}")

    end_1_time = time.time()
    print(f"Time taken for animation: {end_1_time - start_time:.2f} seconds")
    plt.show()

    return prior

posterior = calculate_probability_by_updating_the_data(data, res=res)

print("Data:", data)
print("Probability of heads:", np.mean(data))


# Calculate mean and asymmetric 1-sigma (left and right) for the posterior
theta = np.arange(0, 1 + res, res)
mean = np.trapezoid(theta * posterior, theta)

# Compute the cumulative distribution function (CDF)
cdf = np.cumsum(posterior)
cdf = cdf / cdf[-1]

# Find the 16th, 50th, and 84th percentiles (for 1-sigma region)
p16 = np.interp(0.16, cdf, theta)
p50 = np.interp(0.50, cdf, theta)
p84 = np.interp(0.84, cdf, theta)

left_sigma = p50 - p16
right_sigma = p84 - p50

print(f"Posterior mean: {mean:.4f}")
print(f"Posterior median (p50): {p50:.4f}")
print(f"Left 1-sigma (p50 - p16): {left_sigma:.4f}")
print(f"Right 1-sigma (p84 - p50): {right_sigma:.4f}")

# Plot posterior with mean, median, and 1-sigma region
plt.figure(figsize=(8, 4))
plt.plot(theta, posterior, label="Posterior")
plt.axvline(mean, color='r', linestyle='--', label=f"Mean = {mean:.3f}")
plt.axvline(p50, color='g', linestyle='--', label=f"Median = {p50:.3f}")
plt.axvspan(p16, p84, color='orange', alpha=0.3, label="1-sigma region")
plt.xlabel("p")
plt.ylabel("Posterior Probability")
plt.title("Posterior Distribution with Asymmetric 1-sigma")
plt.text(
    p50, max(posterior)*0.95,
    f"Median = ${p50:.4f}^{{+{right_sigma:.4f}}}_{{-{left_sigma:.4f}}}$",
    color='g', ha='center', va='top'
)
plt.legend()
end_2_time = time.time()
print(f"Total Time taken: {end_2_time - end_time:.2f} seconds")
plt.show()


