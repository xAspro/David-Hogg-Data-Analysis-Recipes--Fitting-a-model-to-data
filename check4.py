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
from scipy.interpolate import interp1d

start_time = time.time()


p_head = 0.70  # Probability of head in a coin flip
n = 10  # Number of samples
res=0.005  # Resolution for the posterior calculation

x_arr = np.arange(0, 1 + res/100, res/100)

def make_sample_data(p_head, n=1000):
    p = [1 - p_head, p_head] 
    return np.random.choice(2, n, p=p)

data = make_sample_data(p_head, n)

def calculate_probability_using_batch_method(data):
    def log_prior(p):
        if 0 < p < 1:
            # return 0
            return np.log(1 / (p + res))
        else:
            return -np.inf
        
    def log_likelihood(p, data):
        return np.sum(data * np.log(p) + (1 - data) * np.log(1 - p))

    def log_posterior(p, data):
        lp = log_prior(p)
        if lp == -np.inf:
            return -np.inf

        return log_likelihood(p, data) + lp

    def mcmc(data, n_walkers=100, n_samples=1000, burn_in=100):
        sampler = emcee.EnsembleSampler(nwalkers=n_walkers, ndim=1, 
                                        log_prob_fn=log_posterior, 
                                        args=[data])

        p0 = np.random.uniform(0, 1, size=(n_walkers, 1))
        sampler.run_mcmc(p0, n_samples + burn_in, progress=True)

        # Get all samples (no discard)
        all_samples = sampler.get_chain()
        # Get samples after burn-in
        samples = sampler.get_chain(discard=burn_in)

        # Plot chains (no discard), each walker in a different color
        n_steps, n_walkers, ndim = all_samples.shape

        # plt.figure(figsize=(10, 4))
        # for i in range(n_walkers):
        #     plt.plot(all_samples[:, i, 0], alpha=0.7, label=f"Walker {i+1}" if n_walkers <= 10 else None)
        # if n_walkers <= 10:
        #     plt.legend()
        # plt.title("Chains (no discard)")
        # plt.xlabel("Step")
        # plt.ylabel("p")
        # plt.show()

        # # Plot corner (no discard)
        # try:
        #     corner.corner(all_samples.reshape(-1, all_samples.shape[-1]), labels=["p"], show_titles=True)
        #     plt.suptitle("Corner plot (no discard)")
        #     plt.show()
        # except ImportError:
        #     print("corner package not installed, skipping corner plot.")

        # # Plot chains (with burn-in discarded)
        # plt.figure(figsize=(10, 4))
        # for i in range(n_walkers):
        #     plt.plot(samples[:, i, 0], alpha=0.7, label=f"Walker {i+1}" if n_walkers <= 10 else None)
        # if n_walkers <= 10:
        #     plt.legend()
        # plt.title("chains after burn-in")
        # plt.xlabel("steps")
        # plt.ylabel("p")
        # plt.show()


        # # Plot corner (with burn-in discarded)
        # try:
        #     corner.corner(samples.reshape(-1, samples.shape[-1]), labels=["p"], show_titles=True)
        #     plt.suptitle("Corner plot (after burn-in)")
        #     plt.show()
        # except ImportError:
        #     print("corner package not installed, skipping corner plot.")

        return samples

    samples = mcmc(data)

    # Calculate and print autocorrelation time and acceptance rate
    try:
        tau = emcee.autocorr.integrated_time(samples, quiet=True)
        print(f"Autocorrelation time: {tau[0]:.2f}")
    except Exception as e:
        print(f"Could not compute autocorrelation time: {e}")

    acceptance_fraction = np.mean(samples)
    print(f"Mean acceptance rate: {np.mean(samples):.3f}")
    return samples


samples = calculate_probability_using_batch_method(data)

# Plot the posterior from MCMC and the analytical Beta posterior for comparison
plt.figure(figsize=(8, 4))
# Histogram of MCMC samples (posterior)
plt.hist(samples.reshape(-1), bins=50, density=True, alpha=0.6, label="MCMC Posterior")
# Analytical Beta posterior
wins = np.sum(data)
losses = len(data) - wins
theta_grid = np.linspace(0, 1, 1000)
beta_pdf = beta.pdf(theta_grid, wins + 1, losses + 1)
plt.plot(theta_grid, beta_pdf, 'r--', label=f"Beta({wins+1},{losses+1}) Posterior")
# Calculate median and 1-sigma for Beta posterior

beta_cdf = np.cumsum(beta_pdf)
beta_cdf = beta_cdf / beta_cdf[-1]
beta_p16 = np.interp(0.16, beta_cdf, theta_grid)
beta_p50 = np.interp(0.50, beta_cdf, theta_grid)
beta_p84 = np.interp(0.84, beta_cdf, theta_grid)
beta_left_sigma = beta_p50 - beta_p16
beta_right_sigma = beta_p84 - beta_p50
plt.axvline(beta_p50, color='purple', linestyle=':', label=f"Beta Median = {beta_p50:.3f}")
plt.axvspan(beta_p16, beta_p84, color='cyan', alpha=0.2, label="Beta 1-sigma region")

# Calculate median and 1-sigma for MCMC posterior
mcmc_samples = samples.reshape(-1)
mcmc_p16 = np.percentile(mcmc_samples, 16)
mcmc_p50 = np.percentile(mcmc_samples, 50)
mcmc_p84 = np.percentile(mcmc_samples, 84)
mcmc_left_sigma = mcmc_p50 - mcmc_p16
mcmc_right_sigma = mcmc_p84 - mcmc_p50
plt.axvline(mcmc_p50, color='g', linestyle='--', label=f"MCMC Median = {mcmc_p50:.3f}")
plt.axvspan(mcmc_p16, mcmc_p84, color='orange', alpha=0.3, label="MCMC 1-sigma region")
plt.xlabel("p")
plt.ylabel("Probability Density")
plt.title("Posterior Distribution: MCMC vs Beta Function")
plt.legend()
plt.savefig("check4_mcmc_vs_beta_posterior.png")
plt.show()

print()

def calculate_probability_by_updating_the_data(data, res=0.001):
    theta = np.arange(0, 1 + res, res)

    def set_prior():
        # prior = np.ones_like(theta) * 1 / len(theta)
        prior = 1 / (theta + res)
        # normalized_prior = prior / np.trapezoid(prior, theta)
        # print(f"Normalized prior: {normalized_prior}")
        # return normalized_prior
        return prior
    
    prior = set_prior()
    last_frame = -np.inf

    def compute_posterior(new_data, prior):
        likelihood = theta * new_data + (1 - theta) * (1 - new_data)
        unnormalized_posterior = prior * likelihood

        evidence = np.trapezoid(unnormalized_posterior, theta)
        posterior = unnormalized_posterior / evidence

        if evidence == 0:
            sys.exit("\nEvidence is zero, cannot normalize posterior.\n")

        return posterior

    fig, ax = plt.subplots()
    # print(f"Initial prior: {prior}")
    # print(f"Initial theta: {theta}")
    line, = ax.plot(theta, prior)
    line.set_ydata(prior)
    # ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("p")
    ax.set_ylabel("Posterior")
    title = ax.set_title("Frame 0")

    def update(frame):
        nonlocal prior, last_frame

        if frame > last_frame:
            last_frame = frame
            if frame == -1:
                prior = set_prior()
                posterior = prior
            else:
                posterior = prior
                new_data = data[frame]
                # print(f"\nFrame {frame + 1}, new_data: {new_data}")
                # print(f"Prior: {prior}")
                posterior = compute_posterior(new_data, prior)
                # print(f"Posterior: {posterior}")
                prior = posterior 

            ############################
            line.set_ydata(prior)
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
            beta_pdf = beta.pdf(x_arr, wins + 1, losses + 1)
            beta_line, = ax.plot(x_arr, beta_pdf / np.trapezoid(beta_pdf, x_arr), 'b--', label='Beta posterior')

            # Calculate median and 1-sigma for Beta posterior
            beta_cdf = np.cumsum(beta_pdf)
            beta_cdf = beta_cdf / beta_cdf[-1]
            beta_p16 = np.interp(0.16, beta_cdf, x_arr)
            beta_p50 = np.interp(0.50, beta_cdf, x_arr)
            beta_p84 = np.interp(0.84, beta_cdf, x_arr)
            # Plot median
            beta_median_line = ax.axvline(beta_p50, color='purple', linestyle=':', label='Beta Median')
            # Plot 1-sigma region
            beta_sigma_patch = ax.axvspan(beta_p16, beta_p84, color='cyan', alpha=0.2, label='Beta 1-sigma')

            # Add legend (only once)
            if not hasattr(ax, "_legend_added"):
                ax.legend()
                ax._legend_added = True

            # Store artists for removal in next frame
            ax._bayes_artists = [median_line, sigma_patch, beta_line, beta_median_line, beta_sigma_patch]

            max_post = np.max(posterior)
            if max_post > ax.get_ylim()[1] * 0.95:
                ax.set_ylim(0, max_post * 1.1)
            elif max_post < ax.get_ylim()[1] * 0.5:
                ax.set_ylim(0, max_post * 1.5)

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
        prior = set_prior()
        last_frame = -np.inf
        line.set_ydata(prior)
        title.set_text("Frame 0")
        ax.set_ylim(0, 1)
        return line, title

    frames = [-1] * 20 + get_variable_frames(data)

    ani = FuncAnimation(fig, update, frames=frames, blit=False, repeat=False, interval=50)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"check4_bayes_theorem_animation_{now}.mp4"
    ani.save(filename, writer='ffmpeg')
    reset()

    # ani.save(filename.replace('.mp4', '.mov'), writer='ffmpeg')
    # reset()
    # print(f"Animation saved as {filename}")

    end_1_time = time.time()
    print(f"Time taken for animation: {end_1_time - start_time:.2f} seconds")
    plt.show()

    return prior

posterior = calculate_probability_by_updating_the_data(data, res=res)

# print("Data:", data)
print(f"Fraction of heads: \t{np.sum(data)} / {len(data)} = {np.mean(data):.4f}")


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
plt.plot(x_arr, beta.pdf(x_arr, np.sum(data) + 1, len(data) - np.sum(data) + 1), 'r--', label="Beta Posterior")
plt.axvline(mean, color='r', linestyle='--', label=f"Mean = {mean:.3f}")
plt.axvline(p50, color='g', linestyle='--', label=f"Median = {p50:.3f}")
plt.axvspan(p16, p84, color='orange', alpha=0.3, label="1-sigma region")
# Calculate median and 1-sigma for Beta posterior
beta_pdf = beta.pdf(x_arr, np.sum(data) + 1, len(data) - np.sum(data) + 1)
beta_pdf /= np.trapezoid(beta_pdf, x_arr)
beta_cdf = np.cumsum(beta_pdf)
beta_cdf = beta_cdf / beta_cdf[-1]
beta_p16 = np.interp(0.16, beta_cdf, x_arr)
beta_p50 = np.interp(0.50, beta_cdf, x_arr)
beta_p84 = np.interp(0.84, beta_cdf, x_arr)
beta_left_sigma = beta_p50 - beta_p16
beta_right_sigma = beta_p84 - beta_p50

plt.axvline(beta_p50, color='purple', linestyle=':', label=f"Beta Median = {beta_p50:.3f}")
plt.axvspan(beta_p16, beta_p84, color='cyan', alpha=0.2, label="Beta 1-sigma region")
plt.text(
    beta_p50, max(beta_pdf)*0.35,
    f"Beta Median = ${beta_p50:.4f}^{{+{beta_right_sigma:.4f}}}_{{-{beta_left_sigma:.4f}}}$",
    color='purple', ha='center', va='bottom'
)
plt.xlabel("p")
plt.ylabel("Posterior Probability")
plt.title("Posterior Distribution with Asymmetric 1-sigma")
plt.text(
    p50, max(posterior)*0.65,
    f"Median = ${p50:.4f}^{{+{right_sigma:.4f}}}_{{-{left_sigma:.4f}}}$",
    color='g', ha='center', va='top'
)
plt.legend()
end_2_time = time.time()
print(f"Total Time taken: {end_2_time - start_time:.2f} seconds")
plt.savefig("check4_posterior_final.png")
plt.show()




# Compare the two methods: direct (updating) vs MCMC
plt.figure(figsize=(10, 4))

plt.plot(theta, posterior, label="Bayes Law Posterior", color='blue')
hist_vals, hist_bins = np.histogram(samples.reshape(-1), bins=theta, density=True)
hist_centers = 0.5 * (hist_bins[:-1] + hist_bins[1:])
# plt.plot(hist_centers, hist_vals, label="MCMC Posterior", color='orange', linestyle='--')
plt.hist(samples.reshape(-1), bins=theta, density=True, alpha=0.3, color='orange', label="MCMC Posterior Histogram")

plt.xlabel("p")
plt.ylabel("Posterior Probability Density")
plt.title("Comparison of Posterior Distributions")
plt.legend()
plt.tight_layout()
plt.savefig("check4_comparison_posteriors.png")
plt.show()

# Plot the residual between the two methods
# Interpolate MCMC histogram to theta grid for direct subtraction

mcmc_interp = interp1d(hist_centers, hist_vals, kind='linear', bounds_error=False, fill_value=0)
mcmc_on_theta = mcmc_interp(theta)
residual = posterior - mcmc_on_theta

plt.figure(figsize=(10, 4))
plt.plot(theta, residual, color='purple')
# Calculate and plot mean and variance of the residual
residual_mean = np.mean(residual)
residual_var = np.var(residual)
plt.axhline(residual_mean, color='red', linestyle='-', label=f"Mean = {residual_mean:.2e}")
plt.axhline(residual_mean + residual_var, color='green', linestyle=':', label=f"Mean + Var = {residual_mean + residual_var:.2e}")
plt.axhline(residual_mean - residual_var, color='green', linestyle=':', label=f"Mean - Var = {residual_mean - residual_var:.2e}")
plt.legend()
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("p")
plt.ylabel("Residual (Bayes Law - MCMC)")
plt.title("Residual Between Bayes Law and MCMC Posterior")
plt.tight_layout()
plt.savefig("check4_residual_posteriors.png")
plt.show()