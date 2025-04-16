"""
This file is to check if my understanding of this idea shown in David Hogg, is compatible with my project in hand.

In this code, initially, I will try to look at the change with sigy in segments. Later I will also think of what will happen if b itself is changed.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import emcee
import corner


def function(x, parameters):
    """
    Polynomial function in x.
    """
    return np.polyval(parameters, x)


def make_data(n_data_points, xmin, xmax, n_segments, sigy_min, sigy_max, parameters, log_scale=False):
    """
    Create data with a linear model and add noise.
    """
    x = np.linspace(xmin, xmax, n_data_points)
    NUM = len(parameters)

    random_indices = np.sort(np.random.choice(len(x) - 1, n_segments - 1, replace=False))

    # Add 0 and len(x) to the indices to define segment boundaries
    boundaries = np.concatenate(([0], random_indices + 1, [len(x)]))

    # Split the x array into segments
    segments = [x[boundaries[i]:boundaries[i+1]] for i in range(len(boundaries) - 1)]


    # Generate random sigy for each segment
    sigy = np.random.uniform(sigy_min, sigy_max, n_segments)

    # sigy = [0.5, 5, 1]
    sigyi = np.concatenate([np.full(len(segment), sigy_val) for segment, sigy_val in zip(segments, sigy)])
    print(f"sigy = {sigy}")
    print(f"sigy^2 = {sigy**2}")

    # # Print the segments
    # cnt = 0
    # for i, segment in enumerate(segments):
    #     print(f"Segment {i+1}: {segment}\n\tsigy = {sigy[i]}\n\tsigyi = ", end='')
    #     for j in range(len(segment)):
    #         print(f"{sigyi[cnt]:.2f}", end=' ')
    #         cnt += 1
    #     print()

    y = function(x, parameters)
    y += np.random.normal(0, sigyi, size=x.shape)

    plot_data(x, y, parameters, xmin, xmax, sigy, segments, filename='check1_orignal_data', log_scale=log_scale)
    return x, y, sigy, segments, NUM

def logprior(params, NUM):
    """
    Calculate the log prior probability of the parameters.
    """
    parameters = params[:NUM]
    sigyi2 = params[NUM:]

    # # Uniform prior for m and b
    # if not (-100 < m < 100 and -2000 < b < 2000):
    #     return -np.inf

    if not np.all((-10000 < parameters) & (parameters < 10000)):
        return -np.inf

    # Uniform prior for sigyi2
    if not np.all((0 < sigyi2) & (sigyi2 < 30000)):
        return -np.inf

    return 0.0

def loglikelihood(params, NUM, xi, yi):
    """
    Calculate the log likelihood of the data given the model parameters and noise parameters.
    """
    # Unpack the parameters
    parameters = params[:NUM]
    sigyi2 = params[NUM:]

    if len(sigyi2) != len(xi):
        import time
        time.sleep(2)
        raise ValueError("Length of sigyi2 must match length of xi")

    return np.sum(-0.5 * (((yi - (function(x, parameters)))**2 / sigyi2) + np.log(2 * np.pi * sigyi2)))

def logposterior_segments(params, NUM, xi, yi, segments):
    """
    Calculate the log posterior probability of the parameters given the data and segments.
    """
    lp = logprior(params, NUM)
    if not np.isfinite(lp):
        return -np.inf

    # Unpack the parameters
    parameters = params[:NUM]
    sigyi2 = params[NUM:]

    # Check if the number of segments matches the number of sigyi2 values
    if len(sigyi2) != len(segments):
        print(f"Number of segments: {len(segments)}, Number of sigyi2 values: {len(sigyi2)}")
        import sys
        sys.exit("Number of segments does not match number of sigyi2 values.")
        return -np.inf

    seg_lengths = [len(segment) for segment in segments]
    ll = 0.0
    lower_end = 0
    for i in range(len(seg_lengths)):
        upper_end = lower_end + seg_lengths[i]
        
        yi_segment = yi[lower_end:upper_end]
        sigyi2_segment = sigyi2[i]

        # Calculate the log likelihood for the current segment
        ll += np.sum(-0.5 * (((yi_segment - function(segments[i], parameters))**2 / sigyi2_segment) + np.log(2 * np.pi * sigyi2_segment)))

        # Update the lower end for the next segment
        lower_end = upper_end

    return lp + ll


def run_mcmc(x, y, NUM, segments, nwalkers=2000, n_burn=500, n_prod=600):
    """
    Run MCMC to fit the data.
    """
    N = len(segments)
    # Set up the initial position of the walkers
    p0 = np.random.rand(nwalkers, NUM + N)
    p0[:, :NUM] = np.random.uniform(-100, 100, (nwalkers, NUM))  # m, b
    p0[:, NUM:] = np.random.uniform(1, 3000, (nwalkers, N))  # sigyi2

    # Set up the MCMC sampler
    sampler = emcee.EnsembleSampler(nwalkers, NUM + N, logposterior_segments, args=(NUM, x, y, segments))

    # Run the burn-in phase
    sampler.run_mcmc(p0, n_burn, progress=True)

    # Reset the sampler and run the production phase
    sampler.reset()
    sampler.run_mcmc(None, n_prod, progress=True)

    return sampler

def plot_data(x, y, parameters, xmin, xmax, sigy, segments, filename=None, log_scale=False):
    

    x_dif = xmax - xmin
    y_dif = np.max(y) - np.min(y)

    # Create a 2D color plot for the Gaussian function with uncertainty for each segment
    X, Y = np.meshgrid(
        np.logspace(np.log10(xmin) - 0.25 * np.log10(x_dif), np.log10(xmax) + 0.25 * np.log10(x_dif), 500) if log_scale else np.linspace(xmin - 0.1 * x_dif, xmax + 0.1 * x_dif, 500),
        np.logspace(np.log10(np.min(y)) - 0.25 * np.log10(y_dif), np.log10(np.max(y)) + 0.25 * np.log10(y_dif), 500) if log_scale else np.linspace(np.min(y) - 0.1 * y_dif, np.max(y) + 0.1 * y_dif, 500)
    )

    # Check for NaN values in X or Y
    if np.isnan(X).any():
        print()
        print()
        print()
        print()
        print("X contains NaN values.")
        print()
        print()
        print()
    if np.isnan(Y).any():
        print()
        print()
        print()
        print()
        print("Y contains NaN values.")
        print()
        print()
        print()
    Z = np.zeros_like(X)


    # Calculate Gaussian deviation for each segment
    for i, segment in enumerate(segments):
        prev_end = xmin - 0.1 * x_dif if i == 0 else (segment[0] + segments[i-1][-1]) / 2
        next_start = xmax + 0.1 * x_dif if i == len(segments) - 1 else (segment[-1] + segments[i+1][0]) / 2

        mask = (X >= prev_end) & (X <= next_start)
        Z[mask] += np.exp(-0.5 * ((Y[mask] - function(X[mask], parameters)) / sigy[i])**2)

    x_arr = np.unique(X)

    # Normalize Z to range [0, 1]
    Z /= np.max(Z)

    # Define contour levels for 90%, 50%, and 10%
    contour_levels = [0.1, 0.5, 0.9]


    # Plot the color map
    plt.figure(figsize=(8, 6))
    if log_scale:
        contourf = plt.contourf(X, Y, Z, levels=100, cmap='viridis', alpha=0.8, norm=LogNorm())
    else:
        contourf = plt.contourf(X, Y, Z, levels=100, cmap='viridis', alpha=0.8)
    plt.colorbar(contourf, label='Gaussian Value')


    # Add contours for specific levels
    contours = plt.contour(X, Y, Z, levels=contour_levels, colors=['red', 'blue', 'white'], linewidths=1.5)
    plt.clabel(contours, inline=True, fontsize=6, fmt={0.1: '10%', 0.5: '50%', 0.9: '90%'})

    if log_scale:
        plt.loglog(x, y, 'o', c='red', label='Data Points', markeredgecolor='black')
        plt.loglog(x_arr, function(x_arr, parameters), color='black', label='Original Line')
    else:
        plt.scatter(x, y, c='red', label='Data Points', edgecolor='black')
        plt.plot(x_arr, function(x_arr, parameters), color='black', label='Original Line')


    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Gaussian Function with Uncertainty for Each Segment')
    plt.legend()
    if log_scale:
        filename = filename + "_log"
    if filename:
        plt.savefig(filename + ".pdf", bbox_inches='tight')
        plt.savefig(filename + ".png", bbox_inches='tight')
    plt.show()


# make_data(19, 2, 3, 1, 10, 5, 1, 10)

# x, y, m, b, sigy, segments = make_data(9999, 2, 3, 1, 10, 7, 1, 4)

parameters = (2, 3, 100)
log_scale = False
log_scale = True

x, y, sigy, segments, NUM = make_data(999, 1, 10, 5, 1, 50, parameters, log_scale=log_scale)

sampler = run_mcmc(x, y, NUM, segments)

samples = sampler.get_chain(flat=True)


param_names = [f"parameter_{i+1}" for i in range(NUM)] 
for i in range(len(param_names)):
    print(f'parameter_{i+1} = {parameters[i]:.2f}')

# Extract the samples for each parameter
param_names = param_names + [f"sigyi2_{i+1}" for i in range(len(segments))]
results = {}

for i, param in enumerate(param_names):
    # Get the samples for the current parameter
    param_samples = samples[:, i]
    
    # Compute the 16th, 50th (median), and 84th percentiles
    q16, q50, q84 = np.percentile(param_samples, [16, 50, 84])
    
    # Compute the uncertainties
    err_minus = q50 - q16
    err_plus = q84 - q50
    
    # Store the results
    results[param] = {
        "median": q50,
        "err_minus": err_minus,
        "err_plus": err_plus
    }

    # Print the result in the format used by corner
    print(f"{param} = {q50:.2f} +{err_plus:.2f} -{err_minus:.2f}")

# Plot the results
labels = [f"parameter_{i+1}" for i in range(NUM)] + [f"sigyi2_{i+1}" for i in range(len(segments))]
fig = plt.figure(figsize=(8, 8), dpi=100, tight_layout=True)
corner.corner(samples, labels=labels, fig=fig, show_titles=True, bins=30)
plt.savefig('check1_corner.png', bbox_inches='tight')
plt.savefig('check1_corner.pdf', bbox_inches='tight')
plt.show()

# Plot the chains for each parameter
fig, axes = plt.subplots(len(param_names), figsize=(10, 2 * len(param_names)), sharex=True)
for i, param in enumerate(param_names):
    ax = axes[i]
    ax.plot(sampler.get_chain()[:, :, i], alpha=0.5)
    ax.set_ylabel(param)
    ax.set_xlabel("Step")
    ax.set_title(f"Chain for {param}")
plt.tight_layout()
plt.savefig('check1_chains.png', bbox_inches='tight')
plt.savefig('check1_chains.pdf', bbox_inches='tight')
plt.show()

# Extract the best-fit parameters (mean of the posterior samples)
best_fit_params = np.mean(samples, axis=0)
parameters_fit = best_fit_params[:NUM]

# Plot the data with the currently estimated line
plot_data(x, y, parameters_fit, xmin=1, xmax=10, sigy=sigy, segments=segments, filename='check1_guessed_soln', log_scale=log_scale)