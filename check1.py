"""
This file is to check if my understanding of this idea shown in David Hogg, is compatible with my project in hand.

In this code, initially, I will try to look at the change with sigy in segments. Later I will also think of what will happen if b itself is changed.
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner


# def function(x, parameters):
#     """
#     Polynomial function in x.
#     """
#     return np.polyval(parameters, x)

def function(M, parameters):
    log_phi_star = parameters[0]
    M_star = parameters[1]
    alpha = parameters[2]
    beta = parameters[3]

    # print(f"phi_star = {phi_star}")
    # print(f"M_star = {M_star}")
    # print(f"alpha = {alpha}")
    # print(f"beta = {beta}")
    # print(f'M = {M}')
    # print()

    # import sys
    # sys.exit("Check the function")

    # Debugging: Print intermediate values
    term1 = 10**(0.4 * (alpha + 1) * (M - M_star))
    term2 = 10**(0.4 * (beta + 1) * (M - M_star))
    # print(f"M = {M}, term1 = {term1}, term2 = {term2}")

    # # Check for overflow
    # if np.isinf(term1).any() or np.isinf(term2).any():
    #     print("Overflow detected in term1 or term2!")

    #     if np.isinf(term1).any():
    #         print("term1 overflowed")
    #         print(f'alpha = {alpha}, M = {M}, M_star = {M_star}')
    #         print(f'alpha + 1 = {alpha + 1}')
    #         print(f'M - M_star = {M - M_star}')
    #         print(f'(0.4 * (alpha + 1) * (M - M_star)) = {0.4 * (alpha + 1) * (M - M_star)}')
    #         print(f'10**(0.4 * (alpha + 1) * (M - M_star)) = {term1}')

    #     if np.isinf(term2).any():
    #         print("term2 overflowed")
    #         print(f'beta = {beta}, M = {M}, M_star = {M_star}')
    #         print(f'beta + 1 = {beta + 1}')
    #         print(f'M - M_star = {M - M_star}')
    #         print(f'(0.4 * (beta + 1) * (M - M_star)) = {0.4 * (beta + 1) * (M - M_star)}')
    #         print(f'10**(0.4 * (beta + 1) * (M - M_star)) = {term2}')

    #     import sys
    #     sys.exit("Overflow detected in term1 or term2!")



    return log_phi_star - np.log10((10**(0.4 * (alpha + 1) * (M- M_star)) + 10**(0.4 * (beta + 1) * (M - M_star))))


def make_data(n_data_points, xmin, xmax, n_segments, sigy_min, sigy_max, parameters):
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

    plot_data(x, y, parameters, xmin, xmax, sigy, segments, filename='check1_orignal_data')
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

    # if parameters[0] < 0 or parameters[0] > 1:
    #     return -np.inf

    # if parameters[0] >= 0:
    #     return -np.inf

    # if not np.all((-50 < parameters) & (parameters < 1)):
    #     return -np.inf

    min = [-14.0, -32.0, -7.0, -10.0]
    max = [-1.0, -15.0, 0.0, 15.0]

    # # Get the current time in milliseconds
    # import time
    # current_time = time.time()
    # milliseconds = int((current_time * 100000) % 100000)

    # # Check if the millisecond component is exactly 0
    # if milliseconds == 0:
    #     print(f"min = {min}")
    #     print(f"max = {max}")
    #     print(f"parameters = {parameters}")

    if not np.all((min < parameters) & (parameters < max)):
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

    sum = np.sum(np-0.5 * (((yi - (function(x, parameters)))**2 / sigyi2) + np.log(2 * np.pi * sigyi2)))

    if np.isfinite(sum):
        return sum
    else:
        print("Log likelihood is not finite")
        print(f"yi = {yi}")
        print(f"xi = {xi}")
        print(f"parameters = {parameters}")
        print(f"sigyi2 = {sigyi2}")
        import sys
        sys.exit("Log likelihood is not finite")
        return -np.inf

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

    return (lp + ll) / np.log10(np.e)


def run_mcmc(x, y, NUM, segments, nwalkers=2000, n_burn=10, n_prod=2000):
    """
    Run MCMC to fit the data.
    """
    N = len(segments)


    min = [-14.0, -32.0, -7.0, -10.0]
    max = [-1.0, -15.0, 0.0, 15.0]
    # Set up the initial position of the walkers
    p0 = np.random.rand(nwalkers, NUM + N)
    # p0[:, :NUM] = np.random.uniform(-20, 0, (nwalkers, NUM)) 
    p0[:, :NUM] = np.random.uniform(min, max, (nwalkers, NUM))  # parameters
    p0[:, NUM:] = np.random.uniform(1, 3000, (nwalkers, N))  # sigyi2

    # Set up the MCMC sampler
    sampler = emcee.EnsembleSampler(nwalkers, NUM + N, logposterior_segments, args=(NUM, x, y, segments))

    # Run the burn-in phase
    sampler.run_mcmc(p0, n_burn, progress=True)

    # Reset the sampler and run the production phase
    sampler.reset()
    sampler.run_mcmc(None, n_prod, progress=True)

    return sampler

def find_MAP(sampler, N, NUM):
    """
    Find the maximum a posteriori (MAP) estimate of the parameters, 
    after marginalising over the variance parameters.
    """

    print("In find_MAP")
    samples = sampler.get_chain(flat=True)
    print(f"Number of samples: {len(samples)}")

    # Extract the first 4 parameters from the samples
    samples_reduced = samples[:, :4]

    # Define the number of bins for the histogram
    bin_count = 50
    bins = [bin_count] * 4  # Only for the first 4 dimensions

    # Compute the 4D histogram for the first 4 parameters
    hist, edges = np.histogramdd(samples_reduced, bins=bins)

    # Find the index of the maximum value in the marginal distribution
    max_index = np.unravel_index(np.argmax(hist), hist.shape)
    print("Index of maximum value in the marginal distribution:")
    print(max_index)

    # Get the corresponding parameter values
    param_values = [edges[i][max_index[i]] for i in range(len(max_index))]
    print("Parameter values corresponding to the maximum value:")
    print(param_values)
    # Return the parameter values
    return param_values


def plot_data(x, y, parameters, xmin, xmax, sigy, segments, filename=None):
    print("Inside plot_data")

    x_dif = xmax - xmin
    y_dif = np.max(y) - np.min(y)

    print(f'xmin = {xmin}, xmax = {xmax}')
    print(f'np.min(y) = {np.min(y)}, np.max(y) = {np.max(y)}')
    print(f'x_dif = {x_dif}')
    print(f'y_dif = {y_dif}')
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    # import time
    # time.sleep(10)

    bin_count = 10

    # Create a 2D color plot for the Gaussian function with uncertainty for each segment
    X, Y = np.meshgrid(np.linspace(xmin - 0.1 * x_dif, xmax + 0.1 * x_dif, bin_count), np.linspace(np.min(y) - 0.1 * y_dif, np.max(y) + 0.1 * y_dif, bin_count))
    Z = np.zeros_like(X)


    # Calculate Gaussian deviation for each segment
    for i, segment in enumerate(segments):
        prev_end = xmin - 0.1 * x_dif if i == 0 else (segment[0] + segments[i-1][-1]) / 2
        next_start = xmax + 0.1 * x_dif if i == len(segments) - 1 else (segment[-1] + segments[i+1][0]) / 2

        # print(f"prev_end = {prev_end}")
        # print(f"next_start = {next_start}\n")

        # Create a mask for the smoothed range
        mask = (X >= prev_end) & (X <= next_start)
        Z[mask] += np.exp(-0.5 * ((Y[mask] - function(X[mask], parameters)) / sigy[i])**2)

    x_arr = np.unique(X)

    # Normalize Z to range [0, 1]
    Z /= np.max(Z)

    # Define contour levels for 90%, 50%, and 10%
    contour_levels = [0.1, 0.5, 0.9]


    # Plot the color map
    plt.figure(figsize=(8, 6))
    contourf = plt.contourf(X, Y, Z, levels=100, cmap='viridis', alpha=0.8)
    plt.colorbar(contourf, label='Gaussian Value')

    # Add contours for specific levels
    contours = plt.contour(X, Y, Z, levels=contour_levels, colors=['red', 'blue', 'white'], linewidths=1.5)
    plt.clabel(contours, inline=True, fontsize=6, fmt={0.1: '10%', 0.5: '50%', 0.9: '90%'})




    # contour = plt.contourf(X, Y, Z, levels=100, cmap='viridis', alpha=0.8)
    # plt.colorbar(contour, label='Gaussian Value')
    plt.scatter(x, y, c='red', label='Data Points', edgecolor='black')
    plt.plot(x_arr, function(x_arr, parameters), color='black', label='Original Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Gaussian Function with Uncertainty for Each Segment')
    plt.legend()
    plt.gca().invert_xaxis()
    if filename:
        plt.savefig(filename + ".pdf", bbox_inches='tight')
        plt.savefig(filename + ".png", bbox_inches='tight')
    plt.show(block=False)
    print(f"Plot saved as {filename}.pdf and {filename}.png")


# parameters = (2, 3, 100)
parameters = (-6.77, -26.48, -4.72, -1.70)
xrange = (-31, -19)

try:
    x, y, sigy, segments, NUM = make_data(999, *xrange, 3, 1, 5, parameters)

    sampler = run_mcmc(x, y, NUM, segments)
except Exception as e:
    import traceback
    print("An error occurred during execution:")
    print(traceback.format_exc())
    print("Stopping execution and exiting.")
    raise SystemExit("Execution terminated due to an error.")

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
labels = param_names
fig = plt.figure(figsize=(8, 8), dpi=100, tight_layout=True)
corner.corner(samples, labels=labels, fig=fig, show_titles=True, bins=30)
plt.savefig('check1_corner.png', bbox_inches='tight')
plt.savefig('check1_corner.pdf', bbox_inches='tight')
plt.show(block=False)
print("Corner plot saved as check1_corner.png and check1_corner.pdf")

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
plt.show(block=False)
print("Chains plot saved as check1_chains.png and check1_chains.pdf")

# Extract the best-fit parameters (mean of the posterior samples)
parameters_fit = find_MAP(sampler, len(segments), NUM)
plot_data(x, y, parameters_fit, *xrange, sigy=sigy, segments=segments, filename='check1_guessed_soln')