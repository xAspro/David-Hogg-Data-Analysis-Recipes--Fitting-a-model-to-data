"""
To check what is the Bayesian posterior distribution of the parameters when 
x, y, and sigyi of good points are given

"""



import matplotlib.pyplot as plt
import numpy as np
import emcee


# Data
id = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x = np.array([201, 244, 47, 287, 203, 58, 210, 202, 198, 158, 165, 201, 157, 131, 166, 160, 186, 125, 218, 146])
y = np.array([592, 401, 583, 402, 495, 173, 479, 504, 510, 416, 393, 442, 317, 311, 400, 337, 423, 334, 533, 344])
sigy = np.array([61, 25, 38, 15, 21, 15, 27, 14, 30, 16, 14, 25, 52, 16, 34, 31, 42, 26, 16, 22])
sigx = np.array([9, 4, 11, 7, 5, 9, 4, 4, 11, 7, 5, 5, 5, 6, 6, 5, 9, 8, 6, 5])
rhoxy = np.array([-0.84, 0.31, 0.64, -0.27, -0.33, 0.67, -0.02, -0.05, -0.84, -0.69, 0.30, -0.46, -0.03, 0.50, 0.73, -0.52, 0.90, 0.40, -0.78, -0.56])


x = x[4:]
y = y[4:]
sigy = sigy[4:]
N = len(x)

def logprior(params):
    m, b = params
    if 0 <= m <= 5 and -200 <= b <= 200:
        return 0
    return -np.inf  # Reject everything else


# def likelihood(xi, yi, sigyi, m, b, Pb, Yb, Vb):
def loglikelihood(params, xi, yi, sigyi):
    """
    Calculate the log likelihood of the data given the model parameters and noise parameters.
    The likelihood is calculated using the formula:
    Li = (1 - Pb) / sqrt(sigyi**2) * exp(-0.5 * ((yi - (m * xi + b)) / sigyi)**2) + Pb / sqrt(Vb + sigyi**2) * exp(-0.5 * ((yi - Yb)**2 / (Vb + sigyi**2)))
    """
    # Unpack the parameters
    m, b = params
    return np.sum(-0.5 * (((yi - (m * xi + b))**2 / sigyi**2))) 

def logposterior(params, xi, yi, sigyi):
    lp = logprior(params)
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + loglikelihood(params, xi, yi, sigyi) 


def run_mcmc(xi, yi, sigyi, nwalkers=100, nsteps_burn=2000, nsteps_prod=1000):
    """
    Run the MCMC simulation using emcee.
    """
    # Define the number of dimensions
    # m, b, Pb, Yb, Vb
    ndim = 2

    p0 = np.empty((nwalkers, ndim))

    p0[:, 0] = np.random.uniform(0, 2, size=nwalkers)   # m
    p0[:, 1] = np.random.uniform(0, 200, size=nwalkers) # b

    # Create the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logposterior, args=(xi, yi, sigyi))

    # Run the MCMC simulation
    sampler.run_mcmc(p0, nsteps_burn, progress=True)
    sampler.reset()  # clear burn-in

    # Production
    sampler.run_mcmc(None, nsteps_prod, progress=True)

    return sampler

################################################################################################################################################
# Marginalisation is not required. Because even histogram is compressing the joint probability distribution into just the required parameters.
################################################################################################################################################

# def marginalisation(samples, params_to_marginalise=[2, 3, 4], target_params=[0, 1], n_bins=100):
#     """
#     Marginalise the samples over the specified parameters and calculate the MAP value for the target parameters.
#     """
#     # Check if params_to_marginalise is a list or a single parameter
#     if not isinstance(params_to_marginalise, list):
#         params_to_marginalise = [params_to_marginalise]

#     # Marginalize over the specified parameters (sum over them)
#     marginalised_samples = np.sum(samples[:, params_to_marginalise], axis=1)

#     # Create a new array with the remaining parameters (target_params)
#     reduced_samples = samples[:, target_params]

#     # Create a histogram of the reduced samples
#     hist, edges = np.histogram(reduced_samples[:, 0], bins=n_bins, density=True)

#     # Calculate the bin centers
#     bin_centers = 0.5 * (edges[1:] + edges[:-1])

#     # Normalize the histogram
#     hist /= np.sum(hist)

#     # Find the MAP value (bin center with the highest density)
#     map_index = np.argmax(hist)  # Index of the maximum value in the histogram
#     map_value = bin_centers[map_index]  # Corresponding bin center

#     return bin_centers, hist, map_value

# def plot_marginalisation(samples, params_to_marginalise=[2,3,4], target_params=[0,1], n_bins=100):
#     """
#     Plot the marginalisation of the samples over the specified parameters.
#     """
#     bin_centers, hist, map_value = marginalisation(samples, params_to_marginalise, target_params, n_bins)


#     print(f"MAP value for target parameter {target_params}: {map_value}")
#     print()
    
#     plt.figure(figsize=(8, 5))
#     plt.plot(bin_centers, hist, label=f"Marginalisation over parameter {params_to_marginalise}")
#     plt.axvline(map_value, color='red', linestyle='--', label=f"MAP = {map_value:.2f}")
#     plt.xlabel(f"Parameter {params_to_marginalise}")
#     plt.ylabel("Density")
#     plt.title(f"Marginalisation of Parameter {params_to_marginalise}")
#     plt.legend()
#     plt.show()

def plot_results(samples):
    """
    Plot the results of the MCMC simulation.
    """

    H, xedges, yedges = np.histogram2d(samples[:,1], samples[:,0], bins=500)
    i,j = np.unravel_index(np.argmax(H), H.shape)
    b_map = 0.5*(xedges[i]+xedges[i+1])
    m_map = 0.5*(yedges[j]+yedges[j+1])

    print("MAP of m:", m_map)
    print("MAP of b:", b_map)
    print()

    H_normalized = H / np.max(H)
    plt.pcolormesh(xedges, yedges, H_normalized.T, cmap="Greys")

    H_flat = H.flatten()
    H_sorted = np.sort(H_flat)
    cumsum = np.cumsum(H_sorted)
    cumsum /= cumsum[-1]  # Normalize to [0, 1]

    # Define percentiles (e.g., 68%, 95%, 99%)
    levels = [0.25, 0.5, 0.75]
    contour_levels = sorted(set([H_sorted[np.searchsorted(cumsum, level)] for level in levels]))

    # Create a meshgrid for contour plotting
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    plt.contour(X, Y, H.T, levels=contour_levels, colors="black")

    # Add labels and title
    plt.xlabel("b")
    plt.ylabel("m")
    plt.xlim(-125, 125)
    plt.ylim(1.5, 3.1)
    plt.title("2D Histogram with Contours and Density Shading")
    plt.colorbar(label="Normalized Density")
    # # plt.savefig('Exercise6_histogram.png', bbox_inches='tight')
    # # plt.savefig('Exercise6_histogram.pdf', bbox_inches='tight')
    plt.show()
    return samples



def plot_chains(sampler):
    fig, axes = plt.subplots(2, figsize=(8, 5), sharex=True)
    samples = sampler.get_chain()

    labels = ["m", "b"]
    for i in range(2):  # For each parameter
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("Step number")
    plt.show()


def plot_fit_with_samples(x, y, sigy, samples, n_samples_to_plot=100):
    """
    Plot the data with error bars, best-fit line, and sample lines from MCMC.
    """
    # Plot data with error bars
    plt.errorbar(x, y, yerr=sigy, fmt='o', color='red', markersize=4, label="Data", capsize=3, capthick=1)

    # Choose some sample lines to show the uncertainty
    x_plot = np.linspace(0, 300, 300)
    for i in np.random.choice(len(samples), size=n_samples_to_plot, replace=False):
        m, b = samples[i]
        y_sample = m * x_plot + b
        plt.plot(x_plot, y_sample, color='gray', alpha=0.1)

    # # Plot the best-fit line (mean or MAP)
    # m_best = np.mean(samples[:, 0])  # or use MAP
    # b_best = np.mean(samples[:, 1])
    # y_best = m_best * x_plot + b_best

    # Plot the best-fit line (MAP)
    H, xedges, yedges = np.histogram2d(samples[:,1], samples[:,0], bins=500)
    i,j = np.unravel_index(np.argmax(H), H.shape)
    b_map = 0.5*(xedges[i]+xedges[i+1])
    m_map = 0.5*(yedges[j]+yedges[j+1])
    y_best = m_map * x_plot + b_map

    plt.plot(x_plot, y_best, color='blue', label='Best Fit (MAP)')

    # Labels and legend
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Line fit with MCMC uncertainty")
    plt.legend()
    # # plt.savefig('Exercise6_fit.png', bbox_inches='tight')
    # # plt.savefig('Exercise6_fit.pdf', bbox_inches='tight')
    plt.show()



def main():
    # Run the MCMC simulation

    sampler = run_mcmc(x, y, sigy)

    samples = sampler.get_chain(flat=True)

    # Plot the results
    
    # Plot the results
    plot_results(samples)
    # import sys
    # sys.exit()

    # Print the results
    print("Mean of m:", np.mean(samples[:, 0]))
    print("Mean of b:", np.mean(samples[:, 1]))

    print()

    print("Median of m:", np.median(samples[:, 0]))
    print("Median of b:", np.median(samples[:, 1]))

    print()


    # import sys
    # sys.exit()

    import corner



    # corner.corner(samples, labels=["m", "b"], truths=[m, c], bins=50, smooth=1.0, show_titles=True)
    # corner.corner(samples[mask], labels=["m", "b", "Pb", "Yb", "Vb"], quantiles=[0.16, 0.5, 0.84], bins=250, fig=plt.figure(figsize=(12, 7)))
    corner.corner(samples, labels=["m", "b"], quantiles=[0.16, 0.5, 0.84], bins=250, fig=plt.figure(figsize=(12, 7)), show_titles=True)
    # # plt.savefig('Exercise6_corner.png', bbox_inches='tight')
    # # plt.savefig('Exercise6_corner.pdf', bbox_inches='tight')
    plt.show()

    plot_chains(sampler)
    plot_fit_with_samples(x, y, sigy, samples)



if __name__ == "__main__":
    main()
