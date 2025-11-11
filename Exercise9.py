"""
Reproducing the Figure 6 in David Hogg's paper "Data analysis recipes: Fitting a model to data"

Comparing the Mixture model with only the good data points and its uncertainties and 
the same with its uncertainties halved.

This shows how important the uncertainties. Even those are important and integral part of the data.
However, that doesnt mean that increasing the uncertainties arbitrarily will give better results.
Better result depends on how well we understand the data, not better looking plot!
In case, the orignal uncertainties were small and increasing it gives us 'better results' 
(the exact opposite scenario of what we have in this exercise), then also we cant arbitrarily change the data.
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

def logprior(params):
    m, b, Pb, Yb, Vb = params
    if 0 <= Pb <= 1 and Vb > 0:
        if 0 <= m <= 5 and -200 <= b <= 200 and 0 <= Yb <= 1000:
            return -np.log(1 + Pb) - np.log(1 + Vb)  # Prior for m, b, Pb, Yb, Vb
    return -np.inf  # Reject everything else


# def likelihood(xi, yi, sigyi, m, b, Pb, Yb, Vb):
def loglikelihood(params, xi, yi, sigyi):
    """
    Calculate the log likelihood of the data given the model parameters and noise parameters.
    The likelihood is calculated using the formula:
    Li = (1 - Pb) / sqrt(sigyi**2) * exp(-0.5 * ((yi - (m * xi + b)) / sigyi)**2) + Pb / sqrt(Vb + sigyi**2) * exp(-0.5 * ((yi - Yb)**2 / (Vb + sigyi**2)))
    """
    # Unpack the parameters
    m, b, Pb, Yb, Vb = params
    # Check if Pb is between 0 and 1
    if Pb < 0 or Pb > 1:
        # print("Pb is not between 0 and 1")
        return -np.inf
        # return 0
    # Check if Vb is positive
    if Vb <= 0:
        # print("Vb is not positive")
        return -np.inf
        # return 0
    # Check if sigyi is positive
    if np.any(sigyi <= 0):
        print("sigyi is not positive")
        return -np.inf
        # return 0
    # Calculate the likelihood
    return np.sum(np.log((1 - Pb) / np.sqrt(sigyi**2) * np.exp(-0.5 * ((yi - (m * xi + b)) / sigyi)**2) + Pb / np.sqrt(Vb + sigyi**2) * np.exp(-0.5 * ((yi - Yb)**2 / (Vb + sigyi**2)))))

def logposterior(params, xi, yi, sigyi):
    lp = logprior(params)
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + loglikelihood(params, xi, yi, sigyi) 


def run_mcmc(xi, yi, sigyi, nwalkers=2000, nsteps_burn=200, nsteps_prod=1000):
    """
    Run the MCMC simulation using emcee.
    """
    # Define the number of dimensions
    # m, b, Pb, Yb, Vb
    ndim = 5

    p0 = np.empty((nwalkers, 5))

    p0[:, 0] = np.random.uniform(0, 2, size=nwalkers)   # m
    p0[:, 1] = np.random.uniform(0, 200, size=nwalkers) # b
    p0[:, 2] = np.random.uniform(0, 0.1, size=nwalkers)    # Pb
    p0[:, 3] = np.random.uniform(0, 200, size=nwalkers) # Yb
    p0[:, 4] = np.random.uniform(0, 100, size=nwalkers)   # Vb

    # Create the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logposterior, args=(xi, yi, sigyi))

    # Run the MCMC simulation
    sampler.run_mcmc(p0, nsteps_burn, progress=True)
    sampler.reset()  # clear burn-in

    # Production
    sampler.run_mcmc(None, nsteps_prod, progress=True)

    return sampler


def plot_results(part, samples):
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
    levels = [0.10, 0.25, 0.5, 0.75]
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
    if part == 'a':
        str = 'Using correct data uncertainties'
    else:
        str = 'Using data uncertainties / 2'
    plt.text(0.5, 0.9, str, transform=plt.gca().transAxes, fontsize=12, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    plt.colorbar(label="Normalized Density")
    plt.savefig(f'Exercise9_{part}_histogram.png', bbox_inches='tight')
    plt.savefig(f'Exercise9_{part}_histogram.pdf', bbox_inches='tight')
    plt.show()
    return samples



def plot_chains(sampler):
    fig, axes = plt.subplots(5, figsize=(8, 5), sharex=True)
    samples = sampler.get_chain()

    labels = ["m", "b", "Pb", "Yb", "Vb"]
    for i in range(5):  # For each parameter
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("Step number")
    plt.show()


def plot_fit_with_samples(x, y, sigy, part, samples, n_samples_to_plot=10):
    """
    Plot the data with error bars, best-fit line, and sample lines from MCMC.
    """
    # Plot data with error bars
    plt.errorbar(x, y, yerr=sigy, fmt='o', color='red', markersize=4, label="Data", capsize=3, capthick=1)

    # Choose some sample lines to show the uncertainty
    x_plot = np.linspace(min(x), max(x), 200)
    for i in np.random.choice(len(samples), size=n_samples_to_plot, replace=False):
        m, b, Pb, Yb, Vb = samples[i]
        y_sample = m * x_plot + b
        plt.plot(x_plot, y_sample, color='gray', alpha=0.1)

    # Plot the best-fit line (mean or MAP)
    m_best = np.mean(samples[:, 0])  # or use MAP
    b_best = np.mean(samples[:, 1])
    y_best = m_best * x_plot + b_best
    plt.plot(x_plot, y_best, color='blue', label='Best Fit (Mean)')

    # Labels and legend
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Line fit with MCMC uncertainty")
    plt.legend()
    plt.savefig(f'Exercise9_{part}_fit.png', bbox_inches='tight')
    plt.savefig(f'Exercise9_{part}_fit.pdf', bbox_inches='tight')
    plt.show()



def run_calculation(x, y, sigy, part):
    sampler = run_mcmc(x, y, sigy)
    samples = sampler.get_chain(flat=True)

    plot_results(part, samples)

    import corner
    Vb = samples[:, 4]

    if part == 'a':
        lower, upper = 0, 200
        
    else:
        mark = 10

        lower, upper = np.percentile(Vb, [0, 100 - mark])
        

    ranges = [
    (np.min(samples[:, 0]), np.max(samples[:, 0])),  # m
    (np.min(samples[:, 1]), np.max(samples[:, 1])),  # b
    (np.min(samples[:, 2]), np.max(samples[:, 2])),  # Pb
    (np.min(samples[:, 3]), np.max(samples[:, 3])),  # Yb
    (lower, upper)                                   # Vb
    ]

    corner.corner(samples, labels=["m", "b", "Pb", "Yb", "Vb"], range=ranges, quantiles=[0.16, 0.5, 0.84], bins=250, fig=plt.figure(figsize=(12, 7)), show_titles=True)
    plt.savefig(f'Exercise9_{part}_corner.png', bbox_inches='tight')
    plt.savefig(f'Exercise9_{part}_corner.pdf', bbox_inches='tight')
    plt.show()

    # print("Exiting the program...")
    # import sys
    # sys.exit()

    plot_chains(sampler)
    plot_fit_with_samples(x, y, sigy, part, samples)

def main():
    run_calculation(x, y, sigy, "a")
    run_calculation(x, y, sigy / 2, "b")


if __name__ == "__main__":
    main()
