"""
Reproducing the Figure 4 in David Hogg's paper "Data analysis recipes: Fitting a model to data"

Considering all the data points for a linear fit

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

def plot_results(samples, question_part):
    """
    Plot the results of the MCMC simulation.
    """

    H, xedges = np.histogram(samples[:,2], bins=500)


    # Add labels and title
    plt.xlabel("Pb")
    plt.ylabel("Marginalised Posterior Distribution Function")
    # plt.xlim(0, 1)
    # plt.ylim(0, 4)
    plt.title("Marginalised Posterior Distribution Function of Pb")
    if question_part == 'a':
        str = 'Using correct data uncertainities'
    else:
        str = 'Using data uncertainities / 2'

    bin_centers = 0.5 * (xedges[1:] + xedges[:-1])
    plt.plot(bin_centers, H, label=str)
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


def plot_fit_with_samples(x, y, sigy, samples, n_samples_to_plot=10):
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
    # plt.savefig('Exercise6_fit.png', bbox_inches='tight')
    # plt.savefig('Exercise6_fit.pdf', bbox_inches='tight')
    plt.show()



def main():
    # Run the MCMC simulation

    sampler = run_mcmc(x, y, sigy)

    samples = sampler.get_chain(flat=True)

    sampler2 = run_mcmc(x, y, sigy / 2)

    samples2 = sampler2.get_chain(flat=True)

    # Plot the results
    
    # Plot the results
    plot_results(samples, 'a')
    plot_results(samples2, 'b')

    plt.legend()
    plt.savefig('Exercise7.png', bbox_inches='tight')
    plt.savefig('Exercise7.pdf', bbox_inches='tight')
    plt.show()
    # import sys
    # sys.exit()

    # # Print the results
    # print("Mean of m:", np.mean(samples[:, 0]))
    # print("Mean of b:", np.mean(samples[:, 1]))
    # print("Mean of Pb:", np.mean(samples[:, 2]))
    # print("Mean of Yb:", np.mean(samples[:, 3]))
    # print("Mean of Vb:", np.mean(samples[:, 4]))

    # print()

    # print("Median of m:", np.median(samples[:, 0]))
    # print("Median of b:", np.median(samples[:, 1]))
    # print("Median of Pb:", np.median(samples[:, 2]))
    # print("Median of Yb:", np.median(samples[:, 3]))
    # print("Median of Vb:", np.median(samples[:, 4]))

    # print()


    # import sys
    # sys.exit()

    import corner
    Vb = samples[:, 4]
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
    plt.savefig('Exercise7_corner_1.png', bbox_inches='tight')
    plt.savefig('Exercise7_corner_1.pdf', bbox_inches='tight')
    plt.show()

    plot_chains(sampler)
    plot_fit_with_samples(x, y, sigy, samples)

    # Repeat for sampler2
    Vb2 = samples2[:, 4]
    lower2, upper2 = np.percentile(Vb2, [0, 100 - mark])
    ranges2 = [
        (np.min(samples2[:, 0]), np.max(samples2[:, 0])),  # m
        (np.min(samples2[:, 1]), np.max(samples2[:, 1])),  # b
        (np.min(samples2[:, 2]), np.max(samples2[:, 2])),  # Pb
        (np.min(samples2[:, 3]), np.max(samples2[:, 3])),  # Yb
        (lower2, upper2)                                   # Vb
    ]

    corner.corner(samples2, labels=["m", "b", "Pb", "Yb", "Vb"], range=ranges2, quantiles=[0.16, 0.5, 0.84], bins=250, fig=plt.figure(figsize=(12, 7)), show_titles=True)
    plt.savefig('Exercise7_corner_2.png', bbox_inches='tight')
    plt.savefig('Exercise7_corner_2.pdf', bbox_inches='tight')
    plt.show()

    plot_chains(sampler2)
    plot_fit_with_samples(x, y, sigy / 2, samples2)



if __name__ == "__main__":
    main()
