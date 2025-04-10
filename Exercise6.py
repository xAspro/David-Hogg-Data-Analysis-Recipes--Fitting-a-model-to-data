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



# id = id.reshape(-1, 1)
# x = x.reshape(-1, 1)
# y = y.reshape(-1, 1)
# sigy = sigy


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

def run_mcmc(xi, yi, sigyi, nwalkers=200, nsteps_burn=10, nsteps_prod=100):
    """
    Run the MCMC simulation using emcee.
    """
    # Define the number of dimensions
    # m, b, Pb, Yb, Vb
    ndim = 5

    # Define the initial positions of the walkers
    # p0 = np.random.rand(nwalkers, ndim)
    p0 = np.empty((nwalkers, 5))
    # p0[:, 0] = np.random.uniform(0, 10, size=nwalkers)   # m
    # p0[:, 1] = np.random.uniform(-100, 100, size=nwalkers) # b
    # p0[:, 2] = np.random.uniform(0, 1, size=nwalkers)    # Pb
    # p0[:, 3] = np.random.uniform(-100, 100, size=nwalkers) # Yb
    # p0[:, 4] = np.random.uniform(0, 10, size=nwalkers)   # Vb

    p0[:, 0] = np.random.uniform(0, 2, size=nwalkers)   # m
    p0[:, 1] = np.random.uniform(0, 200, size=nwalkers) # b
    p0[:, 2] = np.random.uniform(0, 0.1, size=nwalkers)    # Pb
    p0[:, 3] = np.random.uniform(0, 200, size=nwalkers) # Yb
    p0[:, 4] = np.random.uniform(0, 100, size=nwalkers)   # Vb

    # Create the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, loglikelihood, args=(xi, yi, sigyi))

    # Run the MCMC simulation
    # sampler.run_mcmc(p0, nsteps)

    sampler.run_mcmc(p0, nsteps_burn, progress=True)
    sampler.reset()  # clear burn-in

    # Production
    sampler.run_mcmc(None, nsteps_prod, progress=True)

    return sampler

def plot_results(samples):
    """
    Plot the results of the MCMC simulation.
    """
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(samples[:, 0], samples[:, 1], 'k.', markersize=1)
    plt.xlabel('m')
    plt.ylabel('b')
    plt.title('MCMC Results')
    plt.show()
    return samples



def plot_chains(sampler):
    fig, axes = plt.subplots(5, figsize=(8, 5), sharex=True)
    samples = sampler.get_chain()

    print("Shape:", samples.shape)  # Shape of the array
    print("Data type:", samples.dtype)  # Data type of the elements
    print("Number of dimensions:", samples.ndim)  # Number of dimensions

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
    plt.errorbar(x, y, yerr=sigy, fmt='o', color='red', markersize=4, label="Data")

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
    print("Mean of Pb:", np.mean(samples[:, 2]))
    print("Mean of Yb:", np.mean(samples[:, 3]))
    print("Mean of Vb:", np.mean(samples[:, 4]))

    print()

    print("Median of m:", np.median(samples[:, 0]))
    print("Median of b:", np.median(samples[:, 1]))
    print("Median of Pb:", np.median(samples[:, 2]))
    print("Median of Yb:", np.median(samples[:, 3]))
    print("Median of Vb:", np.median(samples[:, 4]))

    print()

    H, xedges, yedges = np.histogram2d(samples[:,0], samples[:,1], bins=50)
    i,j = np.unravel_index(np.argmax(H), H.shape)
    m_map = 0.5*(xedges[i]+xedges[i+1])
    b_map = 0.5*(yedges[j]+yedges[j+1])

    print("MAP of m:", m_map)
    print("MAP of b:", b_map)
    print()


    import corner
    # corner.corner(samples, labels=["m", "b"], truths=[m, c], bins=50, smooth=1.0, show_titles=True)
    corner.corner(samples, labels=["m", "b", "Pb", "Yb", "Vb"], bins=150, fig=plt.figure(figsize=(12, 7)))
    plt.show()

    # plot_chains(sampler)

    # Plot the fit with samples
    # plot_fit_with_samples(x, y, sigy, samples)



if __name__ == "__main__":
    main()
