"""
This code performs a Bayesian linear regression using Markov Chain Monte Carlo (MCMC) with the emcee package.
It generates synthetic data for a straight line, adds noise, and then fits the model to the noisy data using MCMC.
It also includes functions to plot the results and analyze the posterior distributions of the parameters.
"""

import numpy as np
import emcee
import matplotlib.pyplot as plt


# Data
x = np.linspace(0, 10, 100)  # Generate 100 points between 0 and 10

# Generate y values for a straight line (y = mx + c)
m, c = 2, 5  # Slope and intercept
y = m * x + c

sig = 0.0001  # Standard deviation of the noise

# Add random noise to the y values
noise = np.random.normal(0, sig, size=x.shape)  # Mean 0, standard deviation 1
y_noisy = y + noise

def likelihood(params, xi, yi, sigyi):
    """
    Calculate the likelihood of the data given the model parameters.
    params: [m, b] (array of model parameters)
    xi, yi: Data points
    sigyi: Uncertainties in y
    """
    # sigyi = 1 # Assuming constant error for simplicity
    m, b = params  # Unpack the parameters
    return np.sum(-0.5 * ((yi - (m * xi + b)) / sigyi)**2)

def run_mcmc(xi, yi, sigyi, nwalkers=1000, nsteps_burn=300, nsteps_prod=500):
    """
    Run the MCMC simulation using emcee.
    """
    # Define the number of dimensions
    ndim = 2  # m and b

    # Define the initial positions of the walkers
    # p0 = np.random.rand(nwalkers, ndim)
    
    p0 = np.empty((nwalkers,2))
    p0[:,0] = np.random.uniform(0,10,size=nwalkers)   # m
    p0[:,1] = np.random.uniform(-10,10,size=nwalkers) # b


    # Create the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood, args=(xi, yi, sigyi))

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
    plt.scatter(samples[:, 0], samples[:, 1], s=1, color='black', alpha=0.5)
    # plt.xlim(0, 10)
    plt.xlabel('m')
    plt.ylabel('b')
    plt.title('MCMC Results')
    plt.show()
    return samples


def plot_chains(sampler):
    fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["m", "b"]
    for i in range(2):  # For each parameter
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("Step number")
    plt.show()

# Call this function after running the sampler


def main():
    # Generate random errors for the y values
    # sigy = np.random.uniform(0.5, 2.0, size=y_noisy.shape)  # Random errors between 0.5 and 2.0

    sigy = sig * np.ones_like(y_noisy)  # Assuming constant error for simplicity

    # Run the MCMC simulation
    sampler = run_mcmc(x, y, sigy)

    samples = sampler.get_chain(flat=True)
    # samples = run_mcmc(x, y)

    # Plot the results
    plot_results(samples)

    # Print the results
    print("Mean of m:", np.mean(samples[:, 0]))
    print("Mean of b:", np.mean(samples[:, 1]))

    print()

    print("Median of m:", np.median(samples[:, 0]))
    print("Median of b:", np.median(samples[:, 1]))

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
    corner.corner(samples, labels=["m", "b"], truths=[m, c], bins=150)
    plt.show()

    plot_chains(sampler)



if __name__ == "__main__":
    main()
