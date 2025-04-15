"""
Extending Exercise12 to check if I can do the same, but for unique sigyi
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner


# Data
id = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x = np.array([201, 244, 47, 287, 203, 58, 210, 202, 198, 158, 165, 201, 157, 131, 166, 160, 186, 125, 218, 146])
y = np.array([592, 401, 583, 402, 495, 173, 479, 504, 510, 416, 393, 442, 317, 311, 400, 337, 423, 334, 533, 344])
SIGY = np.array([61, 25, 38, 15, 21, 15, 27, 14, 30, 16, 14, 25, 52, 16, 34, 31, 42, 26, 16, 22])
sigx = np.array([9, 4, 11, 7, 5, 9, 4, 4, 11, 7, 5, 5, 5, 6, 6, 5, 9, 8, 6, 5])
rhoxy = np.array([-0.84, 0.31, 0.64, -0.27, -0.33, 0.67, -0.02, -0.05, -0.84, -0.69, 0.30, -0.46, -0.03, 0.50, 0.73, -0.52, 0.90, 0.40, -0.78, -0.56])


# Remove the first 4 elements
id = id[4:]
x = x[4:]
y = y[4:]
SIGY = SIGY[4:]

N = len(x)


def logprior(params):

    m, b = params[:2]
    sigyi2 = params[2:]

    if 0 <= m <= 5 and -200 <= b <= 200 and np.all(0 < sigyi2):
        return 0
    return -np.inf


def loglikelihood(params, xi, yi):
    """
    Calculate the log likelihood of the data given the model parameters and noise parameters.
    """
    # Unpack the parameters
    m, b = params[:2]
    sigyi2 = params[2:]

    if len(sigyi2) != len(xi):
        import time
        time.sleep(2)
        raise ValueError("Length of sigyi2 must match length of xi")

    return np.sum(-0.5 * (((yi - (m * xi + b))**2 / sigyi2) + np.log(2 * np.pi * sigyi2)))

def logposterior(params, xi, yi):
    lp = logprior(params)
    if not np.isfinite(lp):
        return -np.inf
    
    ll = loglikelihood(params, xi, yi) 
    if not np.isfinite(ll):
        return -np.inf
    
    return lp + ll

def run_mcmc(x, y, nwalkers=1500, n_burn=3000, n_prod=1500):
    # Set up the initial position of the walkers
    p0 = np.random.rand(nwalkers, 2 + N)
    p0[:, 0] = np.random.uniform(0, 5, nwalkers)  # m
    p0[:, 1] = np.random.uniform(-200, 200, nwalkers)  # b
    p0[:, 2:] = np.random.uniform(1, 3000, (nwalkers, N))  # sigyi2

    # Set up the MCMC sampler
    sampler = emcee.EnsembleSampler(nwalkers, 2 + N, logposterior, args=(x, y))

    # Run the burn-in phase
    sampler.run_mcmc(p0, n_burn, progress=True)

    # Reset the sampler and run the production phase
    sampler.reset()
    sampler.run_mcmc(None, n_prod, progress=True)

    # Print the maximum likelihood value for each parameter
    max_likelihood_params = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
    print("Maximum Likelihood Parameters:")
    print(f"m = {max_likelihood_params[0]:.2f}")
    print(f"b = {max_likelihood_params[1]:.2f}")
    print("len(SIGY):", len(SIGY))
    for i, sigyi2 in enumerate(max_likelihood_params[2:], start=1):
        print(f"sigyi_{i} = {np.sqrt(sigyi2):.2f}   \t\t SIGY_{i} = {SIGY[i-1]:.2f}")

    # Add a timer of 10 seconds before proceeding
    import time
    time.sleep(10)

    return sampler

def plot_results(sampler, x, y):
    # Plot the results
    labels = ["m", "b"] + [f"sigyi2_{i+1}" for i in range(N)]
    fig = plt.figure(figsize=(8, 8), dpi=100, tight_layout=True)
    corner.corner(sampler.flatchain, labels=labels, fig=fig, show_titles=True, bins=30)
    plt.show()

    # Find the index of the maximum log posterior probability
    max_index = np.argmax(sampler.flatlnprobability)

    # Retrieve the corresponding parameters (m, b, sigyi^2)
    max_likelihood_params = sampler.flatchain[max_index]
    m_ml, b_ml = max_likelihood_params[:2]
    sigyi2_ml = max_likelihood_params[2:]

    # Print the maximum likelihood parameters
    print("Maximum Likelihood Parameters:")
    print(f"m = {m_ml:.2f}")
    print(f"b = {b_ml:.2f}")
    print(f"sigyi = {[f'{np.sqrt(val):.2f}' for val in sigyi2_ml]}")

    label1 = f"y = {m_ml:.2f}x + {b_ml:.2f}\n√S = {np.sqrt(sigyi2_ml):.2f}"


    plt.errorbar(x, y, yerr=np.sqrt(sigyi2_ml), fmt='o', capsize=3, capthick=2, label='Data')

    x_plot = np.linspace(0, 300, 100)
    y_plot = m_ml * x_plot + b_ml
    plt.plot(x_plot, y_plot)
    plt.text(0.01, 0.99, label1, fontsize=8, ha='left', va='top', transform=plt.gca().transAxes)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 300)
    plt.ylim(0, 700)

    plt.savefig('Exercise12_extended_part_1.png', bbox_inches='tight')
    plt.savefig('Exercise12_extended_part_1.pdf', bbox_inches='tight')
    plt.show()

    print()
    print()

    H1, xedges, yedges = np.histogram2d(sampler.flatchain[:, 0], sampler.flatchain[:, 1], bins=250, density=True)
    i,j = np.unravel_index(np.argmax(H1), H1.shape)
    m_map = 0.5*(xedges[i]+xedges[i+1])
    b_map = 0.5*(yedges[j]+yedges[j+1])

    print("MAP of m:", m_map)
    print("MAP of b:", b_map)

    label2 = f"y = {m_map:.2f}x + {b_map:.2f}\n"

    H2, sigyi2_edges = np.histogram(sampler.flatchain[:, 2], bins=500, density=True)

    sig_map = sigyi2_edges[np.argmax(H2)]
    print("MAP of sigyi:", np.sqrt(sig_map))

    label2 += f"√S = {np.sqrt(sig_map):.2f}"


    plt.errorbar(x, y, yerr=np.sqrt(sig_map), fmt='o', capsize=3, capthick=2)
    x_plot = np.linspace(0, 300, 100)
    y_plot = m_map * x_plot + b_map
    plt.plot(x_plot, y_plot)
    plt.text(0.01, 0.99, label2, fontsize=8, ha='left', va='top', transform=plt.gca().transAxes)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 300)
    plt.ylim(0, 700)
    plt.savefig('Exercise12_extended_part_2.png', bbox_inches='tight')
    plt.savefig('Exercise12_extended_part_2.pdf', bbox_inches='tight')
    plt.show()

def main():
    # Run MCMC
    sampler = run_mcmc(x, y)

    # Plot results
    plot_results(sampler, x, y)

    print()
    print()
    print()
    print("THE ANSWER IS A BIT DIFFERENT FOR b, but m is fine. AND ALSO Sigmas dont match. BUT THAT SHOULDNT BE A PROBLEM RIGHT? I NEED TO MAKE SURE THAT THE SIGMAS DONT VARY MUCH ON DIFFERENT RUNS (That means, i have to approach the correctish value, then it is fine)\n ANY OTHER DIFFERNCE IS BECAUSE OF THE DATA NOT BEING THE BEST ITSELF. IT NEEDNT FOLLOW THE SAME SIGMA AS THE DATA. THIS DATA NEEDNT BE THE BEST REPRESENTATION OF THE TRUE LINE!!!")
    print("\n\nThis makes sense now. I can anytime make the variances spread over many of its maximas, however, when I integrate it out, the effect is gonna be similar. So even if I keep checking the maximum and it keeps changing every time, that needn't mean that, my answer is not correct. It could simply be that, it is slowly converging(because too many parameters), and integral can still converge better!!\n\n")

if __name__ == "__main__":
    main()

