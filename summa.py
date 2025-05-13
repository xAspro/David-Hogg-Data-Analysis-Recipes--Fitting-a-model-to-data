import numpy as np
import matplotlib.pyplot as plt
import corner
import time
import sys
import emcee
import datetime

# Define the current time at the start of the program
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

flag = 0
sig = 20
def function(x, params):
    return np.polyval(params, x)

def create_data(num_good_points=100, num_bad_points=10):
    good_param = [2, 0]
    bad_param = [4, 0]


    x_good = np.sort(np.random.uniform(0, 100, num_good_points))
    y_good = function(x_good, good_param) + np.random.normal(0, sig, num_good_points)

    x_bad = np.sort(np.random.uniform(0, 100, num_bad_points))
    y_bad = function(x_bad, bad_param) + np.random.normal(0, sig, num_bad_points)

    plt.scatter(x_good, y_good, color='blue', label='Good Points')
    plt.scatter(x_bad, y_bad, color='red', label='Bad Points')
    x_arr = np.linspace(0, 100, 100)
    y_arr_good = function(x_arr, good_param)
    y_arr_bad = function(x_arr, bad_param)
    plt.plot(x_arr, y_arr_good, color='blue', label='Good Function')
    plt.plot(x_arr, y_arr_bad, color='red', label='Bad Function')
    plt.title('Good and Bad Data Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(f"summa_real_data_plot_{current_time}.png") 
    plt.close()

    x_combined = np.concatenate((x_good, x_bad))
    y_combined = np.concatenate((y_good, y_bad))
    sorted_indices = np.argsort(x_combined)
    return np.sort(x_combined), y_combined[sorted_indices]

def logprior(params, NUM=2):
    func_params = params[:NUM]
    Pb, Yb, Vb = params[NUM:]

    if Pb < 0 or Pb > 1:
        return -np.inf
    if Vb <= 0:
        return -np.inf

    return - np.log(1 + Pb) - np.log(1 + Vb)

def loglikelihood(params, x, y, NUM=2):
    func_params = params[:NUM]
    Pb, Yb, Vb = params[NUM:]


    epsilon = 1e-10  # Small value to prevent division by zero
    safe_sig2 = sig**2 + epsilon
    safe_Vb = Vb + epsilon

    logforeground_model = np.log((1 / np.sqrt(2 * np.pi * safe_sig2))) + (-0.5 * np.clip(((y - function(x, func_params)) / sig)**2, -1e10, 1e10))
    logbackground_model = np.log((1 / np.sqrt(2 * np.pi * (safe_Vb + safe_sig2)))) + (-0.5 * np.clip(((y - Yb)**2 / (safe_Vb + safe_sig2)), -1e10, 1e10))

    a = np.log(1 - Pb) + logforeground_model
    b = np.log(Pb) + logbackground_model

    log10L = np.sum(np.logaddexp(a, b)) / np.log(10)

    return log10L

def logposterior(params, x, y, NUM=2):
    lp = logprior(params, NUM)
    if not np.isfinite(lp):
        return -np.inf

    ll = loglikelihood(params, x, y, NUM)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll

def run_mcmc_1(data, initial_params, num_samples=10000, num_burnin=1000, num_thin=10, NUM=2):

    # Create data
    x, y = data

    # MCMC sampling
    samples = []
    current_params = initial_params
    accepted_proposals = 0
    total_proposals = 0

    step_sizes = [0.1, 0.1, 0.01, 0.1, 0.1]

    for i in range(num_samples + num_burnin):
        if i % 1000 == 0 and i > 0:  # Gradually increase step sizes every 1000 iterations
            step_sizes = [s / 1.01 for s in step_sizes]

        proposal_params = current_params + np.random.normal(0, step_sizes)
        logp_current = logposterior(current_params, x, y, NUM)
        logp_proposal = logposterior(proposal_params, x, y, NUM)

        if np.log(np.random.rand()) < logp_proposal - logp_current:
            # print("IN!!!")
            accepted_proposals += 1
            current_params = proposal_params

        if i >= num_burnin and i % num_thin == 0:
            samples.append(current_params)
        total_proposals += 1

    global flag
    # print("samples = ", samples)
    # print(f"Acceptance rate: {accepted_proposals / total_proposals:.8f}")
    # if accepted_proposals / total_proposals < 1e-2:
    #     flag += 1
    
    # if flag == 5 and accepted_proposals / total_proposals > 0.001:
    #     print("Quitting MCMC due to frequent low acceptance rate.")
    #     sys.exit(0)


    return np.array(samples), accepted_proposals / total_proposals

def autocorrelation(chain, lag):
    """
    Compute the autocorrelation for a given chain and lag.

    Parameters:
        chain (np.ndarray): 1D array of MCMC samples for a parameter.
        lag (int): The lag at which to compute the autocorrelation.

    Returns:
        float: Autocorrelation value at the given lag.
    """
    n = len(chain)
    mean = np.mean(chain)
    variance = np.var(chain)

    # Compute autocorrelation
    autocorr = np.correlate(chain - mean, chain - mean, mode='full') / (n * variance)
    return autocorr[n - 1 + lag]  # Return the autocorrelation at the given lag

def autocorrelation_time(chain, c=5):
    """
    Estimate the autocorrelation time for a given chain.

    Parameters:
        chain (np.ndarray): 1D array of MCMC samples for a parameter.
        c (int): Safety factor to determine the maximum lag to consider.

    Returns:
        float: Estimated autocorrelation time.
    """
    n = len(chain)
    mean = np.mean(chain)
    variance = np.var(chain)

    # Compute autocorrelation for all lags
    autocorr = np.correlate(chain - mean, chain - mean, mode='full') / (n * variance)
    autocorr = autocorr[n - 1:]  # Keep only positive lags

    # Integrate autocorrelation to estimate autocorrelation time
    tau = 1 + 2 * np.cumsum(autocorr)
    window = np.arange(len(tau)) < c * tau  # Stop summing when the window condition is violated
    return tau[window].max()

def plot_chains(samples, title):
    """
    Plot the chains for each walker and parameter.

    Parameters:
        samples (np.ndarray): MCMC samples of shape (num_walkers, num_samples_per_walker, num_params).
        title (list): List of parameter names.
    """
    num_walkers, num_samples_per_walker, num_params = samples.shape

    for param_idx in range(num_params):
        plt.figure(figsize=(10, 6))
        for walker_idx in range(num_walkers):
            plt.plot(
                samples[walker_idx, :, param_idx],
                label=f'Walker {walker_idx + 1}',
                alpha=0.7,
            )
        plt.title(f'Chains for {title[param_idx]}')
        plt.xlabel('Iteration')
        plt.ylabel('Parameter Value')
        plt.legend(loc='upper right', fontsize='small', ncol=2)
        plt.grid()
        plt.tight_layout()
        plt.show()

# Generate a corner plot
def plot_corner(samples_2d, title):
    """
    Generate a corner plot for the MCMC samples.

    Parameters:
        samples_2d (np.ndarray): Flattened MCMC samples of shape (num_samples, num_params).
        title (list): List of parameter names.
    """
    fig = corner.corner(
        samples_2d,
        labels=title,  # Parameter names
        quantiles=[0.16, 0.5, 0.84],  # Show 16th, 50th (median), and 84th percentiles
        show_titles=True,  # Display titles with statistics
        title_fmt=".2f",  # Format for titles
        title_kwargs={"fontsize": 12},  # Font size for titles
        label_kwargs={"fontsize": 14},  # Font size for labels
        plot_datapoints=True,  # Show individual data points
        # fill_contours=True,  # Fill the contours in 2D plots
        levels=(0.68, 0.95),  # Confidence levels for contours
        color="black",  # Color of the plots
    )
    plt.savefig(f"corner_plot_1_{current_time}.png")  # Save the corner plot as a PNG file with the current time
    plt.show()

    fig = plt.figure(figsize=(12, 8), dpi=100)
    corner.corner(samples_2d, fig=fig, labels=title, show_titles=True, quantiles=[0.16, 0.5, 0.84])
    plt.savefig(f"corner_plot_2_{current_time}.png")  # Save the corner plot as a PNG file with the current time
    plt.show()

def mcmc_1_main(data, num_walkers=50, num_samples=10000, num_burnin=1000, num_thin=10, NUM=2):
    samples = []
    acc_frac = []

    start_time = time.time()

    # param_ranges = [(-5, 5), (-10, 10), (0, 1), (-20, 20), (10, 10)]
    # param_ranges = [(1.95, 2.05), (-0.1, 0.1), (0, 1), (-10, 10), (0, 5)]
    param_ranges = [(1, 3), (-2, 2), (0.1, 0.9), (-10, 10), (1, 5)]

    for i in range(num_walkers):
        initial_params = np.array([np.random.uniform(low, high) for low, high in param_ranges])
        # print(f"Initial params for walker {i + 1}: {initial_params}")
        walker_samples, walker_acc_frac = run_mcmc_1(data, initial_params, num_samples, num_burnin, num_thin, NUM)
        samples.append(walker_samples)
        acc_frac.append(walker_acc_frac)


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for {num_walkers} walkers: {elapsed_time:.2f} seconds")
    print(f"\n\n\nAcceptance fraction for {num_walkers} walkers: {np.mean(acc_frac):.4f}\n\n\n")
    samples = np.array(samples)

    samples_2d = samples.reshape(-1, samples.shape[-1])
    # print(samples.shape)
    # print("samples = ", samples)
    title = ['m', 'c', 'Pb', 'Yb', 'Vb']

    plot_chains(samples, title)
    plot_corner(samples_2d, title)

        # Compute and plot autocorrelation for each parameter and walker
    # lags = np.arange(1, 100)  # Define the range of lags to compute autocorrelation

    # for param_idx in range(samples.shape[-1]):  # Loop over each parameter
    #     plt.figure(figsize=(10, 6))
    #     for walker_idx in range(samples.shape[0]):  # Loop over each walker
    #         chain = samples[walker_idx, :, param_idx]  # Extract chain for walker and parameter
    #         autocorr_values = [autocorrelation(chain, lag) for lag in lags]  # Compute autocorrelation
    #         plt.plot(lags, autocorr_values, label=f'Walker {walker_idx + 1}', alpha=0.7)

    #     plt.title(f'Autocorrelation for Parameter {title[param_idx]}')
    #     plt.xlabel('Lag')
    #     plt.ylabel('Autocorrelation')
    #     plt.legend(loc='upper right', fontsize='small', ncol=2)
    #     plt.grid()
    #     plt.tight_layout()
    #     plt.savefig(f'autocorr_param_{param_idx + 1}.png')  # Save the plot for each parameter
    #     plt.show()

        # Compute autocorrelation time for each parameter
    autocorr_times = []
    for param_idx in range(samples.shape[-1]):  # Loop over each parameter
        param_autocorr_times = []
        for walker_idx in range(samples.shape[0]):  # Loop over each walker
            chain = samples[walker_idx, :, param_idx]  # Extract chain for walker and parameter
            tau = autocorrelation_time(chain)
            param_autocorr_times.append(tau)
        autocorr_times.append(np.mean(param_autocorr_times))  # Average over walkers

    print("\n\n\n")
    # Print autocorrelation times
    for param_idx, tau in enumerate(autocorr_times):
        print(f"Autocorrelation time for parameter {title[param_idx]}: {tau}")


    results = {}

    for i in range(samples_2d.shape[1]):
        param_samples = samples_2d[:, i]
        median = np.median(param_samples)
        lower_1sigma = np.percentile(param_samples, 16)
        upper_1sigma = np.percentile(param_samples, 84)

        results[f"param_{i}"] = {
            "median": median,
            "1sigma_lower": median - lower_1sigma,
            "1sigma_upper": upper_1sigma - median,
            "posterior_samples": param_samples,
        }

    
    for param, stats in results.items():
        print(f"{param}:")
        print(f"  Median: {stats['median']}")
        print(f"  1-Sigma Lower: {stats['1sigma_lower']}")
        print(f"  1-Sigma Upper: {stats['1sigma_upper']}")

    print("\n\nsample_2d.shape = ", samples_2d.shape)
    print("\n\nsamples_2d = ", samples_2d)

    return [stats["median"] for param, stats in results.items()]

def mcmc_2_main(data, nwalkers=50, nprod=1000, nburn=1000, NUM=2):
    # Create data
    x, y = data

    # Set up the MCMC sampler
    ndim = NUM + 3  # Number of parameters (2 for function params, 3 for Pb, Yb, Vb)
    nwalkers = nwalkers

    param_ranges = [(1, 3), (-2, 2), (0.1, 0.9), (-10, 10), (1, 5)]

    p0 = np.array([np.random.uniform(low, high, size=nwalkers) for low, high in param_ranges]).T

    assert p0.shape == (nwalkers, len(param_ranges)), "Shape of p0 is incorrect!"

    sampler = emcee.EnsembleSampler(nwalkers, ndim, logposterior, args=[x, y])

    sampler.run_mcmc(p0, nburn)
    sampler.reset()
    sampler.run_mcmc(None, nprod)

    samples = sampler.get_chain(flat=True)

    # Calculate median and 1-sigma intervals for each parameter
    results = {}
    for i in range(samples.shape[1]):
        param_samples = samples[:, i]
        median = np.median(param_samples)
        lower_1sigma = np.percentile(param_samples, 16)
        upper_1sigma = np.percentile(param_samples, 84)

        results[f"param_{i}"] = {
            "median": median,
            "1sigma_lower": median - lower_1sigma,
            "1sigma_upper": upper_1sigma - median,
        }

    # Print the results
    for param, stats in results.items():
        print(f"{param}:")
        print(f"  Median: {stats['median']}")
        print(f"  1-Sigma Lower: {stats['1sigma_lower']}")
        print(f"  1-Sigma Upper: {stats['1sigma_upper']}")

    fig = plt.figure(figsize=(12, 8), dpi=100)
    corner.corner(samples, labels=["m", "c", "Pb", "Yb", "Vb"], fig=fig, show_titles=True, quantiles=[0.16, 0.5, 0.84])
    plt.savefig(f"summa_corner_plot_3_{current_time}.png")  # Save the corner plot as a PNG file with the current time
    plt.show()
    plt.close()


    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 1, sampler.nwalkers))
    for i in range(ndim):
        ax = axes[i]
        for j in range(nwalkers):
            ax.plot(sampler.chain[j, :, i], color="k", alpha=0.3)
        ax.set_ylabel(f"param {i}")
        ax.yaxis.set_label_coords(-0.1, 0.5)
    plt.xlabel("step number")
    plt.savefig(f"summa_chain_plot_3_{current_time}.png")  # Save the chain plot as a PNG file with the current time
    plt.close()

    # Print acceptance rate
    acceptance_rate = np.mean(sampler.acceptance_fraction)
    print(f"Acceptance rate: {acceptance_rate:.4f}")

    # Compute and print autocorrelation time
    try:
        tau = sampler.get_autocorr_time()
        print("Autocorrelation time for each parameter:")
        for i, t in enumerate(tau):
            print(f"  Parameter {i}: {t:.2f}")
    except emcee.autocorr.AutocorrError:
        print("Warning: Autocorrelation time could not be reliably estimated.")

    return [stats["median"] for param, stats in results.items()]


def find_bad_data(data, param, NUM=2):
    x, y = data
    Pb, Yb, Vb = param[NUM:]

    model_vals = function(x, param[:NUM])

    Npoints = len(x)
    bad_prob_list = []
    # return np.full_like(z, 0.5)
    true_cnt = 0

    for i in range(Npoints):
        p_fg = (1 / np.sqrt(2 * np.pi * sig**2)) * np.exp(-0.5 * ((y[i] - model_vals[i]) / sig)**2)
        p_bg = (1 / np.sqrt(2 * np.pi * (Vb + sig**2))) * np.exp(-0.5 * ((y[i] - Yb)**2 / (Vb + sig**2)))


        epsilon = 1e-20
        numerator = Pb * p_bg + epsilon
        denominator = (1 - Pb) * p_fg + Pb * p_bg + epsilon

        bad_prob = numerator / denominator

        if bad_prob > 0.8:
            true_cnt += 1
            # print("\n\ni = ", i)
            # print(f"Data point = ({z[i]}, {y[i]})")
            # print(f"Corresponding point = ({z[i]}, {model_vals[i]})")
            # print("sigma = ", sig)

            # print("residual = ", y[i] - model_vals[i])
            # print("residual / sigma = ", (y[i] - model_vals[i]) / sig)
            # print("\nPb = ", Pb)
            # print("p_fg = ", p_fg)
            # print("p_bg = ", p_bg)

            # print()
            # print("numerator = ", Pb * p_bg)
            # print("denominator = ", (1 - Pb) * p_fg + Pb * p_bg)
            # print("numerator / denominator = ", Pb * p_bg / ((1 - Pb) * p_fg + Pb * p_bg))
            # print("mask = ", Pb * p_bg / ((1 - Pb) * p_fg + Pb * p_bg) > 0.5)

            ratio = p_fg / p_bg
            print("ratio = ", ratio)  # This should be ~0 if model fits well
        
        bad_prob_list.append(bad_prob)

    print("true_cnt = ", true_cnt)
    print("Implied bad points = ", true_cnt / Npoints * 100, "%")
    return np.array(bad_prob_list)



# data = create_data(15, 3)
# data = create_data(20, 5)
data = create_data(100, 10)
# data = create_data(10, 10)
# print("data points are")
# for i in range(len(data[0])):
#     print(f"({data[0][i]}, {data[1][i]})")

# sys.exit(0)

# mcmc_main(data, num_walkers=3, num_samples=10, num_burnin=2, num_thin=2, NUM=2)
n_walkers = 50
n_prod = 100000
n_burn = 1000
n_thin = 50

res_1 = mcmc_1_main(data, num_walkers=n_walkers, num_samples=n_prod, num_burnin=n_burn, num_thin=n_thin, NUM=2)

res_2 = mcmc_2_main(data, nwalkers=n_walkers, nprod=n_prod, nburn=n_burn, NUM=2)

print(f"\n\nres_1 = {res_1}\nres_2 = {res_2}\n\n")

x_arr = np.linspace(0, 100, 100)
y_arr = function(x_arr, res_1[:2])

y_bad_func_1 = function(x_arr, res_2[3:5])

prob_bad_points_1 = find_bad_data(data, res_1, NUM=2)

print("prob_bad_points = ", prob_bad_points_1)

mask = prob_bad_points_1 > 0.5
print("mask = ", mask)

# print("sigys = ", sigys)
# print("sigy = ", sigy)

# plt.errorbar(A, B, yerr=sig, fmt='o', ms=10, capsize=4, zorder=1)
# plt.scatter(A[mask], B[mask], facecolors='red', edgecolors='red', s=50, label='Bad Data Points', zorder=2)

plt.errorbar(data[0], data[1], yerr=sig, fmt='o', ms=10, capsize=4, zorder=1)
plt.scatter(data[0][mask], data[1][mask], c='red', label='Bad Data Points', alpha=0.7, edgecolor='black', s=50, zorder=2)

# plt.scatter(data[0][~mask], data[1][~mask], c='red', label='Data Points', edgecolor='black', s=14)
# plt.scatter(data[0][mask], data[1][mask], facecolors='none', edgecolors='red', s=8, label='Bad Data Points', alpha=0.5)
# plt.errorbar(data[0], data[1], yerr=sig, fmt='o', alpha=0.6, capsize=0.4)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting for x with Bad data in dataset')
plt.plot(x_arr, y_arr, color='blue', label='Good Function')
plt.plot(x_arr, y_bad_func_1, color='red', label='Bad Function')
plt.xlim(-5, 105)
plt.ylim(-10, 410)
plt.legend()
plt.grid()
plt.savefig(f"summa_plot_1_{current_time}.png")  # Save the plot as a PNG file with the current time
plt.show()
plt.close()


prob_bad_points_2 = find_bad_data(data, res_2, NUM=2)

y_arr_2 = function(x_arr, res_2[:2])
y_bad_func_2 = function(x_arr, res_2[3:5])

print("prob_bad_points = ", prob_bad_points_2)
mask = prob_bad_points_2 > 0.5
print("mask = ", mask)


plt.errorbar(data[0], data[1], yerr=sig, fmt='o', ms=10, capsize=4, zorder=1)
plt.scatter(data[0][mask], data[1][mask], c='red', label='Bad Data Points', alpha=0.7, edgecolor='black', s=50, zorder=2)


# plt.scatter(data[0][~mask], data[1][~mask], c='red', label='Data Points', edgecolor='black', s=14)
# plt.scatter(data[0][mask], data[1][mask], facecolors='none', edgecolors='red', s=8, label='Bad Data Points', alpha=0.5)
# plt.errorbar(data[0], data[1], yerr=sig, fmt='o', alpha=0.6, capsize=0.6)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting for x with Bad data in dataset')
plt.plot(x_arr, y_arr, color='blue', label='Good Function')
plt.plot(x_arr, y_bad_func_2, color='red', label='Bad Function')
plt.xlim(-5, 105)
plt.ylim(-10, 410)
plt.legend()
plt.grid()
plt.savefig(f"summa_plot_2_{current_time}.png")  # Save the plot as a PNG file with the current time
plt.show()
plt.close()


# import matplotlib.pyplot as plt
# import numpy as np

# n = 20
# sig_max = 10

# A = np.sort(np.random.uniform(0, 100, n))
# sig = np.random.uniform(0, sig_max, n)
# B = A + np.random.normal(0, sig, n)

# x = np.linspace(0, 100, 100)


# mask = sig > sig_max / 2
# print(mask)
# print('percent of bad points = ', np.sum(mask) / n * 100,'%')
# # plt.scatter(A[~mask], B[~mask], c='red', label='Data Points', edgecolor='black', s=50)
# plt.errorbar(A, B, yerr=sig, fmt='o', ms=10, capsize=4, zorder=1)
# plt.scatter(A[mask], B[mask], facecolors='red', edgecolors='red', s=50, label='Bad Data Points', zorder=2)

# plt.plot(x, x, color='blue', label='Good Function')
# plt.grid()
# plt.show()