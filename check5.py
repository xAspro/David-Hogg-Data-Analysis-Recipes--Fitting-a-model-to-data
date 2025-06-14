
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from sklearn.mixture import GaussianMixture
from collections import defaultdict
from scipy.stats import norm
import time

start_time = time.time()

Y = np.array([1, 2, 3, 42, 5])
sig = np.random.uniform(0, 1, len(Y))  # Adding some noise
# sig[3] += 1
X = np.arange(len(Y))
plt.figure(figsize=(8, 6))
# plt.scatter(X, Y, color='blue', alpha=0.6, label='Data')
plt.errorbar(X, Y, yerr=sig, fmt='o', color='red', label='Error bars', capsize=5, elinewidth=2, markeredgewidth=2)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Bar Plot of Y Values')
plt.grid()
# plt.show()
plt.close()


def logprior(params):
    m, b, Pb, Yb, Vb = params
    if 0 < m < 2 and -20 < b < 20 and 0 < Pb < 1 and 0 < Vb < 1000 :
        # return -np.log(1 + Pb) - np.log(1 + Vb)  # Log prior for Pb and Vb
        return -np.log(Pb) - np.log(Vb)  # Log prior for Pb and Vb
    return -np.inf

def loglikelihood(params, X, Y, sig):
    m, b, Pb, Yb, Vb = params
    
    model_fg = m * X + b
    weighted_residuals_fg = (Y - model_fg) / sig
    f_fg = 1 / np.sqrt(sig**2) * np.exp(-0.5 * (weighted_residuals_fg)**2)

    model_bg = Yb
    weighted_residuals_bg = (Y - model_bg) / np.sqrt(Vb + sig**2)
    f_bg = 1 / np.sqrt(Vb + sig**2) * np.exp(-0.5 * (weighted_residuals_bg)**2)
    return np.sum(np.log((1 - Pb) * f_fg + Pb * f_bg))

def logposterior(params, X, Y, sig):
    lp = logprior(params)
    if np.isinf(lp):
        return -np.inf  # If prior is not satisfied, return -inf
    return lp + loglikelihood(params, X, Y, sig)

def run_mcmc(X, Y, sig, nwalkers=50, nsteps=10000):
    ndim = 5  # Number of parameters: m, b, Pb, Yb, Vb
    p0 = np.random.rand(nwalkers, ndim) * [2, 20, 1, 10, 10]  # Initial guess for parameters

    sampler = emcee.EnsembleSampler(nwalkers, ndim, logposterior, args=(X, Y, sig))
    sampler.run_mcmc(p0, nsteps)

    return sampler

def plot_corner(samples):
    # Overlay a Gaussian Mixture Model on the 1D histograms and return the GMM parameters
    def get_gmm_params(samples):
        gmm_results = defaultdict(list)
        ndim = samples.shape[1]
        for i in range(ndim):
            data = samples[:, i].reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
            gmm.fit(data)
            # Sort components by weight (dominant first)
            stds = np.sqrt(gmm.covariances_.flatten())
            weights = gmm.weights_.flatten()
            peaks = weights / (np.sqrt(2 * np.pi) * stds)
            idx = np.argsort(peaks)[::-1]  # Sort by peak height, descending
            means = gmm.means_.flatten()[idx]
            stds = stds[idx]
            weights = weights[idx]
            gmm_results['means'].append(means)
            gmm_results['stds'].append(stds)
            gmm_results['weights'].append(weights)
        return gmm_results

    def overlay_normals(fig, samples, gmm_results):
        ndim = samples.shape[1]
        axes = np.array(fig.axes).reshape((ndim, ndim))
        for i in range(ndim):
            ax = axes[i, i]
            data = samples[:, i].reshape(-1, 1)
            means = gmm_results['means'][i]
            stds = gmm_results['stds'][i]
            weights = gmm_results['weights'][i]
            x = np.linspace(data.min(), data.max(), 200).reshape(-1, 1)
            # # Plot each component: dominant first, then subdominant
            for j in range(2):
                comp_pdf = weights[j] * norm.pdf(x, means[j], stds[j])
                comp_y = comp_pdf * (ax.get_ylim()[1] - ax.get_ylim()[0]) / comp_pdf.max() + ax.get_ylim()[0]
                label = 'Dominant' if j == 0 else 'Subdominant'
                ax.plot(x.flatten(), comp_y, lw=2 if j == 0 else 1, ls='-' if j == 0 else '--', label=f'{label} GMM')
            # for j in range(2):
            #     comp_pdf = weights[j] * norm.pdf(x, means[j], stds[j])
            #     label = 'Dominant' if j == 0 else 'Subdominant'
            #     ax.plot(x.flatten(), comp_pdf, lw=2 if j == 0 else 1, ls='-' if j == 0 else '--', label=f'{label} GMM')

            # counts, bins = np.histogram(data, bins=30)
            # bin_width = bins[1] - bins[0]
            # area = counts.sum() * bin_width
            # x = np.linspace(data.min(), data.max(), 200)
            # gmm_pdf = sum(weights[j] * norm.pdf(x, means[j], stds[j]) for j in range(2))
            # ax.plot(x, gmm_pdf * area, color='black', lw=2, label='GMM sum')
            # for j in range(2):
            #     comp_pdf = weights[j] * norm.pdf(x, means[j], stds[j])
            #     ax.plot(x, comp_pdf * area, lw=2 if j == 0 else 1, ls='-' if j == 0 else '--', label=f'GMM {j+1}')
            # gmm_pdf = sum(weights[j] * norm.pdf(x, means[j], stds[j]) for j in range(2))
            # ax.plot(x.flatten(), gmm_pdf, color='black', lw=2, label='GMM sum')
            ax.legend(fontsize=8)

    # Fit GMMs and get parameters
    gmm_results = get_gmm_params(samples)

    # Plot corner
    fig = corner.corner(
        samples,
        labels=['m', 'b', 'Pb', 'Yb', 'Vb'],
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 12},
        fig=plt.figure(figsize=(12, 8)),
        bins=50,
    )
    overlay_normals(fig, samples, gmm_results)
    plt.show()

    # Print GMM values for each parameter
    param_names = ['m', 'b', 'Pb', 'Yb', 'Vb']
    for i, name in enumerate(param_names):
        means = gmm_results['means'][i]
        stds = gmm_results['stds'][i]
        weights = gmm_results['weights'][i]
        print(f"{name}:")
        print(f"  Dominant: mean={means[0]:.3f}, std={stds[0]:.3f}, weight={weights[0]:.3f}")
        print(f"  Subdominant: mean={means[1]:.3f}, std={stds[1]:.3f}, weight={weights[1]:.3f}")

    return gmm_results

# def plot_corner(samples, param_names=None, bins=30):
#     if param_names is None:
#         param_names = [f"param{i}" for i in range(samples.shape[1])]

#     def get_gmm_params(samples):
#         gmm_results = defaultdict(list)
#         ndim = samples.shape[1]
#         for i in range(ndim):
#             data = samples[:, i].reshape(-1, 1)
#             # Fit GMM with 2 components
#             gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
#             gmm.fit(data)
#             # Sort components by weight (dominant first)
#             idx = np.argsort(gmm.weights_)[::-1]
#             means = gmm.means_.flatten()[idx]
#             stds = np.sqrt(gmm.covariances_.flatten())[idx]
#             weights = gmm.weights_.flatten()[idx]
#             gmm_results['means'].append(means)
#             gmm_results['stds'].append(stds)
#             gmm_results['weights'].append(weights)
#         return gmm_results

#     def overlay_gmm(fig, samples, gmm_results, bins):
#         ndim = samples.shape[1]
#         axes = np.array(fig.axes).reshape((ndim, ndim))
#         for i in range(ndim):
#             ax = axes[i, i]
#             data = samples[:, i]
#             means = gmm_results['means'][i]
#             stds = gmm_results['stds'][i]
#             weights = gmm_results['weights'][i]
#             x = np.linspace(data.min(), data.max(), 300)

#             # Get histogram area for scaling
#             counts, bin_edges = np.histogram(data, bins=bins)
#             bin_width = bin_edges[1] - bin_edges[0]
#             area = counts.sum() * bin_width

#             # Plot each GMM component, scaled to histogram area
#             for j in range(2):
#                 comp_pdf = weights[j] * norm.pdf(x, means[j], stds[j])
#                 ax.plot(
#                     x, comp_pdf * area,
#                     lw=2 if j == 0 else 1,
#                     ls='-' if j == 0 else '--',
#                     label='Dominant GMM' if j == 0 else 'Subdominant GMM'
#                 )

#             # Plot the sum of both components
#             gmm_pdf = sum(weights[j] * norm.pdf(x, means[j], stds[j]) for j in range(2))
#             ax.plot(x, gmm_pdf * area, color='black', lw=2, label='GMM sum')
#             ax.legend(fontsize=8)

#     # Fit GMMs and get parameters
#     gmm_results = get_gmm_params(samples)

#     # Plot corner
#     fig = corner.corner(
#         samples,
#         labels=param_names,
#         show_titles=True,
#         title_fmt=".3f",
#         title_kwargs={"fontsize": 12},
#         bins=bins,
#         fig=plt.figure(figsize=(12, 8))
#     )
#     overlay_gmm(fig, samples, gmm_results, bins)
#     plt.show()

#     # Print GMM values for each parameter
#     for i, name in enumerate(param_names):
#         means = gmm_results['means'][i]
#         stds = gmm_results['stds'][i]
#         weights = gmm_results['weights'][i]
#         print(f"{name}:")
#         print(f"  Dominant: mean={means[0]:.3f}, std={stds[0]:.3f}, weight={weights[0]:.3f}")
#         print(f"  Subdominant: mean={means[1]:.3f}, std={stds[1]:.3f}, weight={weights[1]:.3f}")

#     return gmm_results

if __name__ == "__main__":
    # Run MCMC
    sampler = run_mcmc(X, Y, sig)
    DISCARD = 1000  # Number of burn-in samples to discard

    samples = sampler.get_chain(flat=True, discard=DISCARD)  # Discard burn-in samples and flatten the chain


    # Plot corner plot
    gmm_result = plot_corner(samples)


    # Plot chains without discard
    fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
    labels = ['m', 'b', 'Pb', 'Yb', 'Vb']
    for i in range(5):
        axes[i].plot(sampler.get_chain()[:, :, i], alpha=0.5)
        axes[i].set_ylabel(labels[i])
    axes[-1].set_xlabel("Step number")
    fig.suptitle("Chains without discard")
    plt.tight_layout()
    plt.show()

    # Plot chains with discard
    fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
    for i in range(5):
        axes[i].plot(sampler.get_chain(discard=DISCARD)[:, :, i], alpha=0.5)
        axes[i].set_ylabel(labels[i])
    axes[-1].set_xlabel("Step number")
    fig.suptitle(f"Chains with discard={DISCARD}")
    plt.tight_layout()
    plt.show()

    # Print autocorrelation time and acceptance rate
    try:
        tau = sampler.get_autocorr_time()
        print(f"Autocorrelation time: {tau}")
    except Exception as e:
        print(f"Could not compute autocorrelation time: {e}")

    acceptance_fraction = sampler.acceptance_fraction
    print(f"Mean acceptance rate: {np.mean(acceptance_fraction):.3f}")

    # Plot the MAP function and 1-sigma bands
    X_plot = np.linspace(X.min(), X.max(), 100)
    m_mcmc, b_mcmc = np.mean(samples[:, 0]), np.mean(samples[:, 1])
    y_map = m_mcmc * X_plot + b_mcmc

    # Compute 1-sigma credible interval for the function
    y_samples = np.array([m * X_plot + b for m, b, Pb, Yb, Vb in samples[np.random.choice(len(samples), 1000)]])
    y_lower = np.percentile(y_samples, 16, axis=0)
    y_upper = np.percentile(y_samples, 84, axis=0)

    plt.figure(figsize=(8, 6))
    # plt.scatter(X, Y, color='blue', alpha=0.6, label='Data')
    plt.errorbar(X, Y, yerr=sig, fmt='o', color='red', label='Error bars', capsize=5, elinewidth=2, markeredgewidth=2)
    plt.plot(X_plot, y_map, color='black', label='MAP fit')
    plt.fill_between(X_plot, y_lower, y_upper, color='gray', alpha=0.3, label='1$\sigma$ interval')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('MAP Fit with 1$\sigma$ Interval')
    plt.legend()
    plt.grid()
    plt.show()

    # Print summary statistics
    m_mean, b_mean, Pb_mean, Yb_mean, Vb_mean = np.mean(samples, axis=0)
    print(f"Mean parameters:\n  m = {m_mean:.3f}\n  b = {b_mean:.3f}\n  Pb = {Pb_mean:.3f}\n  Yb = {Yb_mean:.3f}\n  Vb = {Vb_mean:.3f}")

    # Plot dominant and subdominant curves from GMM means
    plt.figure(figsize=(8, 6))
    plt.errorbar(X, Y, yerr=sig, fmt='o', color='red', label='Data', capsize=5, elinewidth=2, markeredgewidth=2)

    # Dominant and subdominant means for m and b
    m_dom, m_sub = gmm_result['means'][0]
    b_dom, b_sub = gmm_result['means'][1]

    # Plot dominant curve
    y_dom = m_dom * X_plot + b_dom
    plt.plot(X_plot, y_dom, color='blue', lw=2, label='Dominant GMM fit')

    # Plot subdominant curve
    y_sub = m_sub * X_plot + b_sub
    plt.plot(X_plot, y_sub, color='green', lw=2, ls='--', label='Subdominant GMM fit')

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Dominant and Subdominant GMM Fits')
    plt.legend()
    plt.grid()
    plt.show()

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
