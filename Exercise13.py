"""
Reproducing the Figure 9 in David Hogg's paper "Data analysis recipes: Fitting a model to data"

Plots bestfit line while accounting for both x and y uncertainties, without outliers.

"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from matplotlib.collections import EllipseCollection



# Data
id = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x = np.array([201, 244, 47, 287, 203, 58, 210, 202, 198, 158, 165, 201, 157, 131, 166, 160, 186, 125, 218, 146])
y = np.array([592, 401, 583, 402, 495, 173, 479, 504, 510, 416, 393, 442, 317, 311, 400, 337, 423, 334, 533, 344])
sigy = np.array([61, 25, 38, 15, 21, 15, 27, 14, 30, 16, 14, 25, 52, 16, 34, 31, 42, 26, 16, 22])
sigx = np.array([9, 4, 11, 7, 5, 9, 4, 4, 11, 7, 5, 5, 5, 6, 6, 5, 9, 8, 6, 5])
rhoxy = np.array([-0.84, 0.31, 0.64, -0.27, -0.33, 0.67, -0.02, -0.05, -0.84, 
                  -0.69, 0.30, -0.46, -0.03, 0.50, 0.73, -0.52, 0.90, 0.40, -0.78, -0.56])



id = id[4:].reshape(-1, 1)
x = x[4:].reshape(-1, 1)
y = y[4:].reshape(-1, 1)
sigy = sigy[4:]
sigx = sigx[4:]
rhoxy = rhoxy[4:]


def log_prior(params):
    t, b_perp = params
    if not (0.0 < t < 2 * np.pi and -1e4 < b_perp < 1e4):
        return -np.inf
    return 0.0

def log_likelihood(params, x, y ,sigy, sigx, rhoxy):
    t, b_perp = params      # theta, b_perpendicular

    def cov_mat(sigy, sigx, rhoxy):
        N = len(sigy)
        cov = np.empty((N, 2, 2))
        cov[:, 0, 0] = sigx**2
        cov[:, 1, 1] = sigy**2
        cov[:, 0, 1] = rhoxy * sigx * sigy
        cov[:, 1, 0] = cov[:, 0, 1]
        return cov
    
    v = np.array([-np.sin(t), np.cos(t)])

    Zi = np.vstack((x.ravel(), y.ravel()))

    delta_i = v.T @ Zi - b_perp

    cov = cov_mat(sigy, sigx, rhoxy)
    Sigma_i2 = np.einsum('i,nij,j->n', v, cov, v)

    lnL = - 0.5 * np.sum(delta_i**2/Sigma_i2)

    return lnL

def log_probability(params, x, y, sigy, sigx, rhoxy):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params, x, y, sigy, sigx, rhoxy)
    return lp + ll


def run_mcmc(x, y, sigy, sigx, rhoxy, nwalkers=50, n_burn=500, n_prod=5000):
    ndim = 2  # number of parameters (theta, b_perp)

    # pos = np.zeros((nwalkers, ndim))
    # pos[:, 0] = np.random.uniform(0.0, 2 * np.pi, size=nwalkers)  # theta
    # pos[:, 1] = np.random.uniform(-1e4, 1e4, size=nwalkers)  # b_perp

    initial = np.array([np.pi/4, 0.0])  # initial guess for (theta, b_perp)
    pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                    log_probability,
                                    args=(x, y, sigy, sigx, rhoxy))
    
    sampler.run_mcmc(pos, n_burn, progress=True)
    sampler.reset()
    sampler.run_mcmc(None, n_prod, progress=True)
    return sampler

def analyze_sampler(sampler):
    samples = sampler.get_chain(flat=True)
    
    corner_fig = corner.corner(samples, 
                                labels=[r'$\theta$', r'$b_\perp$'],
                                truths=None,
                                show_titles=True)
    corner_fig.savefig('Exercise13_corner.png', bbox_inches='tight')
    plt.show()

    # plot MCMC chains for each parameter
    chain = sampler.get_chain()  # shape (nsteps, nwalkers, ndim)
    nsteps, nwalkers, ndim = chain.shape
    param_names = [r'$\theta$', r'$b_\perp$']

    fig, axes = plt.subplots(ndim, 1, figsize=(10, 3 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]

    for i in range(ndim):
        ax = axes[i]
        ax.plot(chain[:, :, i], alpha=0.4, lw=0.8)
        ax.set_ylabel(param_names[i])
        ax.grid(alpha=0.3)

        # plot median and a few walker traces highlighted
        median_trace = np.median(chain[:, :, i], axis=1)
        ax.plot(median_trace, color='k', lw=1.5, label='median (over walkers)')
        # highlight first 3 walkers
        for w in range(min(3, nwalkers)):
            ax.plot(chain[:, w, i], lw=1.2)

    axes[-1].set_xlabel('Step')
    plt.tight_layout()
    fig.savefig('Exercise13_chains.png', bbox_inches='tight')
    plt.show()


def plot_data_and_fit(x, y, sigy, sigx, rhoxy, sampler):
    samples = sampler.get_chain(flat=True)
    theta_mcmc, b_perp_mcmc = np.percentile(samples, [16, 50, 84], axis=0).T

    t_median = theta_mcmc[1]
    b_perp_median = b_perp_mcmc[1]

    slope = np.tan(t_median)
    intercept = b_perp_median / np.cos(t_median)

    x_fit = np.linspace(0, 300, 100)
    y_fit = slope * x_fit + intercept


    c = 1

    fig, ax = plt.subplots(figsize=(8,6))



    widths = []
    heights = []
    angles = []
    offsets = []

    for xi, yi, sx, sy, rho in zip(x.flatten(), y.flatten(), sigx, sigy, rhoxy):
        cov = np.array([[sx*sx, rho * sx * sy],
                        [rho * sx * sy, sy*sy]])
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
        widths.append(2.0 * np.sqrt(c * vals[0]))
        heights.append(2.0 * np.sqrt(c * vals[1]))
        angles.append(angle)
        offsets.append((xi, yi))

    ec = EllipseCollection(widths, heights, angles, units='xy', offsets=offsets,
                        transOffset=ax.transData, facecolors='none', edgecolors='black', linewidths=1.5, alpha=0.7)
    ax.add_collection(ec)

    plt.plot(x_fit, y_fit, 'r-', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')
    plt.scatter(x, y, color='blue', label='Data Points')
    plt.errorbar(x.flatten(), y.flatten(), yerr=sigy, xerr=sigx, fmt='o', capsize=3, capthick=2, label='Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 300)
    plt.ylim(0, 700)
    plt.legend()
    plt.grid()
    plt.savefig('Exercise13.png', bbox_inches='tight')
    plt.show()
    



# Run MCMC
sampler = run_mcmc(x, y, sigy, sigx, rhoxy, nwalkers=100, n_burn=1000, n_prod=5000)
analyze_sampler(sampler)
plot_data_and_fit(x, y, sigy, sigx, rhoxy, sampler)