"""
Reproducing the Figure 10 in David Hogg's paper "Data analysis recipes: Fitting a model to data"

Plots bestfit line while accounting for both x and y uncertainties with outliers

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



id = id.reshape(-1, 1)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
sigy = sigy
sigx = sigx
rhoxy = rhoxy


# Part I

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



# Part II

def log_prior_2(params):
    t, b_perp, Pb, xb, yb, Vx, Vy = params
    if not (0.0 < t < 2 * np.pi and -1e4 < b_perp < 1e4 and
            0.0 < Pb < 1.0 and -1e4 < xb < 1e4 and -1e4 < yb < 1e4 and
            0.0 < Vx < 1e4 and 0.0 < Vy < 1e4):
        return -np.inf
    return 0.0

def log_likelihood_2(params, x, y ,sigy, sigx, rhoxy, eps=1e-15):
    t, b_perp, Pb, xb, yb, Vx, Vy = params

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

    Zb = np.vstack((xb, yb))

    S_bgi = cov + np.array([[Vx, 0], [0, Vy]])[np.newaxis, :, :]

    del_Z = Zi - Zb

    det_S_bgi = np.linalg.det(S_bgi)

    delta_bgi2 = np.einsum('ni,nij,nj->n', del_Z.T, np.linalg.inv(S_bgi), del_Z.T)


    log_pbgi = -np.log(2.0 * np.pi) - 0.5 * np.log(det_S_bgi) - 0.5 * delta_bgi2
    log_pfgi = -0.5 * np.log(2.0 * np.pi * Sigma_i2) - 0.5 * (delta_i**2 / Sigma_i2)

    Pb_clamped = np.clip(Pb, eps, 1.0 - eps)
    log_w_bg = np.log(Pb_clamped) + log_pbgi
    log_w_fg = np.log1p(-Pb_clamped) + log_pfgi

    # logsum for each data point (vectorized two-term version)
    log_mixture = np.logaddexp(log_w_bg, log_w_fg)

    # total log-likelihood
    lnL = np.sum(log_mixture)

    return lnL

def log_probability_2(params, x, y, sigy, sigx, rhoxy):
    lp = log_prior_2(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_2(params, x, y, sigy, sigx, rhoxy)
    return lp + ll

def run_mcmc_2(x, y, sigy, sigx, rhoxy, nwalkers=100, n_burn=1000, n_prod=5000):
    ndim = 7  # number of parameters (theta, b_perp, Pb, xb, yb, Vx, Vy)
    initial = np.array([np.pi/4, 0.0, 0.1, 150.0, 400.0, 50.0, 50.0])  # initial guess
    pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                    log_probability_2,
                                    args=(x, y, sigy, sigx, rhoxy))
    
    sampler.run_mcmc(pos, n_burn, progress=True)
    sampler.reset()
    sampler.run_mcmc(None, n_prod, progress=True)
    return sampler

def analyze_sampler_2(sampler):
    samples = sampler.get_chain(flat=True)
    
    fig = plt.figure(figsize=(15, 8))
    corner_fig = corner.corner(samples, 
                                labels=[r'$\theta$', r'$b_\perp$', r'$P_b$', r'$x_b$', r'$y_b$', r'$V_x$', r'$V_y$'],
                                truths=None,
                                show_titles=True, fig=fig)
    corner_fig.savefig('Exercise14_2_corner.png', bbox_inches='tight')
    plt.show()


def plot_data_and_fit_2(x, y, sigy, sigx, rhoxy, sampler):
    samples = sampler.get_chain(flat=True)

    theta = samples[:, 0]
    bperp = samples[:, 1]

    slope_samples = np.tan(theta)
    intercept_samples = bperp / np.cos(theta)

    slope_lo, slope_med, slope_hi = np.percentile(slope_samples, [16, 50, 84])
    int_lo,   int_med,   int_hi   = np.percentile(intercept_samples, [16, 50, 84])

    # median of all parameters (theta, b_perp, Pb, xb, yb, Vx, Vy)
    med_params_all = np.median(samples, axis=0)

    x_fit = np.linspace(0, 300, 100)
    y_fit = slope_med * x_fit + int_med


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
                           transOffset=ax.transData, facecolors='none', edgecolors='blue')
    ax.add_collection(ec)

    plt.plot(x_fit, y_fit, 'r-', label=f'Fit: y = {slope_med:.2f}x + {int_med:.2f}')

    # plot 10 random posterior samples (lines) from the sampler
    n_draws = min(10, samples.shape[0])
    idx = np.random.choice(samples.shape[0], size=n_draws, replace=False)
    for j, i in enumerate(idx):
        t_i, b_perp_i = samples[i, 0], samples[i, 1]
        slope_i = np.tan(t_i)
        intercept_i = b_perp_i / np.cos(t_i)
        y_i = slope_i * x_fit + intercept_i
        if j == 0:
            ax.plot(x_fit, y_i, color='gray', alpha=0.6, linewidth=1, label='Posterior samples')
        else:
            ax.plot(x_fit, y_i, color='gray', alpha=0.6, linewidth=1)

    plt.scatter(x, y, color='blue', label='Data Points')
    plt.errorbar(x.flatten(), y.flatten(), yerr=sigy, xerr=sigx, fmt='o', capsize=3, capthick=2, label='Data')
    bad_points_mask = find_bad_points(med_params_all, x, y, sigy, sigx, rhoxy, threshold=0.5)

    plt.scatter(x.flatten()[bad_points_mask], y.flatten()[bad_points_mask], facecolors='red', 
                edgecolors='red', s=100, label='Identified Outliers', zorder=5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 300)
    plt.ylim(0, 700)
    plt.legend()
    plt.grid()
    plt.savefig('Exercise14_2.png', bbox_inches='tight')
    plt.show()

def find_bad_points(med_params_all, x, y, sigy, sigx, rhoxy, threshold=0.5):
    t_med, b_perp_med, Pb_med, xb_med, yb_med, Vx_med, Vy_med = med_params_all

    v = np.array([-np.sin(t_med), np.cos(t_med)])
    Zi = np.vstack((x.ravel(), y.ravel()))
    Zb = np.vstack((xb_med, yb_med))
    delta = v.T @ Zi - b_perp_med

    def cov_mat(sigy, sigx, rhoxy):
        N = len(sigy)
        cov = np.empty((N, 2, 2))
        cov[:, 0, 0] = sigx**2
        cov[:, 1, 1] = sigy**2
        cov[:, 0, 1] = rhoxy * sigx * sigy
        cov[:, 1, 0] = cov[:, 0, 1]
        return cov
    
    cov = cov_mat(sigy, sigx, rhoxy)
    Sigma_i2 = np.einsum('i,nij,j->n', v, cov, v)
    del_Z = Zi - Zb
    S_bgi = cov + np.array([[Vx_med, 0], [0, Vy_med]])[np.newaxis, :, :]
    det_S_bgi = np.linalg.det(S_bgi)
    delta_bgi2 = np.einsum('ni,nij,nj->n', del_Z.T, np.linalg.inv(S_bgi), del_Z.T)
    log_pbgi = -np.log(2.0 * np.pi) - 0.5 * np.log(det_S_bgi) - 0.5 * delta_bgi2
    log_pfgi = -0.5 * np.log(2.0 * np.pi * Sigma_i2) - 0.5 * (delta**2 / Sigma_i2)
    log_Pb = np.log(Pb_med)
    log_1mPb = np.log1p(-Pb_med)
    log_w_bg = log_Pb + log_pbgi
    log_w_fg = log_1mPb + log_pfgi

    bad_point_mask = log_w_bg > log_w_fg

    return bad_point_mask

# Run MCMC for Part II
sampler_2 = run_mcmc_2(x, y, sigy, sigx, rhoxy, nwalkers=100, n_burn=1000, n_prod=5000)
analyze_sampler_2(sampler_2)
plot_data_and_fit_2(x, y, sigy, sigx, rhoxy, sampler_2)