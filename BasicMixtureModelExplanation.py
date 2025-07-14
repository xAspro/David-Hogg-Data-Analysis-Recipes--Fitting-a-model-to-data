import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import emcee
import corner
import sys


# Generate synthetic data
def generate_data(sigma):
    m, b = 0.0, 2.0
    # x1 = np.array([2, 4, 6])  # y = mx + b
    # x2 = np.array([1, 3, 5]) +1  # y = mx + b + 1
    # x3 = np.array([0, 8])     # y = mx + b + 0.5


    x1 = np.array([2, 4, 6])  # y = mx + b
    x2 = np.array([1, 3, 5]) +1  # y = mx + b + 1
    x3 = np.array([0, 8])     # y = mx + b + 0.5

    y1 = m * x1 + b
    y2 = m * x2 + b + 1
    y3 = m * x3 + b + 0.5

    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])
    yerr = np.full_like(y, sigma)

    return x, y, yerr

# Fit function
def linear_model(x, m, b):
    return m * x + b

# Do standard fit
sigmas = [1.0, 0.1, 0.01]
fits = []

for sigma in sigmas:
    x, y, yerr = generate_data(sigma)
    popt, pcov = curve_fit(linear_model, x, y, sigma=yerr)
    fits.append((sigma, x, y, yerr, popt))

# Plot standard fits
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
xplot = np.linspace(-1, 8, 100)

for i, (sigma, x, y, yerr, popt) in enumerate(fits):
    axs[i].errorbar(x, y, yerr=yerr, fmt='o', label='Data')
    axs[i].plot(xplot, linear_model(xplot, *popt), label=f'Fit: m={popt[0]:.2f}, b={popt[1]:.2f}')
    axs[i].set_title(f'Standard Fit (σ={sigma})')
    axs[i].legend()
    axs[i].set_xlabel("x")
    axs[i].set_ylabel("y")

plt.tight_layout()
# plt.show()
plt.close()




# Log-likelihood with a mixture model
def log_likelihood(theta, x, y, yerr):
    m, b, Pb, Yb, Vb = theta
    if prior_tag == 2:
        Pb = Pb_val
    model = m * x + b
    sigma2_fg = yerr**2
    log_p_fg = -0.5 * ((y - model)**2 / sigma2_fg + np.log(2 * np.pi * sigma2_fg))

    sigma2_bg = yerr**2 + Vb
    log_p_bg = -0.5 * ((y - Yb)**2 / sigma2_bg + np.log(2 * np.pi * sigma2_bg))

    log_numerator = np.log(Pb) + log_p_bg
    log_denominator = np.logaddexp(np.log(1 - Pb) + log_p_fg, log_numerator)

    return np.sum(log_denominator)

# Log-prior
def log_prior(theta):
    m, b, Pb, Yb, Vb = theta
    # if prior_tag == 1:
        # if -2 < m < 8 and -10 < b < 5 and 0 < Pb < 1 and -100 < Yb < 100 and 0 < Vb < 15:
        #     return 0.0
    # elif prior_tag == 2:
        # if -2 < m < 8 and -10 < b < 5 and Pb_val - Pb_dif < Pb < Pb_val + Pb_dif and -100 < Yb < 100 and 0 < Vb < 15:
        #     return 0.0
    if -2 < m < 8 and -10 < b < 5 and 0 < Pb < 1 and -100 < Yb < 100 and 0 < Vb < 15:
        return 0.0
    return -np.inf

# Full log-probability
def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

# MCMC setup
sigma = 0.15
prior_tag = 1
number_of_runs = 30000
discard = number_of_runs // 3
x, y, yerr = generate_data(sigma)

ndim = 5
nwalkers = 10
initial = np.array([2, 1.0, 0.5, 0, 1])
pos = initial + 0.5 * np.random.uniform(-1, 1, size=(nwalkers, ndim))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
sampler.run_mcmc(pos, number_of_runs, progress=True)

# Plot MCMC chains for each parameter
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels = ["m", "b", "Pb", "Yb", "Vb"]
samples = sampler.get_chain(discard=discard, flat=False)
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], alpha=0.5)
    ax.set_ylabel(labels[i])
axes[-1].set_xlabel("Step number")
plt.tight_layout()
plt.savefig(f"Mixture_Model_Example_Chains_{number_of_runs}.png")
plt.savefig(f"Mixture_Model_Example_Chains_{number_of_runs}.pdf")
plt.show()


fig = plt.figure(figsize=(8, 6))
# Plot the corner plot for the posterior samples
figure = corner.corner(
    sampler.get_chain(flat=True),
    bins=50,
    quantiles=[0.16, 0.5, 0.84],
    levels=[0.1175, 0.393, 0.676, 0.865],
    smooth=True,
    fill_contours=True,
    labels=["m", "b", "Pb", "Yb", "Vb"],
    # truths=[m_mcmc, b_mcmc, None, None, None],
    show_titles=True,
    title_fmt=".2f",
    fig=fig,
    title_kwargs={"fontsize": 12},
    plot_datapoints=False,
)
plt.savefig(f"Mixture_Model_Example_Corner_{number_of_runs}.png", dpi=300)
plt.savefig(f"Mixture_Model_Example_Corner_{number_of_runs}.pdf", dpi=300)
plt.show()

# Get best-fit parameters from posterior median
flat_samples = sampler.get_chain(discard=discard, flat=True)

m, b, Pb, Yb, Vb = np.median(flat_samples[:, :], axis=0)


# Marginalize over nuisance parameters (Pb, Yb, Vb)

# 2D histogram of posterior samples for m and b
H, m_edges, b_edges = np.histogram2d(flat_samples[:, 0], flat_samples[:, 1], bins=50)

# Find the indices of the maximum value in the 2D histogram
max_idx = np.unravel_index(np.argmax(H), H.shape)
m_mcmc = 0.5 * (m_edges[max_idx[0]] + m_edges[max_idx[0] + 1])
b_mcmc = 0.5 * (b_edges[max_idx[1]] + b_edges[max_idx[1] + 1])

fig = plt.figure(figsize=(8, 6))
# Plot the corner plot for the posterior samples
figure = corner.corner(
    flat_samples,
    bins=50,
    quantiles=[0.16, 0.5, 0.84],
    levels=[0.1175, 0.393, 0.676, 0.865],
    smooth=True,
    fill_contours=True,
    labels=["m", "b", "Pb", "Yb", "Vb"],
    truths=[m_mcmc, b_mcmc, None, None, None],
    show_titles=True,
    title_fmt=".2f",
    fig=fig,
    title_kwargs={"fontsize": 12},
    plot_datapoints=False,
)
plt.savefig(f"Mixture_Model_Example_Corner_with_2D_MAP_{number_of_runs}.png", dpi=300)
plt.savefig(f"Mixture_Model_Example_Corner_with_2D_MAP_{number_of_runs}.pdf", dpi=300)
plt.show()



# Print median values for all parameters
param_names = ["m", "b", "Pb", "Yb", "Vb"]
medians = np.median(flat_samples, axis=0)
print("Posterior median values:")
for name, val in zip(param_names, medians):
    print(f"{name}: {val:.4f}")

# Print MAP (maximum a posteriori) estimate
log_probs = np.array([log_probability(theta, x, y, yerr) for theta in flat_samples])
map_idx = np.argmax(log_probs)
map_params = flat_samples[map_idx]
print("\nMAP (maximum a posteriori) values:")
for name, val in zip(param_names, map_params):
    print(f"{name}: {val:.4f}")

print("MAP Value:")
print(f"m: {m_mcmc:.4f}, b: {b_mcmc:.4f}")


# Calculate bad point probabilities
model = m_mcmc * x + b_mcmc
sigma2_fg = yerr**2
log_p_fg = -0.5 * ((y - model)**2 / sigma2_fg + np.log(2 * np.pi * sigma2_fg))

sigma2_bg = yerr**2 + Vb
log_p_bg = -0.5 * ((y - np.mean(y))**2 / sigma2_bg + np.log(2 * np.pi * sigma2_bg))

log_numerator = np.log(Pb) + log_p_bg
log_denominator = np.logaddexp(np.log(1 - Pb) + log_p_fg, log_numerator)
log_bad_prob = log_numerator - log_denominator
bad_prob = np.exp(log_bad_prob)

# Plot result
plt.figure(figsize=(8, 6))
plt.errorbar(x, y, yerr=yerr, fmt='o', label='Data')
plt.plot(xplot, linear_model(xplot, m_mcmc, b_mcmc), label=f'Mixture Fit: m={m_mcmc:.2f}, b={b_mcmc:.2f}', color='black')

# Calculate 1-sigma uncertainty band for the predicted line

# Get posterior predictive lines
y_pred_samples = np.array([linear_model(xplot, m_s, b_s) for m_s, b_s in flat_samples[:, :2]])
y_pred_median = np.median(y_pred_samples, axis=0)
y_pred_lower = np.percentile(y_pred_samples, 16, axis=0)
y_pred_upper = np.percentile(y_pred_samples, 84, axis=0)

plt.fill_between(xplot, y_pred_lower, y_pred_upper, color='gray', alpha=0.3, label=r'1$\sigma$ band')

# Mark bad points with red color, transparency based on bad_prob
# for xi, yi, pi in zip(x, y, bad_prob):
#     plt.scatter(xi, yi, color='red', alpha=pi, s=100, edgecolor='black', label='Bad prob' if pi > 0.5 and 'Bad prob' not in plt.gca().get_legend_handles_labels()[1] else None)

for xi, yi, pi in zip(x, y, bad_prob):
    if pi > 0.5:
        plt.scatter(xi, yi, color='red', s=100, edgecolor='black', label='Bad point' if 'Bad point' not in plt.gca().get_legend_handles_labels()[1] else None)

print(f"Bad point probabilities: {bad_prob}")
plt.title("Mixture Model Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"MixtureModelFit_with_2D_MAP_{number_of_runs}.pdf", dpi=300)
plt.savefig(f"MixtureModelFit_with_2D_MAP_{number_of_runs}.png", dpi=300)
plt.show()


with open(f"Mixture_Model_Example_Final_Output_{number_of_runs}.txt", "w") as f:
    f.write("Posterior median values:\n")
    for name, val in zip(param_names, medians):
        f.write(f"{name}: {val:.4f}\n")
    f.write("\nMAP (maximum a posteriori) values:\n")
    for name, val in zip(param_names, map_params):
        f.write(f"{name}: {val:.4f}\n")
    f.write("\nBad point probabilities:\n")
    f.write(f"{bad_prob}\n")

print("Saving output to Mixture_Model_Example_Final_Output.txt")




import sys
sys.exit(0)



# Repeat the analysis with prior_tag = 2
prior_tag = 2
Pb_dif = 0.05
Pb_val = map_params[-3]
x, y, yerr = generate_data(sigma)

# Re-initialize walkers
pos = initial + 0.5 * np.random.uniform(-1, 1, size=(nwalkers, ndim))
pos[:, 2] = Pb_val + np.random.uniform(-0.0015, 0.0015, size=nwalkers)
sampler2 = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
sampler2.run_mcmc(pos, number_of_runs, progress=True)

# Plot MCMC chains for each parameter
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples2 = sampler2.get_chain(discard=discard, flat=False)
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples2[:, :, i], alpha=0.5)
    ax.set_ylabel(labels[i])
axes[-1].set_xlabel("Step number")
plt.tight_layout()
plt.savefig(f"Mixture_Model_Example_Chains_prior2_{number_of_runs}.png")
plt.savefig(f"Mixture_Model_Example_Chains_prior2_{number_of_runs}.pdf")
plt.show()

fig = plt.figure(figsize=(8, 6))
corner.corner(
    sampler2.get_chain(flat=True),
    bins=50,
    quantiles=[0.16, 0.5, 0.84],
    levels=[0.1175, 0.393, 0.676, 0.865],
    smooth=True,
    fill_contours=True,
    labels=labels,
    show_titles=True,
    title_fmt=".2f",
    fig=fig,
    title_kwargs={"fontsize": 12},
    plot_datapoints=False,
)
plt.savefig(f"Mixture_Model_Example_Corner_prior2_{number_of_runs}.png", dpi=300)
plt.savefig(f"Mixture_Model_Example_Corner_prior2_{number_of_runs}.pdf", dpi=300)
plt.show()

flat_samples2 = sampler2.get_chain(discard=discard, flat=True)
m2, b2, Pb2, Yb2, Vb2 = np.median(flat_samples2, axis=0)

# 2D histogram for MAP
H2, m_edges2, b_edges2 = np.histogram2d(flat_samples2[:, 0], flat_samples2[:, 1], bins=50)
max_idx2 = np.unravel_index(np.argmax(H2), H2.shape)
m_mcmc2 = 0.5 * (m_edges2[max_idx2[0]] + m_edges2[max_idx2[0] + 1])
b_mcmc2 = 0.5 * (b_edges2[max_idx2[1]] + b_edges2[max_idx2[1] + 1])

fig = plt.figure(figsize=(8, 6))
corner.corner(
    flat_samples2,
    bins=50,
    quantiles=[0.16, 0.5, 0.84],
    levels=[0.1175, 0.393, 0.676, 0.865],
    smooth=True,
    fill_contours=True,
    labels=labels,
    truths=[m_mcmc2, b_mcmc2, None, None, None],
    show_titles=True,
    title_fmt=".2f",
    fig=fig,
    title_kwargs={"fontsize": 12},
    plot_datapoints=False,
)
plt.savefig(f"Mixture_Model_Example_Corner_with_2D_MAP_prior2_{number_of_runs}.png", dpi=300)
plt.savefig(f"Mixture_Model_Example_Corner_with_2D_MAP_prior2_{number_of_runs}.pdf", dpi=300)
plt.show()

medians2 = np.median(flat_samples2, axis=0)
print("Posterior median values (prior_tag=2):")
for name, val in zip(param_names, medians2):
    print(f"{name}: {val:.4f}")

log_probs2 = np.array([log_probability(theta, x, y, yerr) for theta in flat_samples2])
map_idx2 = np.argmax(log_probs2)
map_params2 = flat_samples2[map_idx2]
print("\nMAP (maximum a posteriori) values (prior_tag=2):")
for name, val in zip(param_names, map_params2):
    print(f"{name}: {val:.4f}")

print("MAP Value (prior_tag=2):")
print(f"m: {m_mcmc2:.4f}, b: {b_mcmc2:.4f}")

# Calculate bad point probabilities for prior_tag=2
model2 = m_mcmc2 * x + b_mcmc2
sigma2_fg2 = yerr**2
log_p_fg2 = -0.5 * ((y - model2)**2 / sigma2_fg2 + np.log(2 * np.pi * sigma2_fg2))
sigma2_bg2 = yerr**2 + Vb2
log_p_bg2 = -0.5 * ((y - np.mean(y))**2 / sigma2_bg2 + np.log(2 * np.pi * sigma2_bg2))
log_numerator2 = np.log(Pb2) + log_p_bg2
log_denominator2 = np.logaddexp(np.log(1 - Pb2) + log_p_fg2, log_numerator2)
log_bad_prob2 = log_numerator2 - log_denominator2
bad_prob2 = np.exp(log_bad_prob2)

plt.figure(figsize=(8, 6))
plt.errorbar(x, y, yerr=yerr, fmt='o', label='Data')
plt.plot(xplot, linear_model(xplot, m_mcmc2, b_mcmc2), label=f'Mixture Fit: m={m_mcmc2:.2f}, b={b_mcmc2:.2f}', color='black')
y_pred_samples2 = np.array([linear_model(xplot, m_s, b_s) for m_s, b_s in flat_samples2[:, :2]])
y_pred_median2 = np.median(y_pred_samples2, axis=0)
y_pred_lower2 = np.percentile(y_pred_samples2, 16, axis=0)
y_pred_upper2 = np.percentile(y_pred_samples2, 84, axis=0)
plt.fill_between(xplot, y_pred_lower2, y_pred_upper2, color='gray', alpha=0.3, label=r'1$\sigma$ band')
for xi, yi, pi in zip(x, y, bad_prob2):
    if pi > 0.5:
        plt.scatter(xi, yi, color='red', s=100, edgecolor='black', label='Bad point' if 'Bad point' not in plt.gca().get_legend_handles_labels()[1] else None)
print(f"Bad point probabilities (prior_tag=2): {bad_prob2}")
plt.title("Mixture Model Fit (prior_tag=2)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"MixtureModelFit_with_2D_MAP_prior2{number_of_runs}.pdf", dpi=300)
plt.savefig(f"MixtureModelFit_with_2D_MAP_prior2_{number_of_runs}.png", dpi=300)
plt.show()

with open(f"Mixture_Model_Example_Final_Output_prior2_{number_of_runs}.txt", "w") as f:
    f.write("Posterior median values (prior_tag=2):\n")
    for name, val in zip(param_names, medians2):
        f.write(f"{name}: {val:.4f}\n")
    f.write("\nMAP (maximum a posteriori) values (prior_tag=2):\n")
    for name, val in zip(param_names, map_params2):
        f.write(f"{name}: {val:.4f}\n")
    f.write("\nBad point probabilities (prior_tag=2):\n")
    f.write(f"{bad_prob2}\n")

print("Saving output to Mixture_Model_Example_Final_Output_prior2.txt")