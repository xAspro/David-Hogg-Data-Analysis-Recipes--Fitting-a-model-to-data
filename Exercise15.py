"""
Reproducing the Figure 12 in David Hogg's paper "Data analysis recipes: Fitting a model to data"

Compares forward and reverse fitting without the outliers. The result is significantly different.

"""


import matplotlib.pyplot as plt
import numpy as np

# Data
id = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x = np.array([201, 244, 47, 287, 203, 58, 210, 202, 198, 158, 165, 201, 157, 131, 166, 160, 186, 125, 218, 146])
y = np.array([592, 401, 583, 402, 495, 173, 479, 504, 510, 416, 393, 442, 317, 311, 400, 337, 423, 334, 533, 344])
sigy = np.array([61, 25, 38, 15, 21, 15, 27, 14, 30, 16, 14, 25, 52, 16, 34, 31, 42, 26, 16, 22])
sigx = np.array([9, 4, 11, 7, 5, 9, 4, 4, 11, 7, 5, 5, 5, 6, 6, 5, 9, 8, 6, 5])
rhoxy = np.array([-0.84, 0.31, 0.64, -0.27, -0.33, 0.67, -0.02, -0.05, -0.84, 
                  -0.69, 0.30, -0.46, -0.03, 0.50, 0.73, -0.52, 0.90, 0.40, -0.78, -0.56])

x = x[4:].reshape(-1, 1)
y = y[4:].reshape(-1, 1)
sigy = sigy[4:]
sigx = sigx[4:]


def weighted_fit(x, y, sigma):
    A = np.hstack((np.ones_like(x), x))
    Cinv = np.diag(1 / sigma**2)
    cov = np.linalg.inv(A.T @ Cinv @ A)
    params = cov @ A.T @ Cinv @ y
    b, m = params.flatten()
    db, dm = np.sqrt(np.diag(cov))
    return m, b, dm, db


# Forward fit (y = mx + b, using σy)
m_f, b_f, dm_f, db_f = weighted_fit(x, y, sigy)

print("\nForward Fit (y vs x, using sig_y):")
print(f"  y = ({m_f:.3f} ± {dm_f:.3f}) x + ({b_f:.1f} ± {db_f:.1f})")


# Reverse fit (x = ay + b, using σx)
a_r, b_r, da_r, db_r = weighted_fit(y, x, sigx)

print("\nReverse Fit (x vs y, using sig_x):")
print(f"  x = ({a_r:.4f} ± {da_r:.4f}) y + ({b_r:.2f} ± {db_r:.2f})")



# Reversing the reverse fit
m_rev = 1 / a_r
b_rev = -b_r / a_r

# Uncertainties by error propagation
dm_rev = da_r / a_r**2
db_rev = np.sqrt((db_r / a_r)**2 + (b_r * da_r / a_r**2)**2)

print("\nInverted Reverse Fit (expressed as y vs x):")
print(f"  y = ({m_rev:.3f} ± {dm_rev:.3f}) x + ({b_rev:.1f} ± {db_rev:.1f})\n")


plt.figure(figsize=(8, 5))
plt.errorbar(x.flatten(), y.flatten(), yerr=sigy, fmt='o', label="Data", capsize=3)

xfit = np.linspace(0, 300, 200)
plt.plot(xfit, b_f + m_f * xfit, 'r', label=f"Forward fit, y = ({m_f:.3f} ± {dm_f:.3f}) x + ({b_f:.1f} ± {db_f:.1f})")
plt.plot(xfit, b_rev + m_rev * xfit, 'g', label=f"Reverse fit (inverted), y = ({m_rev:.3f} ± {dm_rev:.3f}) x + ({b_rev:.1f} ± {db_rev:.1f})")

plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0, 300)
plt.ylim(0, 700)
plt.legend()
plt.grid(True)
plt.show()

