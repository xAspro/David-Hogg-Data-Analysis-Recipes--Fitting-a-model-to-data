"""
Reproducing the Figure 3 in David Hogg's paper "Data analysis recipes: Fitting a model to data"

Considering data points for a quadratic fit. 

"""



import matplotlib.pyplot as plt
import numpy as np


# Data
id = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x = np.array([201, 244, 47, 287, 203, 58, 210, 202, 198, 158, 165, 201, 157, 131, 166, 160, 186, 125, 218, 146])
y = np.array([592, 401, 583, 402, 495, 173, 479, 504, 510, 416, 393, 442, 317, 311, 400, 337, 423, 334, 533, 344])
sigy = np.array([61, 25, 38, 15, 21, 15, 27, 14, 30, 16, 14, 25, 52, 16, 34, 31, 42, 26, 16, 22])
sigx = np.array([9, 4, 11, 7, 5, 9, 4, 4, 11, 7, 5, 5, 5, 6, 6, 5, 9, 8, 6, 5])
rhoxy = np.array([-0.84, 0.31, 0.64, -0.27, -0.33, 0.67, -0.02, -0.05, -0.84, -0.69, 0.30, -0.46, -0.03, 0.50, 0.73, -0.52, 0.90, 0.40, -0.78, -0.56])

id = id[4:].reshape(-1, 1)
x = x[4:].reshape(-1, 1)
y = y[4:].reshape(-1, 1)
sigy = sigy[4:]

# print(id)
# print(x)
# print(y)
# print(sigy)

A = np.hstack((np.ones_like(x), x, x**2))
# print("A = ", A)
C = np.diag(sigy**2)
C_inv = np.linalg.inv(C)

cov_matrix = np.linalg.inv(A.T @ C_inv @ A)
# print(cov_matrix)

def fit_curve(A, C_inv, cov_matrix, y):
    X = cov_matrix @ A.T@ C_inv @ y
    # print("X = ", X)
    return X

# Fit the curve
# Unpack the returned array directly
b, m, q = fit_curve(A, C_inv, cov_matrix, y).flatten()

# print("b = ", b)
# print("m = ", m)
# print("q = ", q)

db, dm, dq = np.sqrt(np.diag(cov_matrix))
# print("db = ", db)
# print("dm = ", dm)
# print("dq = ", dq)


# Print the equation
print(f"\ny = ({q:.4f} ± {dq:.4f}) x² + ({m:.2f} ± {dm:.2f})x + ({b:.0f} ± {db:.0f})\n")

# Plot the data and the fit

xlim_min = 0
xlim_max = 300
ylim_min = 0
ylim_max = 700

plt.errorbar(x.flatten(), y.flatten(), yerr=sigy, fmt='o', capsize=3, capthick=2, label='Data')
x_fit = np.linspace(xlim_min, xlim_max, 100)
y_fit = b + m * x_fit + q * x_fit**2
plt.plot(x_fit, y_fit, label=f'Fit: y = ({q:.4f} ± {dq:.4f}) x$^2$ + ({m:.2f} ± {dm:.2f}) x + ({b:.0f} ± {db:.0f})')

plt.xlabel('x')
plt.ylabel('y')
plt.xlim(xlim_min, xlim_max)
plt.ylim(ylim_min, ylim_max)
plt.legend()
plt.savefig('Exercise3.png', bbox_inches='tight')
plt.show()
