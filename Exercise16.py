"""
Reproducing the Figure 12 in David Hogg's paper "Data analysis recipes: Fitting a model to data"

Principal Component Analysis for inliers. The fit doesnt even look at the uncertainties in x and y.

"""
import numpy as np
import matplotlib.pyplot as plt

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


def compute_Q(x, y, sigy, sigx):
    N = len(x)
    Zi = np.vstack((x.ravel(), y.ravel()))
    z_mean = np.mean(Zi, axis=1, keepdims=True)
    delta_i = Zi - z_mean
    Qi = delta_i @ delta_i.T
    return Qi, z_mean

def principal_eigvec(Q):
    eigvals, eigvecs = np.linalg.eig(Q)
    
    D = np.diag(eigvals)
    P = eigvecs
    
    biggest_eigval_index = np.argmax(eigvals)
    principal_eigvec = eigvecs[:, biggest_eigval_index]
    
    return principal_eigvec

def plot_data_and_fit(x, y, sigy, sigx, principal_eigvec, Z_mean):
    plt.figure(figsize=(8, 5))
    plt.errorbar(x.flatten(), y.flatten(), yerr=sigy, xerr=sigx, fmt='o', label="Data", capsize=3)

    xfit = np.linspace(0, 300, 200)
    slope = principal_eigvec[1] / principal_eigvec[0]
    intercept = (Z_mean[1] - slope * Z_mean[0])[0]
    yfit = intercept + slope * xfit
    plt.plot(xfit, yfit, 'r', label=f"Principal Component Fit: y = {slope:.2f}x + {intercept:.2f}")
    print(f"\n\nPrincipal Component Fit: y = {slope:.2f}x + {intercept:.2f}\n\n")
    plt.scatter(Z_mean[0], Z_mean[1], color='red', label='Mean Point', zorder=5, s=100)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 300)
    plt.ylim(0, 700)
    plt.grid()
    plt.legend()
    plt.savefig('Exercise16.png', bbox_inches='tight')
    plt.show()

Q, z_mean = compute_Q(x, y, sigy, sigx)
v = principal_eigvec(Q)
plot_data_and_fit(x, y, sigy, sigx, v, z_mean)
