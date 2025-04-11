# """
# Reproducing the Figure 1 in David Hogg's paper "Data analysis recipes: Fitting a model to data"

# Not considering the first 4 data points which deviate a lot from the rest of the data points

# """



import matplotlib.pyplot as plt
import numpy as np


# Data
id = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x = np.array([201, 244, 47, 287, 203, 58, 210, 202, 198, 158, 165, 201, 157, 131, 166, 160, 186, 125, 218, 146])
y = np.array([592, 401, 583, 402, 495, 173, 479, 504, 510, 416, 393, 442, 317, 311, 400, 337, 423, 334, 533, 344])
sigy = np.array([61, 25, 38, 15, 21, 15, 27, 14, 30, 16, 14, 25, 52, 16, 34, 31, 42, 26, 16, 22])
sigx = np.array([9, 4, 11, 7, 5, 9, 4, 4, 11, 7, 5, 5, 5, 6, 6, 5, 9, 8, 6, 5])
rhoxy = np.array([-0.84, 0.31, 0.64, -0.27, -0.33, 0.67, -0.02, -0.05, -0.84, -0.69, 0.30, -0.46, -0.03, 0.50, 0.73, -0.52, 0.90, 0.40, -0.78, -0.56])


x = x[4:].reshape(-1, 1)
y = y[4:].reshape(-1, 1)

def calc_chi_sq(sigy):

    A = np.hstack((np.ones_like(x), x))
    C = np.diag(sigy**2)
    C_inv = np.linalg.inv(C)

    cov_matrix = np.linalg.inv(A.T @ C_inv @ A)

    def fit_curve(A, C_inv, cov_matrix, y):
        X = cov_matrix @ A.T@ C_inv @ y
        return X

    X = fit_curve(A, C_inv, cov_matrix, y)

    chi_sq = (y - A @ X).T @ C_inv @ (y - A @ X)

    return chi_sq

def plot():

    S = np.arange(10, 1500, 1)
    S_sqrt = np.sqrt(S)
    chi_sq = np.ones_like(S_sqrt, dtype=float)

    temp = np.outer(S_sqrt, np.ones_like(x))

    for i in range(len(S)):
        t = calc_chi_sq(temp[i]).item()
        chi_sq[i] = t


    N = len(x)
    dof = N - 2

    plt.plot(S, chi_sq, label='Chi-squared statistic')
    plt.axhline(y=dof, color='r', linestyle='--', label='Degrees of freedom')
    plt.xlabel('S')
    plt.ylabel('Chi-squared')
    plt.xlim(0, 1500)
    plt.ylim(6, 24)
    plt.title('Chi-squared statistic vs S')
    plt.legend()
    plt.grid()
    plt.savefig('chi_squared_vs_S.png', dpi=300)
    plt.savefig('chi_squared_vs_S.pdf', dpi=300)

    plt.xlim(900, 1000)
    plt.ylim(12, 16)
    plt.savefig('chi_squared_vs_S_zoomed.png', dpi=300)
    plt.savefig('chi_squared_vs_S_zoomed.pdf', dpi=300)
    plt.show()


if __name__ == "__main__":
    plot()