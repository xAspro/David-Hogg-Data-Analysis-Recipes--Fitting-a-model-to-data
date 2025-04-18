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


def run_exercise(x, y, sigy):

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)



    A = np.hstack((np.ones_like(x), x))
    # print("A = ", A)
    C = np.diag(sigy**2)
    C_inv = np.linalg.inv(C)

    cov_matrix = np.linalg.inv(A.T @ C_inv @ A)
    # print(cov_matrix)

    def fit_curve(A, C_inv, cov_matrix, y):
        X = cov_matrix @ A.T@ C_inv @ y
        return X

    X = fit_curve(A, C_inv, cov_matrix, y)
    b, m = X.flatten()

    print("b = ", b)
    print("m = ", m)

    db, dm = np.sqrt(np.diag(cov_matrix))
    # print("db = ", db)
    # print("dm = ", dm)


    # Print the equation
    print(f"\ny = ({m:.2f} ± {dm:.2f})x + ({b:.2f} ± {db:.2f})\n")

    # Calculate the chi-squared statistic
    chi_sq = (y - A @ X).T @ C_inv @ (y - A @ X)
    return chi_sq


chi_sq = run_exercise(x, y, sigy)
print(f"chi_sq = {chi_sq.item():.2f}")


id = id[4:]
x = x[4:]
y = y[4:]
sigy = sigy[4:]

chi_sq2 = run_exercise(x, y, sigy)
print(f"chi_sq2 = {chi_sq2.item():.2f}")

print('\n')

if chi_sq2 < chi_sq:
    print("The chi-squared statistic is lower when the first 4 data points are removed.")
else:
    print("The chi-squared statistic is higher when the first 4 data points are removed.")

print()