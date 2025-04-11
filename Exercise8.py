"""
Reproducing the Figure 1 in David Hogg's paper "Data analysis recipes: Fitting a model to data"

Not considering the first 4 data points which deviate a lot from the rest of the data points

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

# x = x[4:]
# y = y[4:]
# sigy = sigy[4:]

N = len(x)

def compute_best_fit(x, y, sigy):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    sigy = sigy

    A = np.hstack((np.ones_like(x), x))
    C = np.diag(sigy**2)
    C_inv = np.linalg.inv(C)

    cov_matrix = np.linalg.inv(A.T @ C_inv @ A)

    def fit_curve(A, C_inv, cov_matrix, y):
        X = cov_matrix @ A.T@ C_inv @ y
        return X

    b, m = fit_curve(A, C_inv, cov_matrix, y).flatten()


    db, dm = np.sqrt(np.diag(cov_matrix))
    cov = cov_matrix[0, 1]

    return b, m, db, dm, cov

def jackknife(x, y, sigy, params):
    b, m, _, _, _ = params

    b_jackknife = np.zeros(N)
    m_jackknife = np.zeros(N)

    for i in range(N):
        x_jackknife = np.delete(x, i)
        y_jackknife = np.delete(y, i)
        sigy_jackknife = np.delete(sigy, i)

        b_jackknife[i], m_jackknife[i], _, _, _ = compute_best_fit(x_jackknife, y_jackknife, sigy_jackknife)

    dif_b = b_jackknife - b
    dif_m = m_jackknife - m

    var_b = (N - 1) / N * np.sum(dif_b**2)
    var_m = (N - 1) / N * np.sum(dif_m**2)
    cov_b_m = (N - 1) / N * np.sum(dif_b * dif_m)

    sigma_b = np.sqrt(var_b)
    sigma_m = np.sqrt(var_m)

    return sigma_b, sigma_m, cov_b_m

def bootstrap(x, y, sigy, params, n_bootstrap=1000):
    b, m, db, dm, cov = params

    b_bootstrap = np.zeros(n_bootstrap)
    m_bootstrap = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        indices = np.random.choice(N, N, replace=True)
        x_bootstrap = x[indices]
        y_bootstrap = y[indices]
        sigy_bootstrap = sigy[indices]

        b_bootstrap[i], m_bootstrap[i], _, _, _ = compute_best_fit(x_bootstrap, y_bootstrap, sigy_bootstrap)

    dif_b = b_bootstrap - b
    dif_m = m_bootstrap - m

    var_b = np.sum((dif_b)**2) / (n_bootstrap)
    var_m = np.sum((dif_m)**2) / (n_bootstrap)
    cov_b_m = np.sum((dif_b) * (dif_m)) / (n_bootstrap)

    sigma_b = np.sqrt(var_b)
    sigma_m = np.sqrt(var_m)

    return sigma_b, sigma_m, cov_b_m

def main():
    # Compute the best fit
    params = compute_best_fit(x, y, sigy)
    b, m, db, dm, cov = params

    # Print the equation
    print(f"\ny = ({m:.2f} ± {dm:.2f})x + ({b:.2f} ± {db:.2f})\n")
    print("Covariance matrix:")
    print(f"[[{db**2:.2f}, {cov:.2f}]")
    print(f" [{cov:.2f}, {dm**2:.2f}]]")

    # Jackknife
    sigma_b_jackknife, sigma_m_jackknife, cov_b_m_jackknife = jackknife(x, y, sigy, params)
    print(f"\nJackknife errors:")
    print(f"b = {b:.2f} ± {sigma_b_jackknife:.2f}")
    print(f"m = {m:.2f} ± {sigma_m_jackknife:.2f}")
    print(f"Covariance = {cov_b_m_jackknife:.2f}")
    print(f"Correlation coefficient = {cov_b_m_jackknife / (sigma_b_jackknife * sigma_m_jackknife):.2f}")
    print(f"Correlation coefficient (from covariance matrix) = {cov / (db * dm):.2f}")
    
    # Bootstrap
    sigma_b_bootstrap, sigma_m_bootstrap, cov_b_m_bootstrap = bootstrap(x, y, sigy, params)
    print(f"\nBootstrap errors:")
    print(f"b = {b:.2f} ± {sigma_b_bootstrap:.2f}")
    print(f"m = {m:.2f} ± {sigma_m_bootstrap:.2f}")
    print(f"Covariance = {cov_b_m_bootstrap:.2f}")
    print(f"Correlation coefficient = {cov_b_m_bootstrap / (sigma_b_bootstrap * sigma_m_bootstrap):.2f}")
    print(f"Correlation coefficient (from covariance matrix) = {cov / (db * dm):.2f}")




if __name__ == "__main__":
    main()
