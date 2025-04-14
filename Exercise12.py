import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner


# Data
id = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x = np.array([201, 244, 47, 287, 203, 58, 210, 202, 198, 158, 165, 201, 157, 131, 166, 160, 186, 125, 218, 146])
y = np.array([592, 401, 583, 402, 495, 173, 479, 504, 510, 416, 393, 442, 317, 311, 400, 337, 423, 334, 533, 344])
# assuming sigyi are the same for all points
# sigy = np.array([61, 25, 38, 15, 21, 15, 27, 14, 30, 16, 14, 25, 52, 16, 34, 31, 42, 26, 16, 22])
sigx = np.array([9, 4, 11, 7, 5, 9, 4, 4, 11, 7, 5, 5, 5, 6, 6, 5, 9, 8, 6, 5])
rhoxy = np.array([-0.84, 0.31, 0.64, -0.27, -0.33, 0.67, -0.02, -0.05, -0.84, -0.69, 0.30, -0.46, -0.03, 0.50, 0.73, -0.52, 0.90, 0.40, -0.78, -0.56])


# Remove the first 4 elements
id = id[4:]
x = x[4:]
y = y[4:]


def logprior(params):
    m, b = params
    if 0 <= m <= 5 and -200 <= b <= 200:
        return 0
    return -np.inf


def loglikelihood(params, xi, yi, sigyi):
    """
    Calculate the log likelihood of the data given the model parameters and noise parameters.
    """
    # Unpack the parameters
    m, b = params

    if np.any(sigyi <= 0):
        print("sigyi is not positive")
        return -np.inf
    return np.sum(-0.5 * ((yi - (m * xi + b)) / sigyi)**2)

def logposterior(params, xi, yi, sigyi):
    lp = logprior(params)
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + loglikelihood(params, xi, yi, sigyi) 

