"""Doesnt work for my function!!!"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Your data
x_data = np.array([
    0.307, 0.502, 0.717, 0.914, 1.103, 1.302, 1.503, 1.706,
    1.977, 3.877, 4.352, 4.922, 5.999
])
y_data = np.array([
    -5.152, -4.943, -4.27, -4.055, -4.071, -3.82, -3.777,
    -4.015, -3.471, -3.877, -3.769, -3.873, -4.415
])

# Smooth piecewise linear function with logistic transition
def smooth_piecewise(x, f, m, a, b):
    S = 1 / (1 + 10**(-(x - m)))  # logistic blend with base-10
    return f + (x - m) * (a - (a - b) * S)

def smooth_function(x, m1, c1, m2, c2):
    k = 1
    a = - (c2 - c1) / (m2 - m1)
    w = 1 / (1 + np.exp(-k * (x - a)))
    y1 = m1 * x + c1
    y2 = m2 * x + c2
    
    return w * y1 + (1 - w) * y2

# Initial guess for [f, m, a, b]
# p0 = [-4, 2.5, 1, -0.5]
p0 = [2, 1, 1, 4]

# left side slope from fig = 0.97
# right side slope from fig = -0.34


# Fit the model
# params, _ = curve_fit(smooth_piecewise, x_data, y_data, p0=p0)
params, _ = curve_fit(smooth_function, x_data, y_data, p0=p0)

# Generate smooth curve
x_fit = np.linspace(min(x_data) - 5, max(x_data) + 5, 500)
# y_fit = smooth_piecewise(x_fit, *params)
y_fit = smooth_function(x_fit, *params)
y_initial = smooth_function(x_fit, *p0)

# Print fitted parameters
# print(f"Fitted parameters:\n  f = {params[0]:.3f}\n  m = {params[1]:.3f}\n  a = {params[2]:.3f}\n  b = {params[3]:.3f}")
print(f"Fitted parameters:\n  m1 = {params[0]:.3f}\n  c1 = {params[1]:.3f}\n  m2 = {params[2]:.3f}\n  c2 = {params[3]:.3f}")


a = - (params[3] - params[1]) / (params[2] - params[0])
# Plot
plt.figure(figsize=(8, 5))
plt.scatter(x_data, y_data, label='Data', color='black')
plt.plot(x_fit, y_fit, label='Fitted smooth linear blend', color='red')
# plt.plot(x_fit, y_initial, label='Initial guess', color='blue', linestyle='--')
plt.axvline(a, color='gray', linestyle='--', label=f'a = {params[1]:.2f}')

x_arr1 = np.linspace(min(x_fit), a, 100)
x_arr2 = np.linspace(a, max(x_fit), 100)
y_arr1 = params[0] * x_arr2 + params[1]
y_arr2 = params[2] * x_arr1 + params[3]
plt.plot(x_arr1, y_arr1, color='blue', linestyle='--', label='Left side fit')
plt.plot(x_arr2, y_arr2, color='green', linestyle='--', label='Right side fit')
plt.title('Smooth Piecewise Linear Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
