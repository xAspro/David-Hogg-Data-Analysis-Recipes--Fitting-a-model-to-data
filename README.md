# 📘 David Hogg (2010) — *Data Analysis Recipes: Fitting a Model to Data*  
### **Complete Python Solutions to All 18 Exercises**

This repository contains my full Python implementations of all exercises from David W. Hogg’s 2010 paper:

**“Data Analysis Recipes: Fitting a Model to Data.”**

Each exercise appears as a standalone Python script.  
All scripts contain their own data internally—no external files are required.

The repository illustrates a complete progression from classical least-squares fitting to full Bayesian inference with intrinsic scatter.

---

## 📂 Repository Layout

Each `Exercise*.py` file corresponds to one of the exercises described in the paper.  
The exception is `Exercise4` and `Exercise5`, which are provided as PDF files.

---

## 🧠 Summary of Exercises Implemented

### **Exercise 1**  
Weighted least-squares line fit for points 5–20 (ignoring σx and correlations).  
Compute slope uncertainty and reproduce Figure 1.

### **Exercise 2**  
Repeat Exercise 1 using *all* points.  
Compare slope uncertainty and discuss issues introduced by the extra points.

### **Exercise 3**  
Extend the linear model to a quadratic fit \( g(x) = qx^2 + mx + b \).  
Plot and compare to Figure 3.

### **Exercise 4**  
Derive the log-likelihood for repeated measurements of a single true quantity.  
Show that the MLE is the weighted mean.

### **Exercise 5**  
Use matrix calculus to show that minimizing  
\( \chi^2 = (y - AX)^T C^{-1} (y - AX) \)  
gives the normal equation \( X = (A^T C^{-1} A)^{-1} A^T C^{-1} y \).

### **Exercise 6**  
Fit a line using a Bayesian mixture model to handle outliers.  
Sample the five-dimensional parameter space and marginalize over nuisance parameters.  
Plot the 2D posterior \((m, b)\) and compare with the classical fit.

### **Exercise 7**  
Fully marginalize over \((m, b, Y_b, V_b)\) to obtain the posterior for \( P_b \).  
Repeat after reducing σ by a factor of two and compare results.

### **Exercise 8**  
Compute slope uncertainty from Exercise 2, and compare with jackknife and bootstrap uncertainties.

### **Exercise 9**  
Repeat the mixture-model fit for points 5–20 and again with reduced uncertainties.  
Plot marginalized posteriors for \((m, b)\) and compare their sizes.

### **Exercise 10**  
Compute χ² and assess goodness of fit for Exercises 1 and 2.

### **Exercise 11**  
Repeat Exercise 1 assuming all data points have the same variance \(S\).  
Find the value of \(S\) that makes χ² = N − 2.  
Compare with the mean/median of the original variances.

### **Exercise 12**  
Write the full Bayesian inference and marginalization for the straight-line model with a common variance \(S\).  
Compute the MAP line and MAP error bars.  
Also compute results after marginalizing over \(S\).

### **Exercise 13**  
Fit a line using full 2D uncertainties (σx, σy, σxy).  
Plot error ellipses and the best-fit line.

### **Exercise 14**  
Repeat Exercise 13 for all data points.  
Add a robust Bayesian mixture model for bad points.  
Plot best-fit lines and a sample of posterior lines.

### **Exercise 15**  
Perform the forward–reverse fitting procedure.  
Fit y(x) using σy, then x(y) using σx, and plot both lines.  
Comment on why this method is flawed.

### **Exercise 16**  
Perform PCA on points 5–20 and derive the principal direction of the data cloud.  
Plot the PCA line and comment on PCA method.

### **Exercise 17**  
Fit a line allowing for orthogonal intrinsic scatter \(V\).  
Plot the ±√V envelope around the maximum-likelihood relation.

### **Exercise 18**   
Sample the posterior in \((\theta, b_\perp, V)\), marginalize over the line parameters,  
and plot the posterior for \(V\) with 95% and 99% upper limits.  
Explain why only upper limits are requested.

---

## 🎯 Purpose of This Repository

This project provides a complete worked solution set to the Hogg (2010) paper, demonstrating:

- Classical linear and nonlinear fitting  
- Correct handling of measurement uncertainties  
- Mixture models for outlier rejection  
- PCA-based geometric interpretation  
- Likelihood-based inference  
- Full Bayesian methods using MCMC  
- Proper marginalization and extraction of posterior limits  

The scripts serve as a reference for students and researchers learning rigorous statistical modeling for scientific data.

---

## 📖 Reference

David W. Hogg (2010).  
*Data Analysis Recipes: Fitting a Model to Data.*  
arXiv:1008.4686

