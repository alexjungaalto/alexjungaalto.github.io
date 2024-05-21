#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:12:06 2024

@author: junga1
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Set random seed for reproducibility
np.random.seed(42)

# Generate data for each class
X0, _ = make_blobs(n_samples=100, centers=[[-2, -2]], cluster_std=0.8)
X1, _ = make_blobs(n_samples=10, centers=[[2, 2]], cluster_std=0.8)
X2, _ = make_blobs(n_samples=100, centers=[[0, 0]], cluster_std=1.5)

# Combine the datasets
X = np.vstack([X0, X1, X2])
y = np.array([0]*100 + [1]*10 + [2]*100)  # Labels

# Create an OvR logistic regression model
model = OneVsRestClassifier(LogisticRegression())

# Create a multinomial logistic regression model
#model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
#multinomial_model.fit(X, y)

model.fit(X, y)

# Create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict each point on the mesh
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], color='green', label='Class 2')
plt.title('Toy Dataset for Logistic Regression Comparison')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.contourf(xx, yy, Z, alpha=0.4,cmap=cmap_light)
plt.legend()
plt.grid(True)
plt.show()