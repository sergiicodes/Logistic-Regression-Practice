import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# Generate some example data
np.random.seed(0)
n_samples = 100
X = np.random.normal(size=(n_samples, 2))
y = (X[:, 0] + X[:, 1] > 0).astype(np.int)

# Scatter plot the data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Fit a logistic regression model to the data
model = LogisticRegression()
model.fit(X, y)

# Get the coefficients of the logistic regression model
coef = model.coef_[0]
intercept = model.intercept_

# Compute the slope and intercept of the line that separates the two classes
x1 = X[:, 0].min()
x2 = X[:, 0].max()
y1 = -(intercept + coef[0] * x1) / coef[1]
y2 = -(intercept + coef[0] * x2) / coef[1]

# Add the dividing line to the plot
plt.plot([x1, x2], [y1, y2], color='black')

plt.show()

slope = -(intercept / coef[1])
intercept = -(coef[0] / coef[1])
print(f"Equation of line:  y = {slope}x{intercept}")
