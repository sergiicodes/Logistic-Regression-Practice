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

# Extract the coefficients and intercept as scalars
coef = model.coef_[0]
intercept = model.intercept_[0]

# Calculate the slope and intercept of the decision boundary
slope = -coef[0] / coef[1]
intercept = -intercept / coef[1]

# Print the equation of the decision boundary
eqn = f"Decision boundary: y = {slope:.2f}x + {intercept:.2f}"
print(eqn)

# Plot the decision boundary
x_vals = np.array([X[:, 0].min(), X[:, 0].max()])
y_vals = slope * x_vals + intercept
plt.plot(x_vals, y_vals, '--', color='black')

# Add the equation text to the plot
plt.text(0.05, 1.1, eqn, transform=plt.gca().transAxes)

# Show the plot
plt.show()
