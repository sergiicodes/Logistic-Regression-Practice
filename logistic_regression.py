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
plt.show()

# Fit a logistic regression model to the data
model = LogisticRegression()
model.fit(X, y)

# Predict the class of a new data point
new_point = np.array([[0.5, 0.5]])
predicted_class = model.predict(new_point)
print("Predicted class:", predicted_class)
