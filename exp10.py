from sklearn.mixture import GaussianMixture
import numpy as np

# Input data (directly given)
X = np.array([[1], [2], [3], [10], [11], [12]])

# Create EM model with 2 clusters
model = GaussianMixture(n_components=2, random_state=0)

# Train the model
model.fit(X)

# Predict cluster labels
labels = model.predict(X)

print("Input Data:", X.flatten())
print("Cluster Labels:", labels)
