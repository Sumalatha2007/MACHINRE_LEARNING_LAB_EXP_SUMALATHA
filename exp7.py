from sklearn.linear_model import LogisticRegression

# Input data (directly given)
X = [[1], [2], [3], [4], [5]]
y = [0, 0, 0, 1, 1]

# Create Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X, y)

# Test input
test_data = [[3]]

# Prediction
prediction = model.predict(test_data)

print("Training Data:", X)
print("Class Labels:", y)
print("Test Input:", test_data)
print("Predicted Output:", prediction)
