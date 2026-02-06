from sklearn.linear_model import LinearRegression

# Input data (directly given)
X = [[1], [2], [3], [4]]
y = [2, 4, 6, 8]

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Test input
test_data = [[5]]

# Prediction
prediction = model.predict(test_data)

print("Training Data:", X)
print("Output Values:", y)
print("Test Input:", test_data)
print("Predicted Output:", prediction)
