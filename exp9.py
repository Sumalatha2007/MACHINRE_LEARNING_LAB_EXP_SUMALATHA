import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Input data (non-linear relationship)
X = np.array([[1], [2], [3], [4]])
y = [1, 4, 9, 16]

# ---------- Linear Regression ----------
linear_model = LinearRegression()
linear_model.fit(X, y)
linear_pred = linear_model.predict([[5]])

# ---------- Polynomial Regression ----------
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
poly_pred = poly_model.predict(poly.transform([[5]]))

print("Input Data:", X.flatten())
print("Actual Output:", y)
print("Linear Regression Prediction for 5:", linear_pred)
print("Polynomial Regression Prediction for 5:", poly_pred)
