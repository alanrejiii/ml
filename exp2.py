import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
df = fetch_openml(name="autoMpg", version=1, as_frame=True).frame
if 'mpg' in df.columns:
    target = 'mpg'
elif 'class' in df.columns:
    target = 'class'
else:
    print("Available columns:", df.columns)
    raise ValueError("MPG column not found!")
df = df[['displacement', target]]
df['displacement'] = pd.to_numeric(df['displacement'], errors='coerce')
df[target] = pd.to_numeric(df[target], errors='coerce')
df.dropna(inplace=True)
X = df[['displacement']]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
pr = LinearRegression()
pr.fit(X_train_poly, y_train)
y_pred_pr = pr.predict(X_test_poly)
print("Linear Regression")
print("MSE :", mean_squared_error(y_test, y_pred_lr))
print("R2  :", r2_score(y_test, y_pred_lr))
print("\nPolynomial Regression")
print("MSE :", mean_squared_error(y_test, y_pred_pr))
print("R2  :", r2_score(y_test, y_pred_pr))
x = np.linspace(X.min().values[0], X.max().values[0], 300).reshape(-1,1)
plt.figure(figsize=(8,5))
plt.scatter(X, y, color='gray', label='Actual Data')
plt.plot(x, lr.predict(x), color='blue', linewidth=2,
         label='Linear Regression')
plt.title(f"Linear Regression\nMSE = {mean_squared_error(y_test, y_pred_lr):.2f}    "
          f"R² = {r2_score(y_test, y_pred_lr):.2f}")
plt.xlabel("Engine Displacement")
plt.ylabel("Miles Per Gallon (MPG)")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(8,5))
plt.scatter(X, y, color='gray', label='Actual Data')
plt.plot(x, pr.predict(poly.transform(x)), color='red', linewidth=2,
         label='Polynomial Regression')
plt.title(f"Polynomial Regression\nMSE = {mean_squared_error(y_test, y_pred_pr):.2f}    "
          f"R² = {r2_score(y_test, y_pred_pr):.2f}")
plt.xlabel("Engine Displacement")
plt.ylabel("Miles Per Gallon (MPG)")
plt.legend()
plt.grid(True)
plt.show()