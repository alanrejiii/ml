import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


housing = fetch_california_housing(as_frame=True)


X = housing.data[['AveRooms']].values
y = housing.target.values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


mean = X_train.mean()
std = X_train.std()

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std



m = len(X_train_scaled)


w = 0
b = 0

learning_rate = 0.01
epochs = 1000


for epoch in range(epochs):

    y_pred = w * X_train_scaled + b

    dw = (1/m) * np.sum((y_pred - y_train.reshape(-1,1)) * X_train_scaled)
    db = (1/m) * np.sum(y_pred - y_train.reshape(-1,1))

    w -= learning_rate * dw
    b -= learning_rate * db


gd_predictions = w * X_test_scaled + b


gd_mse = mean_squared_error(y_test, gd_predictions)
gd_r2 = r2_score(y_test, gd_predictions)

plt.figure(figsize=(8,6))

plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Data')

sorted_index = np.argsort(X_test[:,0])

plt.plot(
    X_test[sorted_index],
    gd_predictions[sorted_index],
    color='red',
    linewidth=2,
    label='Gradient Descent Line'
)


plt.text(
    0.02, 1.05,
    f"MSE = {gd_mse:.4f}\nR² = {gd_r2:.4f}",
    transform=plt.gca().transAxes,
    fontsize=11,
    bbox=dict(facecolor='white', edgecolor='black')
)

plt.xlabel("Average Number of Rooms (AveRooms)")
plt.ylabel("Median House Value")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.grid(True)

plt.show()


print("Gradient Descent Results")
print("------------------------")
print("Weight:", w)
print("Bias:", b)
print("MSE:", gd_mse)
print("R² Score:", gd_r2)


X_train_ne = np.c_[np.ones((len(X_train),1)), X_train]
X_test_ne = np.c_[np.ones((len(X_test),1)), X_test]

theta = np.linalg.inv(X_train_ne.T @ X_train_ne) @ X_train_ne.T @ y_train

normal_predictions = X_test_ne @ theta

normal_mse = mean_squared_error(y_test, normal_predictions)
normal_r2 = r2_score(y_test, normal_predictions)

print("\nNormal Equation Results")
print("------------------------")
print("Intercept:", theta[0])
print("Slope:", theta[1])
print("MSE:", normal_mse)
print("R² Score:", normal_r2)



plt.figure(figsize=(8,6))

plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Data')

sorted_index = np.argsort(X_test[:,0])

plt.plot(
    X_test[sorted_index],
    normal_predictions[sorted_index],
    color='green',
    linewidth=2,
    label='Normal Equation Line'
)


plt.text(
    0.02, 1.05,
    f"MSE = {normal_mse:.4f}\nR² = {normal_r2:.4f}",
    transform=plt.gca().transAxes,
    fontsize=11,
    bbox=dict(facecolor='white', edgecolor='black')
)

plt.xlabel("Average Number of Rooms (AveRooms)")
plt.ylabel("Median House Value")
plt.title("Linear Regression using Normal Equation")
plt.legend()
plt.grid(True)

plt.show()