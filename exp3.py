import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


linear_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

linear_mse = mean_squared_error(y_test, linear_pred)
linear_r2 = r2_score(y_test, linear_pred)


ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])

ridge_alphas = np.logspace(-3, 3, 20)

ridge_grid = GridSearchCV(
    ridge_pipeline,
    {'model__alpha': ridge_alphas},
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True
)

ridge_grid.fit(X_train, y_train)

ridge_pred = ridge_grid.predict(X_test)

ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)


lasso_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso(max_iter=10000))
])

lasso_alphas = np.logspace(-3, 1, 20)

lasso_grid = GridSearchCV(
    lasso_pipeline,
    {'model__alpha': lasso_alphas},
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True
)

lasso_grid.fit(X_train, y_train)

lasso_pred = lasso_grid.predict(X_test)

lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)


results = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression'],
    'Best Alpha': ['-', ridge_grid.best_params_['model__alpha'],
                   lasso_grid.best_params_['model__alpha']],
    'MSE': [linear_mse, ridge_mse, lasso_mse],
    'R2 Score': [linear_r2, ridge_r2, lasso_r2]
})

print("\nPerformance Comparison\n")
print(results)

print("\nBest Ridge Alpha:", ridge_grid.best_params_['model__alpha'])
print("Best Lasso Alpha:", lasso_grid.best_params_['model__alpha'])


plt.figure(figsize=(7,5))
plt.bar(results['Model'], results['MSE'],
        color=['skyblue','orange','green'])

plt.title("Model Comparison - Mean Squared Error")
plt.ylabel("MSE")
plt.grid(axis='y')
plt.show()


plt.figure(figsize=(7,5))
plt.bar(results['Model'], results['R2 Score'],
        color=['skyblue','orange','green'])

plt.title("Model Comparison - R2 Score")
plt.ylabel("R2 Score")
plt.ylim(0,1)
plt.grid(axis='y')
plt.show()