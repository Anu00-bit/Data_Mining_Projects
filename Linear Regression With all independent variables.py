import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Data loading
df = pd.read_csv("Boston.csv")

print(df.head())
print(df.info())

# Check and handle missing values
print("Missing values-:  before cleaning:\n")
print(df.isnull().sum())

df = df.dropna()

print("Missing values:- after cleaning:\n")
print(df.isnull().sum())

# independent and dependent variables
X = df.drop(columns=["medv"])
y = df["medv"]

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

#Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting with the test dataset
y_pred = model.predict(X_test)

print("Predicted values:")
print(y_pred)

print("Actual values:")
print(y_test.values)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", round(mae, 3))
print("Mean Squared Error (MSE):", round(mse, 3))
print("Root Mean Squared Error (RMSE):", round(rmse, 3))
print("R-squared (R2):", round(r2, 3))

# the regression equation
intercept = model.intercept_
coefficients = model.coef_

equation = f"medv = {intercept:.3f}"

for name, coef in zip(X.columns, coefficients):
    equation += f" + ({coef:.3f} * {name})"

print("Regression Equation:")
print(equation)

# 10. Plotting actual and predicted values
plt.scatter(y_test, y_pred, alpha=0.7)

low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())

plt.plot(
    [low, high],
    [low, high],
    color="red",
    linestyle="--",
    label="Perfect predictions"
)

plt.xlabel("Actual medv ($1,000s)")
plt.ylabel("Predicted medv ($1,000s)")
plt.title("Linear Regression: Actual vs. Predicted Home Values")
plt.legend()
plt.show()


# Predicting a home's value using only the number of rooms
#Here I used the number of rooms plus other housing-related factors
# as these additional factors provide useful information this model
# made more accurate predictions on  test dataset
