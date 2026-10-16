# Used LinearRegression() to train the model and used trained model to predict for test dataset.
# After Evaluating the performance of the model got regression equation
# Interpreting evaluation and a plot to represent your linear regression model.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Boston.csv")
print(df.head())

# Removing index-like column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Cleaning data: missing values only
print("Missing values:", df.isnull().sum().sum())
df = df.dropna()

# independent and dependent variables
X = df[["rm"]]
y = df["medv"]

# 4. Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting test values
y_pred = model.predict(X_test)

# Evaluating performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print ("\nModel Evaluation: ")
print("MAE:", round(mae, 3))
print("MSE:", round(mse, 3))
print("RMSE:", round(rmse, 3))
print("R-squared:", round(r2, 3))

# Printing regression equation
print(
    f"medv = {model.intercept_:.3f} "
    f"+ ({model.coef_[0]:.3f} * rm)"
)

#  results
print(f"R-squared explains {r2 * 100:.1f}% of the variation.")
print(f"Mean absolute error: ${mae:.2f} thousand.")

# 10.regression model
plt.scatter(X_test["rm"], y_test, label="Actual values")

x_line = np.linspace(X_test["rm"].min(),
                     X_test["rm"].max(), 200)

y_line = model.predict(pd.DataFrame({"rm": x_line}))
plt.plot(x_line, y_line, color="red", label="Regression line")

plt.xlabel("Average number of rooms (rm)")
plt.ylabel("Median home value (medv, $1,000s)")
plt.title("Linear Regression: Rooms vs. Home Value")
plt.legend()
plt.show()