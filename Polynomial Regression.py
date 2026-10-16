import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Data load
df = pd.read_csv("Boston.csv")

# Remove an index-like columns
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Cleaning the data
print("Missing values before cleaning:")
print(df.isnull().sum())
df = df.dropna()
print("Missing values after cleaning:")
print(df.isnull().sum())

# independent and dependent variables
X = df[["rm"]]
y = df["medv"]

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#polynomial regression, degree = 2
degree = 2
model = make_pipeline(
    PolynomialFeatures(degree=degree, include_bias=False),
    LinearRegression()
)
model.fit(X_train, y_train)

# Test values prediction
y_pred = model.predict(X_test)

# 7. Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\\nPolynomial Regression (degree = 2)")
print(f"MAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R-squared: {r2:.3f}")

# 8. Print the polynomial regression equation
poly = model.named_steps["polynomialfeatures"]
linear_model = model.named_steps["linearregression"]
intercept = linear_model.intercept_
coef_rm, coef_rm2 = linear_model.coef_

print("\\nRegression equation:")
print(f"medv = {intercept:.3f} + ({coef_rm:.3f} * rm) "
      f"+ ({coef_rm2:.3f} * rm^2)")

# 9. Interpretation
print("\\nInterpretation:")
print(f"R-squared = {r2:.3f}; the model explains about {r2 * 100:.1f}% "
      "of the variation in test-set medv.")
print(f"MAE = {mae:.3f}, meaning average absolute prediction error is "
      f"about ${mae * 1000:.0f}.")
print("Compare these results with Question 1 and with polynomial degrees 3, 4, and 5.")

# actual test points and fitted polynomial curve
plt.figure(figsize=(8, 6))
plt.scatter(X_test["rm"], y_test, alpha=0.7, label="Actual test values")
x_curve = np.linspace(X["rm"].min(), X["rm"].max(), 300).reshape(-1, 1)
y_curve = model.predict(pd.DataFrame(x_curve, columns=["rm"]))
plt.plot(x_curve, y_curve, color="red", label="Polynomial regression (degree 2)")
plt.xlabel("Average number of rooms (rm)")
plt.ylabel("Median home value (medv, $1,000s)")
plt.title("Polynomial Regression (Degree 2)")
plt.legend()
plt.tight_layout()
plt.show()

# Comparing degrees 2 through 5 with linear

results = []
for d in range(1, 6):
    poly_model = make_pipeline(
        PolynomialFeatures(degree=d, include_bias=False),
        LinearRegression()
    )
    poly_model.fit(X_train, y_train)
    train_pred = poly_model.predict(X_train)
    test_pred = poly_model.predict(X_test)

    results.append({
        "Degree": d,
        "Train R2": r2_score(y_train, train_pred),
        "Test MAE": mean_absolute_error(y_test, test_pred),
        "Test MSE": mean_squared_error(y_test, test_pred),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
        "Test R2": r2_score(y_test, test_pred)
    })

results_df = pd.DataFrame(results)
results_df["Model"] = results_df["Degree"].map(
    lambda d: "Question 1: Linear" if d == 1 else f"Polynomial degree {d}"
)
print("\\nPerformance comparison (same train/test split):")
print(results_df[["Model", "Train R2", "Test MAE", "Test MSE",
                  "Test RMSE", "Test R2"]].round(3).to_string(index=False))

# Plotting test R-squared across degrees
plt.figure(figsize=(8, 5))
plt.plot(results_df["Degree"], results_df["Test R2"], marker="o")
plt.xticks(range(1, 6))
plt.xlabel("Polynomial degree")
plt.ylabel("Test R-squared")
plt.title("Test Performance by Polynomial Degree")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#Overfitting happens when a model learns the training data too closely and do not generalize the unseen data.
#to determine the overfitting occurs or not, we should compare training and testing performance and use cross-validation