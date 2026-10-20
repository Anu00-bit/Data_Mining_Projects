import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)

# Loading data
df = pd.read_csv("titanic.csv")
print("First five rows:")
print(df.head())
print("\nDataset information:")
print(df.info())

# Cleaning
print("\nMissing values before cleaning:")
print(df.isnull().sum())

df = df.dropna()

print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("\nShape after cleaning:", df.shape)

# 3. Encode Sex: male = 0, female = 1
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# PassengerId is excluded because it is an identifier, not a passenger attribute.
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

# training and testing sets data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 5. Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# test dataset classification
y_pred = model.predict(X_test)

print("\nPredicted survival values:")
print(y_pred)
print("\nActual survival values:")
print(y_test.to_numpy())

# model performance evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nModel Evaluation:")
print(f"Accuracy:  {accuracy:.3f} ({accuracy * 100:.1f}%)")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")
print("\nConfusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# logistic regression equation
intercept = model.intercept_[0]
coefficients = model.coef_[0]

equation = f"log(p / (1 - p)) = {intercept:.3f}"
for name, coef in zip(X.columns, coefficients):
    sign = "+" if coef >= 0 else "-"
    equation += f" {sign} {abs(coef):.6f} * {name}"

print("\nLogistic regression equation:")
print(equation)
print("where p is the predicted probability that Survived = 1.")
print("Probability equation: p = 1 / (1 + exp(-z)), where z is the equation above.")