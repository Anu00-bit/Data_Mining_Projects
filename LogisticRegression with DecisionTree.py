import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

#Loading and cleaning data
df = pd.read_csv("titanic.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
print("Missing values before cleaning:\\n", df.isnull().sum())
df = df.dropna()
print("\\nMissing values after cleaning:\\n", df.isnull().sum())

#Encode sex and use same variables
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

# 80/20 split and random state
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def evaluate(name, y_true, y_pred):
    print(f"\\n{name}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print("Confusion matrix (actual rows, predicted columns):")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, zero_division=0))

# logistic regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
evaluate("Logistic Regression", y_test, log_pred)

# Training decision tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Classifying test data
tree_pred = tree.predict(X_test)
print("\\nPredicted values:\\n", tree_pred)
print("Actual values:\\n", y_test.to_numpy())

# Evaluation
evaluate("Unpruned Decision Tree", y_test, tree_pred)

# textual tree
print("\\nDecision tree:")
print(export_text(tree, feature_names=list(X.columns)))

# plot tree
plt.figure(figsize=(18, 9))
plot_tree(tree, feature_names=list(X.columns),
          class_names=["Did not survive", "Survived"],
          filled=True, rounded=True, fontsize=8)
plt.title("Titanic Decision Tree (Unpruned)")
plt.tight_layout()
plt.savefig("titanic_decision_tree.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

# Comparing
comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Logistic Regression": [
        accuracy_score(y_test, log_pred),
        precision_score(y_test, log_pred, zero_division=0),
        recall_score(y_test, log_pred, zero_division=0),
        f1_score(y_test, log_pred, zero_division=0)
    ],
    "Decision Tree": [
        accuracy_score(y_test, tree_pred),
        precision_score(y_test, tree_pred, zero_division=0),
        recall_score(y_test, tree_pred, zero_division=0),
        f1_score(y_test, tree_pred, zero_division=0)
    ]
})
print("\\nPerformance comparison:")
print(comparison.round(3).to_string(index=False))

# cost-complexity pruning, choosing alpha on validation data
X_sub, X_val, y_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_sub, y_sub)
candidate_results = []
for alpha in np.unique(path.ccp_alphas):
    candidate = DecisionTreeClassifier(random_state=42, ccp_alpha=float(alpha))