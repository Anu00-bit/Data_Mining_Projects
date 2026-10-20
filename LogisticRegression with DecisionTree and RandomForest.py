import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

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

    candidate.fit(X_sub, y_sub)
    val_pred = candidate.predict(X_val)

    candidate_results.append({
        "alpha": float(alpha),
        "validation_f1": f1_score(y_val, val_pred, zero_division=0),
        "depth": candidate.get_depth(),
        "leaves": candidate.get_n_leaves()
    })

candidate_df = pd.DataFrame(candidate_results)

# Choosing the best pruning alpha using validation F1-score
best = candidate_df.sort_values(
    ["validation_f1", "depth", "leaves"],
    ascending=[False, True, True]
).iloc[0]

best_alpha = float(best["alpha"])

# Training the pruned tree
pruned_tree = DecisionTreeClassifier(
    random_state=42,
    ccp_alpha=best_alpha
)
pruned_tree.fit(X_train, y_train)
pruned_pred = pruned_tree.predict(X_test)

evaluate("Pruned Decision Tree", y_test, pruned_pred)

print("Selected alpha:", best_alpha)
print("Unpruned depth:", tree.get_depth())
print("Unpruned leaves:", tree.get_n_leaves())
print("Pruned depth:", pruned_tree.get_depth())
print("Pruned leaves:", pruned_tree.get_n_leaves())


# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predicting test data
rf_pred = rf_model.predict(X_test)

# Evaluating Random Forest
evaluate("Random Forest", y_test, rf_pred)

#Comparing
comparison_all = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],

    "Logistic Regression": [
        accuracy_score(y_test, log_pred),
        precision_score(y_test, log_pred, zero_division=0),
        recall_score(y_test, log_pred, zero_division=0),
        f1_score(y_test, log_pred, zero_division=0)
    ],

    "Unpruned Decision Tree": [
        accuracy_score(y_test, tree_pred),
        precision_score(y_test, tree_pred, zero_division=0),
        recall_score(y_test, tree_pred, zero_division=0),
        f1_score(y_test, tree_pred, zero_division=0)
    ],

    "Pruned Decision Tree": [
        accuracy_score(y_test, pruned_pred),
        precision_score(y_test, pruned_pred, zero_division=0),
        recall_score(y_test, pruned_pred, zero_division=0),
        f1_score(y_test, pruned_pred, zero_division=0)
    ],

    "Random Forest": [
        accuracy_score(y_test, rf_pred),
        precision_score(y_test, rf_pred, zero_division=0),
        recall_score(y_test, rf_pred, zero_division=0),
        f1_score(y_test, rf_pred, zero_division=0)
    ]
})

print("\nFinal Model Performance Comparison:")
print(comparison_all.round(3).to_string(index=False))


#The Random Forest model has an accuracy of 76.2%, precision of 69.6%, recall of 69.6%, and an F1-score of 69.6%.
# In comparison, the decision tree model has achieved an accuracy of 71.3% and an F1-score of 63.1%.
# Therefore, Random Forest improved accuracy by 4.9 percentage points and performed better across all four evaluation metrics.

#Random Forest combines predictions from multiple decision trees, which can reduce the impact of overfitting associated with an individual tree.
# Combining multiple trees improved the model's ability to predict passenger survival on  test dataset.

#In conclusion, Random Forest performed better than the single decision tree.
# However, its performance should be evaluated on additional data to determine how well it can generalize to unseen passengers.
