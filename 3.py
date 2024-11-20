from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Example metrics for decision tree
y_true = [0, 1, 0, 1, 0, 1, 0, 1]  # Actual outcomes
y_pred = [0, 1, 0, 0, 0, 1, 1, 1]  # Predicted outcomes

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
