import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# Decision tree visualization
model = DecisionTreeClassifier()
X = [[1], [2], [3]]  # Dummy data
y = [0, 1, 0]
model.fit(X, y)

# Visualize tree
dot_data = export_graphviz(model, out_file=None, feature_names=["Feature"], class_names=["No", "Yes"])
graph = graphviz.Source(dot_data)
graph.render("DecisionTree")
