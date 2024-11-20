from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# Simulated high-dimensional data
X = np.random.rand(100, 1024)
y = np.random.choice([0, 1], size=100)

# Dimensionality reduction
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)

# K-NN Model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_reduced, y)

accuracy = model.score(X_reduced, y)
print("Model Accuracy:", accuracy)
