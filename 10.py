import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Simulated gene expression data: 100 samples and 1000 genes
np.random.seed(42)
gene_expression_data = np.random.rand(100, 1000)

# Applying PCA
pca = PCA(n_components=5)
principal_components = pca.fit_transform(gene_expression_data)

# Explained variance ratios
explained_variance = pca.explained_variance_ratio_

# Visualizing the variance captured by each principal component
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Variance Explained by Each Principal Component')
plt.show()

print("Explained Variance Ratios:", explained_variance)

# Challenges:
# 1. Biological interpretability of principal components.
# 2. Potential loss of smaller yet meaningful variations in the data.
