from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Simulated data
X = np.random.rand(100, 20)  # Features
y = np.random.choice([0, 1], size=100)  # Spam=1, Not Spam=0

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear SVM
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)

# Non-linear SVM
nonlinear_svm = SVC(kernel='rbf')
nonlinear_svm.fit(X_train, y_train)

print("Linear SVM Accuracy:", linear_svm.score(X_test, y_test))
print("Non-linear SVM Accuracy:", nonlinear_svm.score(X_test, y_test))
