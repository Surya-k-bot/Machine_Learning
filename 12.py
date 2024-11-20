import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Simulated data
np.random.seed(42)
X = np.random.rand(1000, 5)  # Features
y = np.array([0] * 950 + [1] * 50)  # Imbalanced target: 950 non-fraud, 50 fraud

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balancing the dataset using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Training a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# Predictions and evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Techniques to address imbalance:
# 1. Over-sampling with SMOTE or under-sampling.
# 2. Weighted loss functions in model training.
# 3. Generating synthetic samples to balance classes.
