import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Simulated data
data = {
    "Study_Hours": [2, 3, 4, 5, 1, 6, 7, 3, 8, 4],
    "Attendance_Rate": [80, 85, 78, 92, 70, 88, 95, 75, 90, 82],
    "Socioeconomic_Score": [50, 55, 40, 60, 30, 65, 70, 45, 75, 55],
    "Test_Score": [65, 70, 68, 75, 58, 80, 85, 62, 88, 72]
}

# DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[["Study_Hours", "Attendance_Rate", "Socioeconomic_Score"]]
y = df["Test_Score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Coefficients interpretation
coefficients = model.coef_
print("Feature Coefficients:", coefficients)
