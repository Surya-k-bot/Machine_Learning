import pandas as pd
from sklearn.linear_model import LogisticRegression

# Simulated data
data = {
    "Credit_Score": [600, 650, 700, 550, 720, 500, 580],
    "Income_Level": [40000, 50000, 60000, 30000, 70000, 25000, 35000],
    "Employment_Status": [1, 1, 1, 0, 1, 0, 0],  # 1=Employed, 0=Unemployed
    "Default": [0, 0, 0, 1, 0, 1, 1]  # 0=No Default, 1=Default
}

df = pd.DataFrame(data)

# Features and target
X = df[["Credit_Score", "Income_Level", "Employment_Status"]]
y = df["Default"]

# Model
model = LogisticRegression()
model.fit(X, y)

# Coefficients interpretation
coefficients = model.coef_[0]
print("Coefficients:", coefficients)

# Interpretation: Positive coefficients increase default probability.
