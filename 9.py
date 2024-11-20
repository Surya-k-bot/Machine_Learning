from sklearn.metrics import r2_score

# Simulated data
y_true = [200, 210, 250, 300, 330]
y_pred = [195, 215, 245, 290, 320]

# R-squared
r_squared = r2_score(y_true, y_pred)
print("R-squared Value:", r_squared)
