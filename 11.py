import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Simulated historical weather data (days and temperatures)
days = np.arange(1, 366).reshape(-1, 1)  # 1 year (365 days)
temperatures = 20 + 10 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 2, size=days.shape)

# Simple Linear Regression (without seasonality)
model_simple = LinearRegression()
model_simple.fit(days, temperatures)
predictions_simple = model_simple.predict(days)
mse_simple = mean_squared_error(temperatures, predictions_simple)

# Linear Regression with Seasonal Adjustments
days_seasonal = np.hstack((days, np.sin(2 * np.pi * days / 365), np.cos(2 * np.pi * days / 365)))
model_seasonal = LinearRegression()
model_seasonal.fit(days_seasonal, temperatures)
predictions_seasonal = model_seasonal.predict(days_seasonal)
mse_seasonal = mean_squared_error(temperatures, predictions_seasonal)

# Results
print("MSE without seasonality:", mse_simple)
print("MSE with seasonality:", mse_seasonal)

# Plotting results
plt.figure(figsize=(12, 6))
plt.plot(days, temperatures, label="Actual Temperatures")
plt.plot(days, predictions_simple, label="Simple Linear Regression", linestyle="--")
plt.plot(days, predictions_seasonal, label="Seasonal Adjusted Regression", linestyle=":")
plt.legend()
plt.xlabel("Days")
plt.ylabel("Temperature")
plt.title("Weather Prediction with and without Seasonality")
plt.show()
