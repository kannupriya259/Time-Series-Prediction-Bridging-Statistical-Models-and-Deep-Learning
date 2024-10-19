# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('air_quality_data.csv', parse_dates=['date'], index_col='date')
data = data.resample('H').mean()  # Resampling to hourly frequency if needed
data.dropna(inplace=True)

# Display first few rows of the dataset
data.head()
# Plotting the data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['value'])  # Replace 'value' with the actual column name
plt.title('Air Quality Over Time')
plt.xlabel('Date')
plt.ylabel('Air Quality Index')
plt.show()

# Checking for stationarity
from statsmodels.tsa.stattools import adfuller

result = adfuller(data['value'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
import pmdarima as pm

# Automatically determine the p, d, and q parameters
model = pm.auto_arima(data['value'], seasonal=False, stepwise=True)
print(model.summary())
# Fit the ARIMA model
arima_model = ARIMA(data['value'], order=(1, 1, 1))  # Adjust orders based on the summary
arima_results = arima_model.fit()

# Forecast the next 24 hours
forecast = arima_results.forecast(steps=24)
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['value'], label='Historical Data')
plt.plot(pd.date_range(data.index[-1], periods=24, freq='H'), forecast, label='ARIMA Forecast', color='orange')
plt.title('ARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Air Quality Index')
plt.legend()
plt.show()
