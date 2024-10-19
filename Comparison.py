# Evaluate the models (using mean squared error)
arima_forecast = arima_results.forecast(steps=24)
lstm_forecast = lstm_forecast.flatten()  # Flatten to compare with ARIMA

# Calculate MSE for both models
arima_mse = mean_squared_error(data['value'][-24:], arima_forecast)
lstm_mse = mean_squared_error(data['value'][-24:], lstm_forecast)

print(f'ARIMA MSE: {arima_mse}')
print(f'LSTM MSE: {lstm_mse}')
