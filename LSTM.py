# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['value'].values.reshape(-1, 1))

# Prepare the data for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 24  # Using the last 24 hours for prediction
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)
# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=100, batch_size=32)

# Forecast using the LSTM model
lstm_input = scaled_data[-time_step:].reshape(1, time_step, 1)
lstm_forecast = model.predict(lstm_input)
lstm_forecast = scaler.inverse_transform(lstm_forecast)  # Inverse transform to original scale
# Plotting the LSTM Forecast
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['value'], label='Historical Data')
plt.plot(pd.date_range(data.index[-1], periods=24, freq='H'), lstm_forecast, label='LSTM Forecast', color='red')
plt.title('LSTM Forecast')
plt.xlabel('Date')
plt.ylabel('Air Quality Index')
plt.legend()
plt.show()
