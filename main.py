import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_squared_error

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv(
    "household_power_consumption.csv",
    sep=';',
    low_memory=False,
    na_values='?'
)

df.columns = df.columns.str.strip()

df['datetime'] = pd.to_datetime(
    df['Date'] + ' ' + df['Time'],
    dayfirst=True
)

df.set_index('datetime', inplace=True)

numeric_cols = [
    'Global_active_power',
    'Global_reactive_power',
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3'
]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

df = df.dropna(subset=['Global_active_power'])

df.rename(columns={'Global_active_power': 'consumption'}, inplace=True)

df = df.resample('D').mean(numeric_only= True)
# -------------------------------
# 2. VISUALIZE DATA
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(df['consumption'])
plt.title("Electricity Consumption")
plt.show()

# -------------------------------
# 3. CHECK STATIONARITY (ADF TEST)
# -------------------------------
result = adfuller(df['consumption'])

print("ADF Statistic:", result[0])
print("p-value:", result[1])

# -------------------------------
# 4. DIFFERENCING (if needed)
# -------------------------------
df_diff = df['consumption'].diff().dropna()

# -------------------------------
# 5. ACF & PACF PLOTS
# -------------------------------
plot_acf(df_diff)
plot_pacf(df_diff)
plt.show()

# -------------------------------
# 6. TRAIN-TEST SPLIT
# -------------------------------
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# -------------------------------
# 7. ARIMA MODEL
# -------------------------------
# Replace (p,d,q) after observing ACF/PACF
arima_model = ARIMA(train['consumption'], order=(2,1,2))
arima_fit = arima_model.fit()

print(arima_fit.summary())

# Predictions
arima_pred = arima_fit.forecast(steps=len(test))

# -------------------------------
# 8. SARIMA MODEL
# -------------------------------
# Example: seasonal period = 7 (weekly)
sarima_model = SARIMAX(train['consumption'],
                       order=(2,1,2),
                       seasonal_order=(1,1,1,7))

sarima_fit = sarima_model.fit()

print(sarima_fit.summary())

sarima_pred = sarima_fit.forecast(steps=len(test))

# -------------------------------
# 9. EVALUATION
# -------------------------------
def evaluate(actual, predicted, name="Model"):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"{name} RMSE:", rmse)

evaluate(test['consumption'], arima_pred, "ARIMA")
evaluate(test['consumption'], sarima_pred, "SARIMA")

# -------------------------------
# 10. PLOT RESULTS
# -------------------------------
plt.figure(figsize=(12,6))
plt.plot(train.index, train['consumption'], label="Train")
plt.plot(test.index, test['consumption'], label="Test")
plt.plot(test.index, arima_pred, label="ARIMA Forecast")
plt.plot(test.index, sarima_pred, label="SARIMA Forecast")

plt.legend()
plt.title("Forecast Comparison")
plt.show()

# -------------------------------
# 11. FUTURE FORECAST
# -------------------------------
future_steps = 30

future_forecast = sarima_fit.forecast(steps=future_steps)

plt.figure(figsize=(10,5))
plt.plot(df.index, df['consumption'], label="Actual")
plt.plot(pd.date_range(df.index[-1], periods=future_steps+1, freq='D')[1:],
         future_forecast, label="Future Forecast")

plt.legend()
plt.title("Future Forecast (30 days)")
plt.show()