from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load data ===
df = pd.read_csv('dataset/AAPL_CleanedStockData.csv')
df = df[df['Date'] >= '2015-01-01'].reset_index(drop=True)

features = ['Close', 'Open', 'Range', 'MA20', 'MA50']
target = 'Close'
sequence_length = 100

# === Create target column for next-day prediction ===
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# === Scale ===
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

scaled_features = feature_scaler.fit_transform(df[features])
scaled_target = target_scaler.fit_transform(df[['Target']])

# === Use only the last day of each 100-day window as input ===
X, y, dates = [], [], []
for i in range(sequence_length, len(df) - 1, 5):  # step=5 reduces redundancy
    X.append(scaled_features[i])  # only last day of window
    y.append(scaled_target[i])
    dates.append(df['Date'].iloc[i + 1])

X, y = np.array(X), np.array(y).ravel()
dates = np.array(dates)

# === Train/Test split ===
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_test = dates[split:]

# === Train SVR ===
svr = SVR(kernel='rbf', C=100, gamma=0.01, epsilon=0.01)
svr.fit(X_train, y_train)

# === Predict and invert scale ===
y_pred_scaled = svr.predict(X_test)
y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test = target_scaler.inverse_transform(y_test.reshape(-1, 1))

# === Evaluate ===
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"SVR MAE: ${mae:.2f}, RMSE: ${rmse:.2f}")

# === Plot ===
# === Plot ===
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual', linewidth=2)
plt.plot(y_pred, label='SVR Prediction', linewidth=2)
plt.title('SVR Prediction vs Actual Close Price')
plt.xlabel('Time Step')
plt.ylabel('Close Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
