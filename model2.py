from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from tensorflow.keras.layers import (
    Input, LSTM, Dropout, BatchNormalization,
    AdditiveAttention, Dense, Flatten, Permute, Reshape, Multiply
)
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print('TensorFlow version:', tf.__version__)

loaded_data = pd.read_csv('updated_dataset.csv')

features = ['Close', 'Open', 'Range', 'MA20', 'MA50']

removed_date_data = loaded_data.drop(columns=['Day'])
removed_date_data['Return'] = removed_date_data['Close'].pct_change()
removed_date_data = removed_date_data.dropna()

close_scaler = MinMaxScaler(feature_range=(0, 1))
removed_date_data['Close'] = close_scaler.fit_transform(
    removed_date_data['Close'].values.reshape(-1, 1)
)

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(removed_date_data[features])

X = []
y = []

sequence_length = 100

for i in range(sequence_length, len(data_scaled)):
    X.append(data_scaled[i-sequence_length:i])  # shape: (sequence_length, num_features)
    y.append(removed_date_data['Return'].values[i])  # use 'Return' instead of 'Close'

train_size = int(len(X) * 0.8)
test_size = len(X) - train_size

X_train, X_test = np.array(X[:train_size]), np.array(X[train_size:])
y_train, y_test = np.array(y[:train_size]), np.array(y[train_size:])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))


input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

# First LSTM layer
x = LSTM(50, return_sequences=True)(input_layer)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)

# Second LSTM layer
x = LSTM(50, return_sequences=True)(x)

# Attention mechanism
permute = Permute((2, 1))(x)
reshape = Reshape((50, X_train.shape[1]))(permute)

attention = AdditiveAttention(name='attention_weight')([reshape, reshape])
attention = Permute((2, 1))(attention)
attention = Reshape((X_train.shape[1], 50))(attention)

# Multiply attention output with LSTM output
x = Multiply()([x, attention])

# Final layers
x = Flatten()(x)
output = Dense(1)(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10)
csv_logger = keras.callbacks.CSVLogger(
    'training_log.csv')
history = model.fit(X_train, y_train, epochs=20,
                    batch_size=25, validation_split=0.2)

# Convert X_test and y_test to Numpy arrays if they are not already
X_test = np.array(X_test)
y_test = np.array(y_test)

# Ensure X_test is reshaped similarly to how X_train was reshaped
# This depends on how you preprocessed the training data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Now evaluate the model on the test data
test_loss = model.evaluate(X_test, y_test)
print("Test Loss: ", test_loss)

# Making predictions
y_pred = model.predict(X_test)

last_known_price = removed_date_data['Close'].values[-len(y_test)-1]

predicted_prices = [last_known_price]

for ret in y_pred.flatten():
    next_price = predicted_prices[-1] * (1 + ret)
    predicted_prices.append(next_price)

predicted_prices = predicted_prices[1:]

# Plot
plt.figure(figsize=(15, 6))
plt.plot(removed_date_data['Close'].values[-len(y_test):], label='Actual Close Price', color='blue')
plt.plot(predicted_prices, label='Predicted Close Price', color='orange')
plt.legend()
plt.title("Actual vs Predicted Close Prices (Return Prediction)")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.show()