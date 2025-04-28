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

loaded_data = pd.read_csv('DataCleaned.csv')

removed_date_data = loaded_data.drop(columns=['Date'])


scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(
    removed_date_data['Close'].values.reshape(-1, 1))

X = []
y = []

sequence_length = 100

for i in range(sequence_length, len(data_scaled)):
    X.append(data_scaled[i-sequence_length:i, 0])
    y.append(data_scaled[i, 0])

train_size = int(len(X) * 0.8)
test_size = len(X) - train_size

X_train, X_test = np.array(X[:train_size]), np.array(X[train_size:])
y_train, y_test = np.array(y[:train_size]), np.array(y[train_size:])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


input_layer = Input(shape=(X_train.shape[1], 1))

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
history = model.fit(X_train, y_train, epochs=100,
                    batch_size=25, validation_split=0.2)

# Convert X_test and y_test to Numpy arrays if they are not already
X_test = np.array(X_test)
y_test = np.array(y_test)

# Ensure X_test is reshaped similarly to how X_train was reshaped
# This depends on how you preprocessed the training data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Now evaluate the model on the test data
test_loss = model.evaluate(X_test, y_test)
print("Test Loss: ", test_loss)

# Making predictions
y_pred = model.predict(X_test)

# Calculating MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred, squared=False)

print("Mean Absolute Error: ", mae)
print("Root Mean Square Error: ", rmse)
