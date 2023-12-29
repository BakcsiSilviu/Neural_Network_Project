import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("coin_Aave.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Extract relevant columns
columns_to_predict = ['Close']

# Use the first 1/3 of the data for training
train_size = int(len(data) * 1/3)
train_data = data.iloc[:train_size]

# Normalize the data
scaler = MinMaxScaler()
train_series_scaled = scaler.fit_transform(train_data[columns_to_predict])

# Convert data to PyTorch tensors
lookback = 1
X_train, y_train = [], []
for i in range(len(train_series_scaled) - lookback):
    X_train.append(train_series_scaled[i:i + lookback])
    y_train.append(train_series_scaled[i + lookback])

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)


# LSTM model
class TimeSeriesPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super(TimeSeriesPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout_rate)  # Add dropout
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = 1  # One prediction at a time
hidden_size = 200
output_size = 1
learning_rate = 0.001
epochs = 300
batch_size = 32

# Model, loss, and optimizer
model = TimeSeriesPredictor(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    model.train()
    for batch_x, batch_y in train_dataloader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y.view(-1, output_size))
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Use the remaining 2/3 of the data for testing
test_data = data.iloc[train_size:]
test_series_scaled = scaler.transform(test_data[columns_to_predict])

# Convert test data to PyTorch tensors
X_test, y_test = [], []
for i in range(len(test_series_scaled) - lookback):
    X_test.append(test_series_scaled[i:i + lookback])
    y_test.append(test_series_scaled[i + lookback])

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Make predictions for the test set
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)

# Inverse transform the predictions
test_predictions = scaler.inverse_transform(test_predictions.numpy().reshape(-1, 1)).flatten()
y_test_original = scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()

# Use the remaining 2/3 of the data for testing
test_data = data.iloc[train_size:]

# Plot the predictions
plt.figure(figsize=(10, 5))
plt.plot(test_data.index[lookback:], y_test_original, label='Actual Close', color='blue')
plt.plot(test_data.index[lookback:], test_predictions, label='Predicted Close', linestyle='dashed', color='red')

# Add the 'Open' column to the graph for better representation
plt.plot(test_data.index[lookback:lookback+len(test_predictions)], test_data['Open'].values[:len(test_predictions)], label='Actual Open', linestyle='dashed', color='green')

plt.title('Close Time Series Prediction')
plt.xlabel('Date')
plt.ylabel('Close')
plt.legend()
plt.show()