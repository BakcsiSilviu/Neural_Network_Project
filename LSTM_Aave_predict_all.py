import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Load the dataset
data = pd.read_csv("coin_Aave.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Extract relevant columns
columns_to_predict = ['High', 'Low', 'Open', 'Close']

# Dictionary to store models and predictions
models = {}
predictions = {}

# Define hyperparameters
input_size = 1
hidden_size = 75
output_size = 1
num_epochs = 500
learning_rate = 0.001

# Create subplots
fig, axs = plt.subplots(len(columns_to_predict), 1, figsize=(10, 5 * len(columns_to_predict)))

# Iterate over each column and train a model
for i, column in enumerate(columns_to_predict):
    # Prepare data for the specific column
    series = data[column].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series)

    # Convert data to PyTorch tensors
    series_tensor = torch.tensor(series_scaled, dtype=torch.float32)

    # Create sequences for training
    lookback = 7  # Adjust as needed
    train_size = int(len(series_scaled) * 0.6)
    X_train, y_train = [], []

    for j in range(train_size - lookback):
        X_train.append(series_scaled[j:j + lookback])
        y_train.append(series_scaled[j + lookback])

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # Define LSTM model
    class TimeSeriesPredictor(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(TimeSeriesPredictor, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

    # Instantiate the model
    model = TimeSeriesPredictor(input_size, hidden_size, output_size)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the trained model
    models[column] = model

    # Make predictions for the entire dataset
    with torch.no_grad():
        model.eval()
        all_data = torch.tensor(series_scaled, dtype=torch.float32).view(1, -1, input_size)
        all_predictions = []

        for j in range(len(series_scaled) - lookback):
            current_input = all_data[:, j:j + lookback, :]
            current_prediction = model(current_input)
            all_predictions.append(current_prediction.item())

        all_predictions = np.array(all_predictions)

    # Inverse transform the predictions
    predictions[column] = scaler.inverse_transform(all_predictions.reshape(-1, 1)).flatten()

    # Plot the predictions
    test_size = len(series_scaled) - train_size
    actual_values = data[column][train_size + lookback:].values
    predicted_values = predictions[column][train_size:]
    min_length = min(len(actual_values), len(predicted_values))

    axs[i].plot(data.index, data[column].values, label='Original', color='blue')
    axs[i].axvline(data.index[train_size], color='gray', linestyle='--', label='End of Training Data')
    axs[i].plot(data.index[train_size + lookback:train_size + lookback + min_length], actual_values[:min_length], label='Actual', color='green')
    axs[i].plot(data.index[train_size + lookback:train_size + lookback + min_length], predicted_values[:min_length], label='Predicted', linestyle='dashed', color='red')
    axs[i].set_xlabel('Date')
    axs[i].set_ylabel(column)
    axs[i].legend()

plt.tight_layout()
plt.show()
