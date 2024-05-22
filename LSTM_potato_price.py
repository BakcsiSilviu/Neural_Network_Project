import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler
from torch.optim import lr_scheduler

# Citirea datelor
df = pd.read_csv("Daily_Retail_Price_of_Potato.csv")
timeseries = df[["Price"]].values.astype("float32")

# Normalizarea datelor
scaler = MinMaxScaler(feature_range=(0, 1))
timeseries_normalized = scaler.fit_transform(timeseries)

# Train-test split
train_size = int(len(timeseries) * 0.67)
test_size = len(timeseries) - train_size
train, test = timeseries_normalized[:train_size], timeseries_normalized[train_size:]

# Funcția de creare a setului de date
def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

lookback = 4
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)

# Modelul cu dropout în LSTM
class AirModel(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True, dropout=dropout_rate)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

# Crearea modelului
model = AirModel()
optimizer = optim.Adam(model.parameters())
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

# Antrenarea modelului
n_epochs = 1
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    if epoch % 10 != 0:
        continue

    # Evaluarea modelului pe setul de antrenare și test
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred_train, y_train))
        y_pred_test = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred_test, y_test))

    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

# Plotarea rezultatelor
with torch.no_grad():
    train_plot = np.ones_like(timeseries_normalized) * np.nan
    y_pred_train = model(X_train)
    y_pred_train = y_pred_train[:, -1, :]
    train_plot[lookback:train_size] = scaler.inverse_transform(y_pred_train.cpu().numpy())
    test_plot = np.ones_like(timeseries_normalized) * np.nan
    y_pred_test = model(X_test)
    y_pred_test = y_pred_test[:, -1, :]
    test_plot[train_size + lookback:len(timeseries)] = scaler.inverse_transform(y_pred_test.cpu().numpy())

# Plotarea rezultatelor finale
plt.plot(scaler.inverse_transform(timeseries_normalized), label='Actual Data')
plt.plot(train_plot, c='r', label='Train Predictions')
plt.plot(test_plot, c='g', label='Test Predictions')

# Adăugarea legendei
plt.legend()

# Afișarea plotului
plt.show()

# Salvarea modelului
torch.save(model.state_dict(), 'air_model.pth')
