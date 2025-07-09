import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load dataset
df = pd.read_csv("AAPL.csv")  # Replace with your filename
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
data = df[['Close']].copy()

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Sequence creator
SEQ_LEN = 60
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(seq_len, len(data)):
        xs.append(data[i-seq_len:i])
        ys.append(data[i])
    return np.array(xs), np.array(ys)

# Split data
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

X_train, y_train = create_sequences(train_data, SEQ_LEN)
X_test, y_test = create_sequences(test_data, SEQ_LEN)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Dataset and loader
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=64, shuffle=False)

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Use output of last time step
        out = self.fc(out)
        return out

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
model.train()
for epoch in range(20):
    epoch_loss = 0
    for seqs, targets in train_loader:
        seqs, targets = seqs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(seqs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch [{epoch+1}/20], Loss: {epoch_loss:.4f}")

# Evaluation
model.eval()
predictions = []
actual = []

with torch.no_grad():
    for seqs, targets in test_loader:
        seqs = seqs.to(device)
        output = model(seqs)
        predictions.extend(output.cpu().numpy())
        actual.extend(targets.numpy())

# Inverse scale
predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
actual_prices = scaler.inverse_transform(np.array(actual).reshape(-1, 1))

# Plot
test_dates = df.index[-len(actual_prices):]

plt.figure(figsize=(12,6))
plt.plot(test_dates, actual_prices, label='Actual Price')
plt.plot(test_dates, predicted_prices, label='Predicted Price')
plt.title('Apple Stock Price Prediction using PyTorch LSTM')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.show()

# RMSE
rmse = math.sqrt(mean_squared_error(actual_prices, predicted_prices))
print(f"Test RMSE: {rmse:.2f}")
