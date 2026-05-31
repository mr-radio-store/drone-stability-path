# ============================================
# Drone Flight Stabilization (Dynamic)
# Pandas + PyTorch + Saved Figures (RPi5 Safe)
# ============================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

# --------------------------------------------
# 1. Generate dynamic drone flight data
# --------------------------------------------
np.random.seed(0)

T = 3000  # time steps
dt = 0.02

roll, pitch = 0.0, 0.0
roll_rate, pitch_rate = 0.0, 0.0

records = []

for _ in range(T):
    # PID-like control target
    m1 = 0.5 - 0.02 * roll - 0.01 * pitch_rate
    m2 = 0.5 + 0.02 * roll - 0.01 * pitch_rate
    m3 = 0.5 - 0.02 * pitch - 0.01 * roll_rate
    m4 = 0.5 + 0.02 * pitch - 0.01 * roll_rate

    # Clip motors
    motors = np.clip([m1, m2, m3, m4], 0.3, 0.7)

    # Simple dynamics
    roll_rate += dt * (motors[1] - motors[0])
    pitch_rate += dt * (motors[3] - motors[2])
    roll += dt * roll_rate
    pitch += dt * pitch_rate

    records.append([roll, pitch, roll_rate, pitch_rate, *motors])

df = pd.DataFrame(
    records,
    columns=["roll", "pitch", "roll_rate", "pitch_rate", "m1", "m2", "m3", "m4"]
)

# --------------------------------------------
# 2. Pandas preprocessing
# --------------------------------------------
df = df[(df["roll"].abs() < 20) & (df["pitch"].abs() < 20)]

X_cols = ["roll", "pitch", "roll_rate", "pitch_rate"]
y_cols = ["m1", "m2", "m3", "m4"]

X = df[X_cols].values.astype(np.float32)
y = df[y_cols].values.astype(np.float32)

# --------------------------------------------
# 3. PyTorch Dataset
# --------------------------------------------
class DroneDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

loader = DataLoader(DroneDataset(X, y), batch_size=64, shuffle=True)

# --------------------------------------------
# 4. Deep Neural Network Controller
# --------------------------------------------
class DroneStabilizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)

model = DroneStabilizer()

# --------------------------------------------
# 5. Training
# --------------------------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_history = []
EPOCHS = 20

for epoch in range(EPOCHS):
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

# --------------------------------------------
# 6. Save training loss figure
# --------------------------------------------
plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss (Drone Stabilizer)")
plt.grid(True)
plt.savefig("training_loss.png")
plt.close()

# --------------------------------------------
# 7. Model inference vs ground truth
# --------------------------------------------
with torch.no_grad():
    pred = model(torch.tensor(X[:200]))

plt.figure(figsize=(8,4))
plt.plot(y[:200, 0], label="True m1")
plt.plot(pred[:, 0], "--", label="Predicted m1")
plt.legend()
plt.title("Motor Output Prediction (m1)")
plt.savefig("motor_prediction.png")
plt.close()

print("\nSaved figures:")
print(" - training_loss.png")
print(" - motor_prediction.png")
