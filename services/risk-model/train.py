"""Train RiskNet and save weights + feature normalization stats.
Run once at build time: python train.py"""
import json
import torch
from torch import nn, optim
from data import generate
from model import RiskNet

X, y = generate()
mean, std = X.mean(axis=0), X.std(axis=0)
X_norm = (X - mean) / std

X_t = torch.tensor(X_norm)
y_t = torch.tensor(y).unsqueeze(1)

model = RiskNet()
opt = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

for epoch in range(50):
    opt.zero_grad()
    pred = model(X_t)
    loss = loss_fn(pred, y_t)
    loss.backward()
    opt.step()

print(f"final loss: {loss.item():.4f}")

torch.save(model.state_dict(), "risk_model.pt")
with open("norm_stats.json", "w") as f:
    json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
