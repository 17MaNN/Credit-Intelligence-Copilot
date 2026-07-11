"""Train DocNet on synthetic doc images. Run once at build time: python train.py"""
import torch
from torch import nn, optim
from data import generate
from model import DocNet

X, y = generate()
X_t = torch.tensor(X)
y_t = torch.tensor(y)

model = DocNet()
opt = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

batch_size = 64
n = len(y_t)
for epoch in range(8):
    perm = torch.randperm(n)
    total_loss = 0.0
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        opt.zero_grad()
        pred = model(X_t[idx])
        loss = loss_fn(pred, y_t[idx])
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"epoch {epoch} loss {total_loss:.4f}")

torch.save(model.state_dict(), "doc_model.pt")