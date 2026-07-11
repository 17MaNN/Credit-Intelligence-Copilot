"""Tiny tabular NN. Single definition imported by both train.py and main.py
so architecture never drifts between training and serving."""
import torch.nn as nn

N_FEATURES = 5


class RiskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
