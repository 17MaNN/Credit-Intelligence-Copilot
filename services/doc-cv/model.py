"""Tiny CNN for 4-class doc-type classification. Single definition imported
by both train.py and main.py so architecture never drifts."""
import torch.nn as nn

N_CLASSES = 4


class DocNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64->32
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32->16
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 32),
            nn.ReLU(),
            nn.Linear(32, N_CLASSES),
        )

    def forward(self, x):
        return self.fc(self.conv(x))