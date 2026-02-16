import torch
import torch.nn as nn


class Learner(nn.Module):
    def __init__(self, input_dim=2048):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.6),

            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(0.6),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)
