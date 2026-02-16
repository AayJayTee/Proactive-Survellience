import torch
import torch.nn as nn


class CNN3D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(3, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, return_features=False):
        feat = self.features(x)
        feat = feat.flatten(1)        # (B, 256)
        logits = self.classifier(feat)

        if return_features:
            return logits, feat

        return logits


# ✅ THIS WAS MISSING
def generate_model(num_classes=101):
    """
    Factory function used by the pipeline
    """
    model = CNN3D(num_classes=num_classes)
    return model
